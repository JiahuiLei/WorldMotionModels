"""ManiSkill sim rollout evaluator for UMM.

Bridges between UMM's SE(3) trace representation and ManiSkill's
7D action space (delta_pos + delta_rot + gripper).  The UMM model
acts as a policy in a receding-horizon loop:

    sim obs -> extract robot/object SE(3) -> build model input
    -> model.inference() -> predicted ACT traces
    -> tgt_to_state_action(current, target) -> 7D action -> env.step

Key conversion:
    Training encodes actions as target SE(3) via ``state_action_to_tgt``.
    Sim rollout reverses this via ``tgt_to_state_action``.

Entity layout matches training data (``maniskill_oracle_dataset.py``):
    [objects..., EE, LEFT_FINGER, RIGHT_FINGER, EE_ACT, LEFT_ACT, RIGHT_ACT]
"""

import gc
import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from lib_mani.utils.action_utils import tgt_to_state_action
from lib_mani.sapien_robot_wrapper import RobotKinematicsConverter
from lib_io import UnifiedIO

from umm.datasets.umm_v11.dataset_config import (
    MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
    SEM_MANISKILL_TABLETOP_ACT,
    SEM_MANISKILL_TABLETOP_EE,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    SEM_OBJECT,
    UID_MANISKILL_TABLETOP_ACT,
    UID_MANISKILL_TABLETOP_EE,
    UID_MANISKILL_TABLETOP_LEFT_FINGER,
    UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
)
from umm.datasets.umm_v11.dataset_utils import sample_list


def discover_link_indices(robot) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Find EE (hand), left finger, right finger link indices by name.

    Falls back to common Panda indices if name matching fails.

    Args:
        robot: SAPIEN robot instance.

    Returns:
        (ee_idx, left_finger_idx, right_finger_idx) link indices.
    """
    links = robot.get_links()
    ee_idx, left_idx, right_idx = None, None, None
    link_names = []
    for i, link in enumerate(links):
        name = link.name.lower()
        link_names.append(link.name)
        if ("hand" in name or "tcp" in name) and "finger" not in name:
            ee_idx = i
        elif "left" in name and "finger" in name and "pad" not in name:
            left_idx = i
        elif "right" in name and "finger" in name and "pad" not in name:
            right_idx = i

    if ee_idx is None or left_idx is None or right_idx is None:
        logging.warning(
            f"Link index discovery incomplete: ee={ee_idx}, left={left_idx}, "
            f"right={right_idx}. Links: {link_names}"
        )
    return ee_idx, left_idx, right_idx


def _load_task_sem_info(data_root: str) -> Optional[dict]:
    """Load sem_type_uid from the first state file of a task.

    Returns dict with keys like 'endeffector', 'objects', etc.
    Each value is a list of local UIDs (link indices).
    """
    state_dir = osp.join(data_root, "states")
    if not osp.exists(state_dir):
        return None
    files = sorted(os.listdir(state_dir))
    if not files:
        return None
    state_data = np.load(
        osp.join(state_dir, files[0]), allow_pickle=True
    )
    if "sem_type_uid" in state_data:
        return state_data["sem_type_uid"].item()
    return None


class ManiskillSimRollout:
    """Runs ManiSkill sim rollouts using a UMM model as policy.

    Creates per-task ManiSkill gym environments, runs the model in a
    receding-horizon loop (inference every ``deploy_chunk_size`` steps),
    converts predicted SE(3) ACT traces to 7D actions, and records
    sim-rendered videos with success metrics.

    Args:
        cfg: Full experiment config dict.
        video_root: Directory for rollout videos.
        device: Torch device for model inference.
    """

    def __init__(
        self,
        cfg: dict,
        video_root: Optional[str] = None,
        device: Optional[torch.device] = None,
        video_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        # Accept either video_root or video_dir (callback passes video_dir)
        self.video_root = video_root or video_dir or "/tmp/sim_rollout"
        self.device = device or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        vis_cfg = cfg.get("visualize", {}).get("maniskill", {})
        self.deploy_chunk_size = vis_cfg.get("deploy_chunk_size", 10)
        self.deploy_hist_len = vis_cfg.get("deploy_hist_len", 5)

        robot_env_cfg = cfg.get("robot_env", {})
        self.max_episode_steps = robot_env_cfg.get("max_episode_steps", 200)
        self.num_eval_episodes = robot_env_cfg.get("num_eval_episodes", 2)
        self.render_H = robot_env_cfg.get("H", 256)
        self.render_W = robot_env_cfg.get("W", 256)
        self.safe_run = robot_env_cfg.get("safe_run", True)

        self.task_list = self._get_task_list()
        self.max_T = self._get_dataset_param("max_T", 30)
        self.num_pts = self._get_dataset_param("num_pts", 256)
        self.base_fps = self._get_dataset_param("base_fps", 10.0)
        self.t_rate = self._get_dataset_param("t_rate", 1)

        self.io = UnifiedIO()

        # Pre-load per-task data (PCLs, sem_type_uid, global UIDs)
        self.task_data: Dict[str, dict] = {}
        ms_cfg = cfg.get("model", {}).get("maniskill", {})
        ds_cfgs = cfg.get("dataset", {}).get("configs", [])
        for task_name in self.task_list:
            task_ms = ms_cfg.get(task_name, {})
            # Find dataset config for this task to get data_root
            data_root = None
            for dc in ds_cfgs:
                if dc.get("type") == "maniskill_oracle":
                    dr = dc["data_root"]
                    if osp.basename(osp.normpath(dr)) == task_name:
                        data_root = dr
                        break
            self.task_data[task_name] = self._load_task_data(
                task_name, task_ms, data_root
            )

        logging.info(
            f"ManiskillSimRollout: {len(self.task_list)} tasks, "
            f"chunk={self.deploy_chunk_size}, hist={self.deploy_hist_len}, "
            f"max_T={self.max_T}"
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get_task_list(self) -> List[str]:
        """Extract task names for sim evaluation.

        If ``robot_env.eval_tasks`` is set in the config, use that explicit
        list (validated against dataset configs).  Otherwise fall back to
        all ``maniskill_oracle`` datasets.
        """
        robot_env_cfg = self.cfg.get("robot_env", {})
        explicit = robot_env_cfg.get("eval_tasks", None)

        # All maniskill_oracle tasks in the dataset config (ground truth).
        all_ds_tasks = []
        for dc in self.cfg.get("dataset", {}).get("configs", []):
            if dc.get("type") == "maniskill_oracle":
                all_ds_tasks.append(osp.basename(osp.normpath(dc["data_root"])))

        if explicit is not None:
            # Validate: every eval_task must appear in the dataset.
            unknown = [t for t in explicit if t not in all_ds_tasks]
            if unknown:
                raise ValueError(
                    f"robot_env.eval_tasks contains tasks not in dataset: {unknown}. "
                    f"Available: {all_ds_tasks}"
                )
            logging.info(
                f"ManiskillSimRollout: eval_tasks override — "
                f"{len(explicit)}/{len(all_ds_tasks)} tasks selected"
            )
            return list(explicit)

        return all_ds_tasks

    def _get_dataset_param(self, key: str, default):
        """Read a param from the first maniskill_oracle dataset config."""
        for dc in self.cfg.get("dataset", {}).get("configs", []):
            if dc.get("type") == "maniskill_oracle":
                return dc.get(key, default)
        return default

    def _load_task_data(
        self, task_name: str, ms_cfg: dict, data_root: Optional[str]
    ) -> dict:
        """Load robot/object PCLs and sem info for a task."""
        robot_fn = ms_cfg.get("model_path", "")
        object_fn = ms_cfg.get("object_path", "")

        data: dict = {
            "task_obj_uids": MANISKILL_TABLETOP_TASK_OBJ_UID_MAP.get(
                task_name, []
            ),
            "robot_link_data": None,
            "object_link_data": None,
            "sem_type_uid": None,
            "num_objects": 0,
            "object_local_uids": [],
        }

        if robot_fn and osp.exists(robot_fn):
            data["robot_link_data"] = self.io.load_npzs(robot_fn)
        else:
            logging.warning(f"Robot mesh missing for {task_name}: {robot_fn}")

        if object_fn and osp.exists(object_fn):
            data["object_link_data"] = self.io.load_npzs(object_fn)
        else:
            logging.warning(f"Object mesh missing for {task_name}: {object_fn}")

        if data_root:
            sem_info = _load_task_sem_info(data_root)
            if sem_info is not None:
                data["sem_type_uid"] = sem_info
                obj_uids = sem_info.get("objects", [])
                data["num_objects"] = len(obj_uids)
                data["object_local_uids"] = obj_uids
                logging.info(
                    f"  {task_name}: {len(obj_uids)} objects, "
                    f"ee={sem_info.get('endeffector')}, "
                    f"fingers=L{sem_info.get('gripper_left_finger')}"
                    f"/R{sem_info.get('gripper_right_finger')}"
                )
            else:
                logging.error(
                    f"  {task_name}: sem_type_uid not found in state files. "
                    f"Evaluation will run with 0 object entities."
                )
        else:
            logging.error(
                f"  {task_name}: data_root not found. "
                f"Cannot load sem_type_uid for entity construction."
            )

        return data

    # ------------------------------------------------------------------
    # Entity attribute builders
    # ------------------------------------------------------------------

    def _build_entity_attrs(
        self, task_name: str, num_objects: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build sem_type, uid, group_id, local_pcl tensors.

        Entity order: [objects..., EE, LEFT, RIGHT, EE_ACT, LEFT_ACT, RIGHT_ACT]

        Returns:
            (sem_type, uid, group_id, local_pcl) each with leading dim P.
        """
        td = self.task_data[task_name]
        task_obj_uids = td["task_obj_uids"]
        obj_local_uids = td.get("object_local_uids", list(range(num_objects)))
        P = num_objects + 6

        # -- sem_type --
        sem = [SEM_OBJECT] * num_objects + [
            SEM_MANISKILL_TABLETOP_EE,
            SEM_MANISKILL_TABLETOP_LEFT_FINGER,
            SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
            SEM_MANISKILL_TABLETOP_ACT,
            SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
        ]
        sem_type = torch.tensor(sem, dtype=torch.long)

        # -- uid --
        u = []
        for i, local_uid in enumerate(obj_local_uids):
            idx = int(local_uid)
            if idx < len(task_obj_uids):
                u.append(task_obj_uids[idx])
            else:
                fallback_uid = task_obj_uids[i] if i < len(task_obj_uids) else 0
                logging.warning(
                    f"Object UID fallback: local_uid={local_uid} out of range "
                    f"for {len(task_obj_uids)} task UIDs, using {fallback_uid}"
                )
                u.append(fallback_uid)
        u += [
            UID_MANISKILL_TABLETOP_EE,
            UID_MANISKILL_TABLETOP_LEFT_FINGER,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER,
            UID_MANISKILL_TABLETOP_ACT,
            UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
        ]
        uid = torch.tensor(u, dtype=torch.long)

        # -- group_id --
        gid = torch.zeros(P, dtype=torch.long)
        gid[num_objects:] = 1  # robot group

        # -- local_pcl --
        pcl = self._build_pcl(task_name, num_objects, obj_local_uids)

        return sem_type, uid, gid, pcl

    def _build_pcl(
        self, task_name: str, num_objects: int, obj_local_uids: List[int]
    ) -> torch.Tensor:
        """Build (P, num_pts, 6) point-cloud tensor (xyz+rgb)."""
        td = self.task_data[task_name]
        robot_data = td.get("robot_link_data")
        object_data = td.get("object_link_data")
        sem_info = td.get("sem_type_uid")
        P = num_objects + 6
        pcl_parts: List[torch.Tensor] = []

        # Object PCLs
        for local_uid in obj_local_uids:
            pcl_parts.append(
                self._read_entity_pcl(object_data, local_uid)
            )

        # Robot PCLs: use local UIDs from sem_type_uid
        ee_uid = 10
        left_uid = 12
        right_uid = 14
        if sem_info is not None:
            ee_uid = sem_info.get("endeffector", [10])[0]
            left_uid = sem_info.get("gripper_left_finger", [12])[0]
            right_uid = sem_info.get("gripper_right_finger", [14])[0]

        robot_uids = [ee_uid, left_uid, right_uid]
        for ruid in robot_uids:
            pcl_parts.append(self._read_entity_pcl(robot_data, ruid))
        # ACT PCLs = copy of state PCLs
        for ruid in robot_uids:
            pcl_parts.append(self._read_entity_pcl(robot_data, ruid))

        assert len(pcl_parts) == P, f"PCL parts {len(pcl_parts)} != P={P}"
        return torch.stack(pcl_parts, dim=0)  # (P, num_pts, 6)

    def _read_entity_pcl(
        self, link_data: Optional[dict], local_uid: int
    ) -> torch.Tensor:
        """Read xyz+color PCL for a single entity, zero-pad if missing."""
        if link_data is None:
            return torch.zeros(self.num_pts, 6)
        key = f"{local_uid}_pcl"
        if key not in link_data:
            return torch.zeros(self.num_pts, 6)
        pts = link_data[key]  # (N, 3)
        # Color
        c_key = f"{local_uid}_c"
        pc_key = f"{local_uid}_pcl_c"
        if c_key in link_data:
            c = link_data[c_key].copy()
            if c.max() > 1.0:
                c = c / 255.0
            c = np.tile(c[None, :], (pts.shape[0], 1)).astype(np.float32)
        elif pc_key in link_data:
            c = link_data[pc_key]
        else:
            c = np.ones((pts.shape[0], 3), dtype=np.float32) * 0.5
        pcl = np.concatenate([pts, c], axis=1)  # (N, 6)
        pcl = sample_list(pcl, self.num_pts, random_flag=False)
        return torch.as_tensor(pcl, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Object pose helpers
    # ------------------------------------------------------------------

    def _get_object_poses_from_env(
        self, base_env, num_objects: int
    ) -> List[torch.Tensor]:
        """Extract object SE(3) poses from the ManiSkill env.

        Tries common actor attribute patterns. Returns list of (4,4) tensors.
        Falls back to identity if discovery fails.
        """
        poses: List[torch.Tensor] = []

        # Try common ManiSkill v3 actor attributes
        candidate_attrs = [
            "obj", "cube", "cubeA", "cubeB", "peg", "charger",
            "tool", "ball", "target_object",
        ]
        for attr in candidate_attrs:
            if len(poses) >= num_objects:
                break
            if hasattr(base_env, attr):
                actor = getattr(base_env, attr)
                if hasattr(actor, "pose"):
                    T = actor.pose.to_transformation_matrix()
                    if T.dim() == 3:
                        T = T.squeeze(0)
                    poses.append(T.float().detach())

        # If we found fewer than expected, pad with identity
        while len(poses) < num_objects:
            poses.append(torch.eye(4, dtype=torch.float32))

        # Truncate if more than expected
        return poses[:num_objects]

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        """Run evaluation across all configured tasks.

        Args:
            model: UMM AgentV11 instance (nn.Module).

        Returns:
            Aggregated metrics dict with per-task success rates and
            ``overall_success_rate``.
        """
        all_results: Dict[str, dict] = {}
        for task_name in self.task_list:
            logging.info(f"ManiskillSimRollout: evaluating {task_name}")
            try:
                task_results = self._evaluate_task(model, task_name)
                all_results[task_name] = task_results
            except Exception as e:
                logging.error(
                    f"Sim evaluation failed for {task_name}: {e}",
                    exc_info=True,
                )
                all_results[task_name] = {"error": str(e)}

        return self._aggregate_results(all_results)

    def _evaluate_task(self, model, task_name: str) -> dict:
        """Run closed-loop rollout for one task."""
        import gymnasium as gym
        from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper

        td = self.task_data[task_name]
        num_objects = td["num_objects"]
        P = num_objects + 6

        video_dir = osp.join(self.video_root, task_name)
        os.makedirs(video_dir, exist_ok=True)

        # -- Create env --
        env = gym.make(
            task_name,
            control_mode="pd_ee_delta_pose",
            reward_mode="sparse",
            obs_mode="state_dict",
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            enable_shadow=False,
            render_backend="gpu",
            sensor_configs=dict(width=self.render_W, height=self.render_H),
            max_episode_steps=self.max_episode_steps,
            reconfiguration_freq=1,
        )
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            info_on_video=True,
            source_type="act",
        )

        try:
            # -- Robot setup --
            base_env = env.unwrapped
            robot = base_env.agent.robot
            kin_converter = RobotKinematicsConverter(robot)
            ee_idx, left_idx, right_idx = discover_link_indices(robot)

            if ee_idx is None or left_idx is None or right_idx is None:
                raise RuntimeError(
                    f"Cannot find EE/finger links for {task_name}"
                )

            # -- Entity attributes (constant across episodes) --
            sem_type, uid, group_id, local_pcl = self._build_entity_attrs(
                task_name, num_objects
            )

            metrics = defaultdict(list)

            for ep_idx in range(self.num_eval_episodes):
                ep_metrics = self._run_episode(
                    env=env,
                    model=model,
                    kin_converter=kin_converter,
                    ee_idx=ee_idx,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    num_objects=num_objects,
                    P=P,
                    sem_type=sem_type,
                    uid=uid,
                    group_id=group_id,
                    local_pcl=local_pcl,
                    task_name=task_name,
                    episode_idx=ep_idx,
                )
                for k, v in ep_metrics.items():
                    metrics[k].append(v)

            return dict(metrics)

        finally:
            try:
                env.close()
            except Exception as e:
                logging.warning(f"env.close() failed for {task_name}: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    def _run_episode(
        self,
        env,
        model,
        kin_converter: RobotKinematicsConverter,
        ee_idx: int,
        left_idx: int,
        right_idx: int,
        num_objects: int,
        P: int,
        sem_type: torch.Tensor,
        uid: torch.Tensor,
        group_id: torch.Tensor,
        local_pcl: torch.Tensor,
        task_name: str,
        episode_idx: int,
    ) -> dict:
        """Run a single closed-loop episode.

        Uses receding-horizon: call inference every ``deploy_chunk_size``
        steps, extract ACT trace targets, convert to actions using actual
        current poses at each sim step.

        Returns:
            Dict with ``success_once`` (float 0/1) and ``total_steps``.
        """
        obs, info = env.reset()
        trace_buffer: List[torch.Tensor] = []  # list of (P, 4, 4)
        action_targets: Optional[torch.Tensor] = None  # (chunk, 3, 4, 4)
        chunk_step = 0
        success = False
        last_step = 0

        for step in range(self.max_episode_steps):
            # -- Extract current poses from sim --
            qpos = obs["agent"]["qpos"]
            qpos_t = torch.as_tensor(qpos, dtype=torch.float32)
            if qpos_t.dim() == 2:
                qpos_t = qpos_t.squeeze(0)
            link_poses = kin_converter.qpos_to_link_poses(qpos_t)

            ee_pose = link_poses[ee_idx]      # (4, 4)
            left_pose = link_poses[left_idx]  # (4, 4)
            right_pose = link_poses[right_idx]  # (4, 4)

            # Object poses from env
            obj_poses = self._get_object_poses_from_env(
                env.unwrapped, num_objects
            )

            # -- Build current-step state: (P, 4, 4) --
            cur_state = torch.eye(4).unsqueeze(0).expand(P, 4, 4).clone()
            for i, op in enumerate(obj_poses):
                cur_state[i] = op.cpu() if op.is_cuda else op
            cur_state[num_objects] = ee_pose
            cur_state[num_objects + 1] = left_pose
            cur_state[num_objects + 2] = right_pose
            # ACT entities in history = copy of state poses.
            # During training, ACT carries ground-truth future targets shifted
            # by one step (ACT[t] ~ state[t+1]). During deployment we don't have
            # future targets, so current state is the best proxy. The model's
            # predicted future ACT trace is what we actually use for actions.
            cur_state[num_objects + 3] = ee_pose.clone()
            cur_state[num_objects + 4] = left_pose.clone()
            cur_state[num_objects + 5] = right_pose.clone()

            trace_buffer.append(cur_state)

            # -- Inference every chunk_size steps --
            if step % self.deploy_chunk_size == 0:
                action_targets = self._run_inference(
                    model=model,
                    trace_buffer=trace_buffer,
                    sem_type=sem_type,
                    uid=uid,
                    group_id=group_id,
                    local_pcl=local_pcl,
                    task_name=task_name,
                    num_objects=num_objects,
                    step=step,
                )  # (chunk, 3, 4, 4) for [EE_ACT, LEFT_ACT, RIGHT_ACT]
                chunk_step = 0

            # -- Convert target pose to action --
            if action_targets is not None and chunk_step < action_targets.shape[0]:
                target_ee = action_targets[chunk_step, 0]    # (4, 4)
                target_left = action_targets[chunk_step, 1]  # (4, 4)
                target_right = action_targets[chunk_step, 2]  # (4, 4)

                # All poses must be CPU for tgt_to_state_action
                action_7d = tgt_to_state_action(
                    current_palm_pose=ee_pose.cpu().unsqueeze(0),
                    target_palm_pose=target_ee.cpu().unsqueeze(0),
                    target_left_finger_pose=target_left.cpu().unsqueeze(0),
                    target_right_finger_pose=target_right.cpu().unsqueeze(0),
                )  # (1, 7)
                action = action_7d.squeeze(0).cpu().numpy()
            else:
                action = np.zeros(7, dtype=np.float32)

            chunk_step += 1

            # -- Step sim --
            obs, rew, terminated, truncated, info = env.step(action)

            # Track success from episode-level metrics (CPUGymWrapper tracks
            # cumulative success_once even with ignore_terminations=True)
            if isinstance(info, dict):
                ep_info = info.get("episode", {})
                if ep_info.get("success_once", False):
                    success = True
                elif not success:
                    success = bool(info.get("success", False))

            last_step = step

            if terminated or truncated:
                break

        return {
            "success_once": float(success),
            "total_steps": last_step + 1,
        }

    def _run_inference(
        self,
        model,
        trace_buffer: List[torch.Tensor],
        sem_type: torch.Tensor,
        uid: torch.Tensor,
        group_id: torch.Tensor,
        local_pcl: torch.Tensor,
        task_name: str,
        num_objects: int,
        step: int = 0,
    ) -> Optional[torch.Tensor]:
        """Run model inference and return ACT target poses.

        Follows the langtable rollout pattern: pad history to max_T,
        mark history as known, future as unknown, run inference, extract
        predicted ACT traces.

        Returns:
            (chunk_size, 3, 4, 4) tensor of target poses for
            [EE_ACT, LEFT_ACT, RIGHT_ACT], or None on failure.
        """
        device = self.device

        hist_len = min(len(trace_buffer), self.deploy_hist_len)
        hist = trace_buffer[-hist_len:]
        hist_trace = torch.stack(hist, dim=0)  # (H, P, 4, 4)
        H, P = hist_trace.shape[:2]

        # Pad to max_T
        pad_len = self.max_T - H
        if pad_len < 0:
            # History longer than window -- take last max_T
            hist_trace = hist_trace[-self.max_T:]
            H = self.max_T
            pad_len = 0

        if pad_len > 0:
            pad_trace = (
                torch.eye(4)
                .view(1, 1, 4, 4)
                .expand(pad_len, P, 4, 4)
                .clone()
            )
            trace = torch.cat([hist_trace, pad_trace], dim=0)  # (T, P, 4, 4)
        else:
            trace = hist_trace

        # Masks
        trace_mask = torch.ones(self.max_T, P)
        trace_known_mask = torch.zeros(self.max_T, P)
        trace_known_mask[:H] = 1.0  # History = known

        # Context: no goal state in deployment
        context = (
            torch.eye(4).view(1, 1, 4, 4).expand(1, P, 4, 4).clone()
        )
        context_mask = torch.ones(1, P)
        context_known_mask = torch.zeros(1, P)

        # Timestamps: match training convention (arange - current_ith) * t_rate / fps
        time_unit = float(self.t_rate) / float(self.base_fps)
        realtime = (
            torch.arange(self.max_T, dtype=torch.float32) - H + 1
        ) * time_unit

        try:
            gen_ctx, gen_trace = model.inference(
                trace=trace[None].to(device),
                context=context[None].to(device),
                trace_known_mask=trace_known_mask[None].bool().to(device),
                context_known_mask=context_known_mask[None].bool().to(device),
                trace_mask=trace_mask[None].bool().to(device),
                context_mask=context_mask[None].bool().to(device),
                real_timestep=realtime[None].to(device),
                group_id=group_id[None].to(device),
                sem_type=sem_type[None].to(device),
                rot_mask=torch.ones(1, P).bool().to(device),
                local_pcl=local_pcl[None].to(device),
                local_pcl_mask=torch.ones(1, P).bool().to(device),
                text=[task_name],
                seed=42 + step,
                verbose=False,
            )
        except Exception as e:
            logging.error(f"Inference failed for {task_name}: {e}")
            return None

        # gen_trace: (1, T, P, 4, 4) -> (T, P, 4, 4)
        gen_trace = gen_trace[0].cpu()

        # Extract ACT traces for future steps
        act_ee_idx = num_objects + 3
        act_left_idx = num_objects + 4
        act_right_idx = num_objects + 5

        future_start = H
        future_end = min(H + self.deploy_chunk_size, self.max_T)
        chunk_len = future_end - future_start

        if chunk_len <= 0:
            logging.warning(
                f"No future steps: H={H}, max_T={self.max_T}"
            )
            return None

        # (chunk_len, 3, 4, 4) for [EE_ACT, LEFT_ACT, RIGHT_ACT]
        targets = torch.stack(
            [
                gen_trace[future_start:future_end, act_ee_idx],
                gen_trace[future_start:future_end, act_left_idx],
                gen_trace[future_start:future_end, act_right_idx],
            ],
            dim=1,
        )

        return targets

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(self, all_results: Dict[str, dict]) -> dict:
        """Flatten per-task results into callback-friendly dict."""
        agg: Dict[str, float] = {}
        success_rates = []
        for task, results in all_results.items():
            if "error" in results:
                agg[f"{task}/error"] = 1.0
                continue
            if "success_once" in results:
                sr = float(np.mean(results["success_once"]))
                agg[f"{task}/success_rate"] = sr
                success_rates.append(sr)
            if "total_steps" in results:
                agg[f"{task}/avg_steps"] = float(
                    np.mean(results["total_steps"])
                )

        if success_rates:
            agg["success_once"] = float(np.mean(success_rates))

        return agg
