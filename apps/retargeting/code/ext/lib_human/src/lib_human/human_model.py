import torch
import torch.nn as nn
from lib_human.smplx_customized import create
from lib_human.smplx_customized.lbs import blend_shapes
import numpy as np
from pathlib import Path


class HumanModel(nn.Module):
    """
    A wrapper around the smplx library to support different versions of SMPL family models.
    Supports SMPL, SMPL-H, SMPL-X models specified by model_path.

    ! warning: current model do not use pose dependent shape deformation!!

    # ! if we bind the global pose into the As, the W must sum to 1! now default smpl skinning is normalized

    """

    def __init__(
        self,
        model_path,
        canonical_pose="zero",
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        device="cpu",
        is_rhand=True,  # For MANO: whether it's right hand (True) or left hand (False)
    ):
        """
        Initialize the HumanModel.

        Args:
            model_path (str): Path to the model files directory or specific model file
            canonical_pose (str): Type of canonical pose ('zero' or 't-pose' or 'a-pose')
            model_type (str): Type of model ('smpl', 'smplh', 'smplx', 'mano')
            gender (str): Gender of the model ('neutral', 'male', 'female')
            num_betas (int): Number of shape parameters
            use_pca (bool): Whether to use PCA for hand pose (MANO only)
            flat_hand_mean (bool): Whether to use flat hand mean pose (MANO only)
            device (str): Device to run the model on ('cpu' or 'cuda')
            is_rhand (bool): For MANO only - whether it's right hand (True) or left hand (False)
        """
        super().__init__()
        self.model_path = Path(model_path)
        self.canonical_pose = canonical_pose
        self.model_type = model_type.lower()
        self.gender = gender
        self.num_betas = num_betas
        self.device = torch.device(device)
        self.is_rhand = is_rhand

        # assert model_type != "smpl", "DO NOT USE SMPL FOR EVERY!!!!"

        # Create model-specific kwargs
        create_kwargs = {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "num_betas": self.num_betas,
            "batch_size": 1,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }

        # Add model-specific parameters
        if self.model_type == "mano":
            create_kwargs.update(
                {
                    "is_rhand": self.is_rhand,
                    "use_pca": use_pca,
                    "flat_hand_mean": flat_hand_mean,
                    "create_hand_pose": False,
                }
            )
        else:
            create_kwargs.update(
                {
                    "gender": self.gender,
                    "use_pca": use_pca,
                    "flat_hand_mean": flat_hand_mean,
                    "create_body_pose": False,
                    "create_left_hand_pose": False,
                    "create_right_hand_pose": False,
                    "create_jaw_pose": False,
                    "create_leye_pose": False,
                    "create_reye_pose": False,
                    "create_expression": False,
                }
            )

        # Create the model without creating parameters
        self.model = create(**create_kwargs).to(self.device)  # type: ignore

        # Get faces (topology is constant)
        self.faces = torch.from_numpy(self.model.faces.astype(np.int64)).to(self.device)

        # Store number of joints
        self.num_joints = self.model.J_regressor.shape[0]

        self.register_buffer(
            "canonical_pose_params", self.__get_canonical_pose_params__()
        )  # 1,C

        self.pose_dim = self.num_joints * 3

    def __repr__(self):
        return f"HumanModel(model_path={self.model_path}, model_type={self.model_type}, gender={self.gender}, num_betas={self.num_betas}, device={self.device})\n[num_joints={self.num_joints}, num_faces={self.faces.shape[0]}, num_betas={self.num_betas}]"

    def __get_canonical_pose_params__(self, batch_size=1):
        """Get canonical pose parameters based on canonical_pose setting.
        Returns full pose parameters including global orientation.

        Args:
            batch_size (int): Batch size for the pose parameters

        Returns:
            torch.Tensor: Full pose parameters [batch_size, total_dims]
                - SMPL: [batch_size, 72] (3 global + 69 body)
                - SMPL-H: [batch_size, 66] (3 global + 63 body)
                - SMPL-X: [batch_size, 66] (3 global + 63 body)
                - MANO: [batch_size, 48] (3 global + 45 hand)
        """
        # Determine the correct body_pose dimensions based on model type
        if self.model_type == "smpl":
            # SMPL: 23 body joints * 3 = 69 dims
            body_pose_dim = 69
            left_shoulder_idx = 16  # SMPL joint indices
            right_shoulder_idx = 17
        elif self.model_type in ["smplh", "smplx"]:
            # SMPL-H and SMPL-X: 21 body joints * 3 = 63 dims
            body_pose_dim = 63
            left_shoulder_idx = 13  # SMPL-X joint indices
            right_shoulder_idx = 14
        elif self.model_type == "mano":
            # MANO: 15 hand joints * 3 = 45 dims
            body_pose_dim = 45
            left_shoulder_idx = None  # Not applicable for MANO
            right_shoulder_idx = None
        else:
            # Default to SMPL-X dimensions
            body_pose_dim = 63
            left_shoulder_idx = 13
            right_shoulder_idx = 14

        # Initialize global orientation (always 3 dims, zero rotation)
        global_orient = torch.zeros([batch_size, 3], device=self.device)

        if self.canonical_pose == "zero":
            # Zero pose (all zeros)
            body_pose = torch.zeros([batch_size, body_pose_dim], device=self.device)
        elif self.canonical_pose == "t-pose":
            # T-pose (arms extended to sides) - not applicable for MANO
            body_pose = torch.zeros([batch_size, body_pose_dim], device=self.device)
            if self.model_type != "mano":
                # Set shoulder joints to extend arms (approximate T-pose)
                body_pose[:, left_shoulder_idx * 3 : left_shoulder_idx * 3 + 3] = (
                    torch.tensor([0, 0, -0.3], device=self.device)
                )
                body_pose[:, right_shoulder_idx * 3 : right_shoulder_idx * 3 + 3] = (
                    torch.tensor([0, 0, 0.3], device=self.device)
                )
        elif self.canonical_pose == "a-pose":
            # A-pose (arms at 45 degrees) - not applicable for MANO
            body_pose = torch.zeros([batch_size, body_pose_dim], device=self.device)
            if self.model_type != "mano":
                body_pose[:, left_shoulder_idx * 3 : left_shoulder_idx * 3 + 3] = (
                    torch.tensor([0, 0, -0.15], device=self.device)
                )
                body_pose[:, right_shoulder_idx * 3 : right_shoulder_idx * 3 + 3] = (
                    torch.tensor([0, 0, 0.15], device=self.device)
                )
        else:
            body_pose = torch.zeros([batch_size, body_pose_dim], device=self.device)

        # Concatenate global orientation and body pose to create full pose parameters
        full_pose_params = torch.cat([global_orient, body_pose], dim=1)

        return full_pose_params

    def __prepare_pose_params__(self, pose_params):
        pose_params = torch.as_tensor(pose_params, device=self.device).float()

        # Ensure input has batch dimension
        if pose_params.dim() != 2:
            raise ValueError(
                f"pose_params must have batch dimension [batch_size, pose_dim], got shape {pose_params.shape}"
            )
        # Split pose parameters
        if self.model_type == "smpl":
            # SMPL: 24 joints * 3 = 72 dims
            global_orient = pose_params[:, :3]
            body_pose = pose_params[:, 3:72]
            kwargs = {}
        elif self.model_type == "smplh":
            # SMPL+H: 22 body + 2 hands = 24 body joints + 30 hand joints
            global_orient = pose_params[:, :3]
            body_pose = pose_params[:, 3:66]
            left_hand_pose = (
                pose_params[:, 66:111] if pose_params.shape[1] > 66 else None
            )
            right_hand_pose = (
                pose_params[:, 111:156] if pose_params.shape[1] > 111 else None
            )
            kwargs = {
                "left_hand_pose": left_hand_pose,
                "right_hand_pose": right_hand_pose,
            }
        elif self.model_type == "smplx":
            global_orient = pose_params[:, :3]
            body_pose = pose_params[:, 3:66]
            jaw_pose = pose_params[:, 66:69] if pose_params.shape[1] > 66 else None
            leye_pose = pose_params[:, 69:72] if pose_params.shape[1] > 69 else None
            reye_pose = pose_params[:, 72:75] if pose_params.shape[1] > 72 else None
            left_hand_pose = (
                pose_params[:, 75:120] if pose_params.shape[1] > 75 else None
            )
            right_hand_pose = (
                pose_params[:, 120:165] if pose_params.shape[1] > 120 else None
            )
            if jaw_pose is not None:
                assert (
                    jaw_pose.shape[-1] == 3
                ), f"jaw_pose.shape[-1] = {jaw_pose.shape[-1]}"
            if leye_pose is not None:
                assert (
                    leye_pose.shape[-1] == 3
                ), f"leye_pose.shape[-1] = {leye_pose.shape[-1]}"
            if reye_pose is not None:
                assert (
                    reye_pose.shape[-1] == 3
                ), f"reye_pose.shape[-1] = {reye_pose.shape[-1]}, pose_params.shape = {pose_params.shape}"
            if left_hand_pose is not None:
                assert (
                    left_hand_pose.shape[-1] == 45
                ), f"left_hand_pose.shape[-1] = {left_hand_pose.shape[-1]}, pose_params.shape = {pose_params.shape}"
            if right_hand_pose is not None:
                assert (
                    right_hand_pose.shape[-1] == 45
                ), f"right_hand_pose.shape[-1] = {right_hand_pose.shape[-1]}, pose_params.shape = {pose_params.shape}"
            kwargs = {
                "jaw_pose": jaw_pose,
                "leye_pose": leye_pose,
                "reye_pose": reye_pose,
                "left_hand_pose": left_hand_pose,
                "right_hand_pose": right_hand_pose,
            }

        elif self.model_type == "mano":
            # MANO: 16 joints * 3 = 48 dims (1 global + 15 hand joints)
            global_orient = pose_params[:, :3]
            hand_pose = pose_params[:, 3:48]
            kwargs = {
                "hand_pose": hand_pose,
            }
            # For MANO, we use 'hand_pose' instead of 'body_pose'
            body_pose = hand_pose
        else:
            global_orient = pose_params[:, :3]
            body_pose = pose_params[:, 3:]
            kwargs = {}
        return global_orient, body_pose, kwargs

    def __prepare_shape_params__(self, shape_params):
        shape_params = torch.as_tensor(shape_params, device=self.device).float()

        # Ensure input has batch dimension
        if shape_params.dim() != 2:
            raise ValueError(
                f"shape_params must have batch dimension [batch_size, shape_dim], got shape {shape_params.shape}"
            )
        if shape_params.shape[1] < self.num_betas:
            # Pad with zeros if not enough betas
            padding = torch.zeros(
                shape_params.shape[0],
                self.num_betas - shape_params.shape[1],
                device=self.device,
            )
            shape_params = torch.cat([shape_params, padding], dim=1)
        elif shape_params.shape[1] > self.num_betas:
            # Truncate if too many betas
            shape_params = shape_params[:, : self.num_betas]
        return shape_params

    def forward_original(self, shape_params, pose_params):
        """
        Forward pass to get posed mesh using original smplx model.
        Same input/output interface as forward() but uses self.model directly.

        Args:
            shape_params: Shape parameters [batch_size, C] (batch_size must be 1 or equal to pose batch_size)
            pose_params: Pose parameters [batch_size, D]

        Returns:
            vtx: Posed vertices [batch_size, N, 3]
            face: Face indices [F, 3]
        """
        shape_params = torch.as_tensor(shape_params, device=self.device).float()
        pose_params = torch.as_tensor(pose_params, device=self.device).float()
        assert len(shape_params) == 1 or len(shape_params) == len(pose_params)
        if len(shape_params) == 1:
            shape_params = shape_params.repeat(len(pose_params), 1)

        # Use the centralized pose parameter processing method
        global_orient, body_pose, kwargs = self.__prepare_pose_params__(pose_params)
        shape_params = self.__prepare_shape_params__(shape_params)

        # Forward pass through the model
        ret = self.model(
            betas=shape_params,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True,
            **{k: v if v is not None else None for k, v in kwargs.items()},
        )

        # Always return with batch dimension
        return ret.vertices, self.faces, ret

    def animate_with_Q(self, shape_params, Q):
        # ! Q can have any arbitray translation!!
        B = len(Q)
        device = Q.device
        shape_params = torch.as_tensor(shape_params, device=device).float()
        shape_batch_size = shape_params.shape[0]
        assert shape_batch_size == 1 or shape_batch_size == B
        global_orient0, body_pose0, kwargs0 = self.__prepare_pose_params__(
            self.canonical_pose_params.clone().repeat(shape_batch_size, 1)
        )
        ret0 = self.model(
            betas=shape_params,
            global_orient=global_orient0,
            body_pose=body_pose0,
            return_verts=True,
            **{k: v if v is not None else None for k, v in kwargs0.items()},
        )
        joints0 = ret0.J
        T0 = ret0.A
        Q0 = torch.eye(4, device=device)[None, None].repeat(B, joints0.shape[1], 1, 1)
        Q0[:, :, :3, 3] = joints0
        Q0_inv = torch.inverse(Q0)

        T = torch.einsum("bkmn,bknl->bkml", Q0_inv, T0)
        if len(T) != B:
            assert len(T) == 1
            T = T.repeat(B, 1, 1, 1)
        T = torch.einsum("bkmn,bknl->bkml", Q, T)

        # Q =  T @ T0_inv @ Q0
        # T = Q @ Q0_inv @ T0
        vtx = ret0.v_shaped
        if len(vtx) != B:
            assert len(vtx) == 1
            vtx = vtx.repeat(B, 1, 1)

        w = self.model.lbs_weights
        ret_vtx = self.__skin__(vtx, w, T)
        return ret_vtx, self.faces

    def forward(self, shape_params, pose_params, center_at_joint_i=-1, transl=None):
        """
        Forward pass to get posed mesh using original smplx model.
        Same input/output interface as forward() but uses self.model directly.

        Args:
            shape_params: Shape parameters [batch_size, C] (batch_size must be 1 or equal to pose batch_size)
            pose_params: Pose parameters [batch_size, D]
            center_at_joint_i: Index of joint to center the mesh at
            transl: Translation parameters [batch_size, 3]

        Returns:
            vtx: Posed vertices [batch_size, N, 3]
            face: Face indices [F, 3]
        """
        B = len(pose_params)

        # Prepare shape params first to determine the correct batch size for canonical pose
        shape_params = torch.as_tensor(shape_params, device=self.device).float()
        assert shape_params.ndim <= 2
        if shape_params.ndim == 1:
            shape_params = shape_params.unsqueeze(0)
        shape_batch_size = shape_params.shape[0]
        assert shape_batch_size == 1 or shape_batch_size == B
        global_orient0, body_pose0, kwargs0 = self.__prepare_pose_params__(
            self.canonical_pose_params.clone().repeat(shape_batch_size, 1)
        )
        ret0 = self.model(
            betas=shape_params,
            global_orient=global_orient0,
            body_pose=body_pose0,
            return_verts=True,
            **{k: v if v is not None else None for k, v in kwargs0.items()},
        )
        joints0 = ret0.J
        T0 = ret0.A
        T0_inv = torch.inverse(T0)
        Q0 = torch.eye(4, device=self.device)[None, None].repeat(
            B, joints0.shape[1], 1, 1
        )
        Q0[:, :, :3, 3] = joints0

        # Forward pass through the model
        if len(shape_params) != B:
            assert len(shape_params) == 1
            shape_params = shape_params.repeat(B, 1)
        pose_params = torch.as_tensor(pose_params, device=self.device).float()
        # Use the centralized pose parameter processing method to get batch_size
        global_orient, body_pose, kwargs = self.__prepare_pose_params__(pose_params)
        batch_size = pose_params.shape[0]
        ret = self.model(
            betas=shape_params,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True,
            **{k: v if v is not None else None for k, v in kwargs.items()},
        )

        # T0 is shape dependent
        # Q =  T @ T0_inv @ Q0
        ret_T = torch.einsum("bkmn,bknl->bkml", T0_inv, Q0)
        if ret_T.shape[0] != batch_size:
            assert ret_T.shape[0] == 1
            ret_T = ret_T.repeat(batch_size, 1, 1, 1)
        ret_T = torch.einsum("bkmn,bknl->bkml", ret.A, ret_T)

        if center_at_joint_i != -1:
            center = ret.joints[:, center_at_joint_i, :][:, None, :]
            ret_T[:, :, :3, 3] = ret_T[:, :, :3, 3] - center
            ret.vertices = ret.vertices - center

        if transl is not None:
            ret_T[:, :, :3, 3] = ret_T[:, :, :3, 3] + torch.as_tensor(transl)[
                :, None
            ].to(ret_T)
            ret.vertices = ret.vertices + torch.as_tensor(transl)[:, None].to(ret_T)

        return ret_T, ret.vertices, self.faces, ret

    def __skin__(self, canonical_vtx, skinning_weights, T):
        """
        Apply Linear Blend Skinning to transform canonical vertices.

        Args:
            canonical_vtx: Canonical vertices [B, N, 3]
            skinning_weights: LBS weights [N, J] (shared across batch)
            T: Joint transformation matrices [B, J, 4, 4]

        Returns:
            vtx: Posed vertices [B, N, 3]
        """
        device = canonical_vtx.device

        # Ensure all inputs have proper batch dimensions
        if canonical_vtx.dim() != 3:
            raise ValueError(
                f"canonical_vtx must have shape [B, N, 3], got {canonical_vtx.shape}"
            )
        if T.dim() != 4:
            raise ValueError(f"T must have shape [B, J, 4, 4], got {T.shape}")
        if skinning_weights.dim() != 2:
            raise ValueError(
                f"skinning_weights must have shape [N, J], got {skinning_weights.shape}"
            )

        # Batched processing: [B, N, 3]
        B, N = canonical_vtx.shape[:2]
        v_homo = torch.cat(
            [canonical_vtx, torch.ones(B, N, 1, device=device)], dim=2
        )  # [B, N, 4]
        # T: [B, J, 4, 4], v_homo: [B, N, 4]
        # v_transformed: [B, N, J, 4]
        v_transformed = torch.einsum("bkij,bnj->bnki", T, v_homo)
        # skinning_weights: [N, J], expand to [B, N, J]
        skinning_weights = skinning_weights.unsqueeze(0).expand(B, -1, -1)
        vtx = (v_transformed * skinning_weights.unsqueeze(-1)).sum(dim=2)[
            :, :, :3
        ]  # [B, N, 3]

        return vtx
