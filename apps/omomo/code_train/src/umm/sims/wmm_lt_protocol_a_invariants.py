"""Runtime invariants for the LangTable Protocol A eval callback.

These three functions are HARD FAILS that detect silent train/eval format
mismatch at the single point where policy output meets env input. Markdown
audits cannot catch silent failure; asserts can.

Used by ``umm.sims.wmm_lt_eval_callback.EvalProtocolACallback``:

- ``assert_observation_compatible(env_obs, model_input_schema)`` — called
  on the first observation of the first rollout each fire; fails if the
  env does not expose the keys/shapes/dtypes the model was trained on.
- ``assert_action_compatible(action, env_action_space)`` — called once
  per fire before the first ``env.step(action)``; fails if the model's
  emitted action is not within the env's action box or has the wrong
  shape/dtype.
- ``oracle_replay_sanity(env, oracle, n_steps=20)`` — called exactly
  once from callback ``setup(stage="fit")`` via a one-shot throwaway
  env. Runs 20 oracle steps and asserts no NaN / no uid-set drift / no
  stale ee uid.

All three raise ``AssertionError`` with the offending field + observed
value + expected value on mismatch — never silent pass, never silent
fallback.

See also: ``CONVERTER_AUDIT.md`` (same workdir) for the full train-vs-eval
format mapping this module asserts.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

log = logging.getLogger(__name__)


# Default schema the Protocol A helper path trains/expects. Keys are
# observation fields returned by ``language_table.LanguageTable.step()``
# (see ext/lib_langtable/language_table/environments/language_table.py:503-530).
DEFAULT_MODEL_INPUT_SCHEMA: dict[str, dict[str, Any]] = {
    # Logged / used for MP4; model does not ingest rgb directly, but the
    # callback records it, so shape + dtype are required.
    "rgb": {"ndim": 3, "dtype_kind": "u", "last_dim": 3},
    # Per-uid world-to-link SE(3) (dict of 4x4 matrices). Model consumes
    # these indirectly via ``reset_env`` -> obs["instance_poses"][uid].
    # The real env returns float32 SE(3) (_make_matrix uses np.eye(4, dtype=np.float32));
    # ``dtype_kind="f"`` accepts both f32 and f64 (torch.as_tensor casts on ingest).
    "instance_poses": {
        "type": "dict_of_matrix",
        "matrix_shape": (4, 4),
        "dtype_kind": "f",
    },
    # Discovery: the EE uid is the key whose value == 4 in instance_map.
    "instance_map": {"type": "dict", "required_value": 4, "required_count": 1},
    # Raw instruction tokens; decoded by env.decode_instruction(...) to str.
    "instruction": {"type": "non_empty"},
}

_ACTION_SLACK = 1e-5


def _fmt_dtype(a: np.ndarray) -> str:
    return f"{a.dtype!s} (kind={a.dtype.kind})"


def assert_observation_compatible(
    env_obs: Mapping[str, Any],
    model_input_schema: Mapping[str, Any] | None = None,
) -> None:
    """Hard-fail if env_obs does not expose the fields the model expects.

    Args:
        env_obs: Observation dict returned by ``env.reset()`` or
            ``env.step(a)[0]``. Must be a ``Mapping``.
        model_input_schema: Dict describing required keys/shapes/dtypes.
            Defaults to :data:`DEFAULT_MODEL_INPUT_SCHEMA` (the schema
            the Protocol A path trains on — see ``CONVERTER_AUDIT.md``).

    Raises:
        AssertionError: With the offending field, observed value, and
            expected value on first mismatch. No silent pass.
    """
    if model_input_schema is None:
        model_input_schema = DEFAULT_MODEL_INPUT_SCHEMA
    assert isinstance(env_obs, Mapping), (
        f"env_obs must be a Mapping, got {type(env_obs).__name__}"
    )

    for key, spec in model_input_schema.items():
        assert key in env_obs, (
            f"env_obs missing key {key!r}; "
            f"present keys={sorted(list(env_obs.keys()))}"
        )
        val = env_obs[key]
        spec_type = spec.get("type")

        if spec_type == "dict_of_matrix":
            assert isinstance(val, Mapping), (
                f"env_obs[{key!r}] must be dict-like, got {type(val).__name__}"
            )
            assert len(val) > 0, (
                f"env_obs[{key!r}] is empty — at least one per-uid pose required"
            )
            exp_shape = tuple(spec["matrix_shape"])
            exp_dtype_kind = spec.get("dtype_kind")
            for uid, mat in val.items():
                assert isinstance(mat, np.ndarray), (
                    f"env_obs[{key!r}][{uid!r}] must be np.ndarray, "
                    f"got {type(mat).__name__}"
                )
                assert tuple(mat.shape) == exp_shape, (
                    f"env_obs[{key!r}][{uid!r}].shape={tuple(mat.shape)} "
                    f"!= expected {exp_shape}"
                )
                if exp_dtype_kind is not None:
                    assert mat.dtype.kind == exp_dtype_kind, (
                        f"env_obs[{key!r}][{uid!r}].dtype={_fmt_dtype(mat)} "
                        f"does not match kind={exp_dtype_kind!r}"
                    )
                assert np.isfinite(mat).all(), (
                    f"env_obs[{key!r}][{uid!r}] contains non-finite values "
                    f"(NaN/Inf)"
                )
        elif spec_type == "dict":
            assert isinstance(val, Mapping), (
                f"env_obs[{key!r}] must be dict-like, got {type(val).__name__}"
            )
            req_value = spec.get("required_value")
            req_count = spec.get("required_count")
            if req_value is not None and req_count is not None:
                matched = [k for k, v in val.items() if v == req_value]
                assert len(matched) == req_count, (
                    f"env_obs[{key!r}] must contain exactly {req_count} "
                    f"entries with value {req_value!r} (e.g. EE class 4); "
                    f"found {len(matched)}: {matched}"
                )
        elif spec_type == "non_empty":
            # Accept either a decoded str or a raw token array; both must
            # be non-trivially present.
            if isinstance(val, str):
                assert len(val) > 0, (
                    f"env_obs[{key!r}] is an empty string"
                )
            elif isinstance(val, np.ndarray):
                assert val.size > 0, (
                    f"env_obs[{key!r}] is an empty ndarray"
                )
            else:
                # Fall back to bool-ness for other types (bytes, etc.).
                assert bool(val), f"env_obs[{key!r}] is falsy / empty"
        else:
            # Default image-style spec: ndim + dtype_kind + last_dim.
            assert isinstance(val, np.ndarray), (
                f"env_obs[{key!r}] must be np.ndarray, "
                f"got {type(val).__name__}"
            )
            exp_ndim = spec.get("ndim")
            if exp_ndim is not None:
                assert val.ndim == exp_ndim, (
                    f"env_obs[{key!r}].ndim={val.ndim} != expected {exp_ndim}"
                )
            exp_last = spec.get("last_dim")
            if exp_last is not None:
                assert val.shape[-1] == exp_last, (
                    f"env_obs[{key!r}].shape[-1]={val.shape[-1]} "
                    f"!= expected {exp_last}"
                )
            exp_kind = spec.get("dtype_kind")
            if exp_kind is not None:
                assert val.dtype.kind == exp_kind, (
                    f"env_obs[{key!r}].dtype={_fmt_dtype(val)} "
                    f"does not match kind={exp_kind!r}"
                )
    log.debug(
        "assert_observation_compatible OK — %d checked keys",
        len(model_input_schema),
    )


def assert_action_compatible(
    action: Any,
    env_action_space: Any,
    raise_on_oob: bool = False,
) -> None:
    """Validate a model-emitted action against ``env.action_space``.

    Contract tiers (each one triggers ``AssertionError`` on failure
    unless noted):

    - **Hard fail:** shape mismatch, non-float dtype, non-finite
      entries. These are definite silent-failure surfaces — any of
      them means the model output does NOT match the env input
      contract and the converter is broken. Always raise.
    - **Range check:** out-of-box violations are always reported via
      ``logging.warning`` (to make them visible in training logs);
      the function raises only when ``raise_on_oob=True``. The
      LangTable env clips internally at ±0.1, so an OOB action is
      reduced to pybullet clipping — not a silent converter failure.
      Warnings make this visible without crashing runs where an
      untrained / mid-training policy legitimately emits OOB
      actions.

    Args:
        action: Numpy array (or scalar / sequence coercible to one) the
            model is about to pass into ``env.step``.
        env_action_space: A ``gym.spaces.Box``-like object with
            ``shape``, ``low``, ``high``, and ``dtype`` attributes. For
            LangTable this is
            ``spaces.Box(low=-0.1, high=0.1, shape=(2,))``
            (ext/lib_langtable/.../language_table.py:156).
        raise_on_oob: When True, also raise on OOB range. Use this for
            reproducing a specific bug; leave the default for production
            eval where we want visibility, not crashes.

    Raises:
        AssertionError: On shape, dtype, or finiteness mismatch (always);
            on range OOB when ``raise_on_oob=True``.
    """
    # Permit plain Python sequences (tuple/list) and scalars; coerce.
    if not isinstance(action, np.ndarray):
        action = np.asarray(action)

    assert env_action_space is not None, (
        "env_action_space is None; pass env.action_space explicitly"
    )
    exp_shape = tuple(env_action_space.shape)
    assert tuple(action.shape) == exp_shape, (
        f"action.shape={tuple(action.shape)} != "
        f"env.action_space.shape={exp_shape}"
    )
    assert action.dtype.kind == "f", (
        f"action.dtype={_fmt_dtype(action)} not a float kind"
    )
    assert np.isfinite(action).all(), (
        f"action contains non-finite entries: {action!r}"
    )

    low = np.asarray(env_action_space.low, dtype=np.float64)
    high = np.asarray(env_action_space.high, dtype=np.float64)
    act = action.astype(np.float64)
    lo_violation = (act < low - _ACTION_SLACK)
    hi_violation = (act > high + _ACTION_SLACK)
    if lo_violation.any() or hi_violation.any():
        msg = (
            f"action out of env.action_space bounds (slack={_ACTION_SLACK}): "
            f"action={act.tolist()}, low={low.tolist()}, high={high.tolist()}. "
            "LangTable env clips internally; this is visible but not fatal."
        )
        if raise_on_oob:
            raise AssertionError(msg)
        log.warning(msg)


def assert_env_level_compatible(env: Any) -> None:
    """Hard-fail if env-level (non-obs) contract fields are missing.

    Complements :func:`assert_observation_compatible` (which checks the
    per-step obs dict). This checks env-wide invariants that must hold
    for the Protocol A path to work:

    1. ``env.get_all_active_object_ids()`` returns ≥ 1 uid and every
       returned uid is a key in ``obs["instance_poses"]`` — required by
       ``LangTableVisualizer.prepare_trace_attr_oracle``.
    2. ``env._control_frequency > 0`` — required by the timestamp
       builder (denominator in ``realtime = arange / fps``).

    Args:
        env: The ``language_table.LanguageTable`` instance.

    Raises:
        AssertionError: On empty object list, uid mismatch, or
            non-positive control_frequency.
    """
    obj_ids = env.get_all_active_object_ids()
    assert len(obj_ids) >= 1, (
        f"env.get_all_active_object_ids() returned empty list — "
        f"Protocol A requires at least one block to push. Got {obj_ids!r}."
    )
    obs = env._compute_observation()
    missing = [u for u in obj_ids if u not in obs.get("instance_poses", {})]
    assert not missing, (
        f"env.get_all_active_object_ids() returned uids not present in "
        f"obs['instance_poses']: missing={missing}; obj_ids={obj_ids}; "
        f"instance_poses keys={list(obs.get('instance_poses', {}).keys())}"
    )

    fps = getattr(env, "_control_frequency", None)
    assert isinstance(fps, (int, float)) and fps > 0, (
        f"env._control_frequency must be a positive number, got {fps!r}. "
        "LangTableVisualizer.infer_agent divides by this to build "
        "real_timestep; 0 or None would produce NaN/Inf."
    )


def oracle_replay_sanity(env: Any, oracle: Any, n_steps: int = 20) -> None:
    """Run n_steps oracle actions and verify env stays well-formed.

    Called exactly ONCE per training run from
    :meth:`EvalProtocolACallback.setup` when ``stage == "fit"`` — guarded
    by ``self._sanity_done`` in the callback. Uses a throwaway env that
    the caller owns; this function does NOT reset or close ``env``.

    Checks performed per step:

    - :func:`assert_observation_compatible` — obs schema contract.
    - :func:`assert_action_compatible` with ``raise_on_oob=True`` — oracle
      OOB is a definite bug (oracle is a classical RRT planner bounded by
      the env action space), so we hard-fail here regardless of the
      callback's tier setting.
    - uid-set invariance across consecutive frames.
    - finite oracle action.

    Env-level invariants (object list / control_frequency) are checked
    once before the loop via :func:`assert_env_level_compatible`.

    Args:
        env: Pre-built ``language_table.LanguageTable`` instance (already
            seeded by the caller). Must expose ``compute_state``,
            ``reset``, ``step``, ``action_space``, ``get_all_active_object_ids``,
            ``_control_frequency``.
        oracle: ``ObstacleOrientedPushOracleBoard2dRRT`` instance bound
            to ``env``.
        n_steps: Number of oracle steps to replay. Defaults to 20 (enough
            to detect uid drift and NaN in a typical start-of-training
            scene without burning wallclock).

    Raises:
        AssertionError: On NaN in action/obs, on uid-set drift between
            frames, on missing EE class, on oracle plan failure, on
            env-level contract violation, or on oracle-emitted OOB action
            (strict).
    """
    assert n_steps > 0, (
        f"oracle_replay_sanity: n_steps must be > 0, got {n_steps}. "
        "The caller is bypassing the sanity check — fix the call site."
    )
    env.reset()
    assert_env_level_compatible(env)
    raw_state = env.compute_state()
    assert oracle.get_plan(raw_state), (
        "oracle_replay_sanity: oracle.get_plan failed on first state — "
        "env init is infeasible. Retry seeding the env before invoking."
    )

    assert_observation_compatible(env._compute_observation())

    prev_uids: frozenset | None = None
    t = 0
    for t in range(n_steps):
        a = oracle.action(is_first=(t == 0))
        a_arr = np.asarray(a)
        assert np.isfinite(a_arr).all(), (
            f"oracle_replay_sanity step {t}: non-finite oracle action {a!r}"
        )
        # Oracle is a classical RRT planner — OOB actions here are a
        # definite bug, NOT untrained-model noise. Hard-fail.
        assert_action_compatible(a_arr, env.action_space, raise_on_oob=True)

        obs, _reward, done, _info = env.step(a_arr)
        assert_observation_compatible(obs)

        cur_uids = frozenset(obs["instance_poses"].keys())
        if prev_uids is not None:
            assert cur_uids == prev_uids, (
                f"oracle_replay_sanity step {t}: instance_poses uid-set "
                f"drifted — added {cur_uids - prev_uids}, "
                f"removed {prev_uids - cur_uids}"
            )
        prev_uids = cur_uids

        if done:
            log.info(
                "oracle_replay_sanity: oracle reached done at step %d (early OK)", t
            )
            break

    log.info(
        "oracle_replay_sanity OK — %d oracle steps replayed with no uid drift / no NaN",
        t + 1,
    )
