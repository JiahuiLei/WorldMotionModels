# lib_mani (ManiSkill Library)

This library provides utilities for replaying ManiSkill demonstrations, evaluating ManiSkill environments, and converting them into a format compatible with WMM.

## Installation

```bash
cd ext/lib_mani
pip install -e .
```

## Structure

- `lib_mani/`: Main package containing:
    - `replay.py`: Utilities for replaying ManiSkill demonstrations.
    - `replay_ms.py`: Additional replay script (formerly in `src/umm/sims/lib_maniskill/`).
    - `maniskill_evaluator.py`: ManiSkill environment evaluator.
    - `mesh_utils.py`: Utilities for handling robot and object meshes.
    - `viser_rollout.py`: Viser-based visualization tool.
    - `utils/`: Subpackage for action and state utilities.

## Usage

### 1. Download Demonstrations

```bash
python -m mani_skill.utils.download_demo PickCube-v1
```

### 2. Process Data

```bash
python prepare_data/maniskill/replay.py --env_id PickCube-v1 --traj_path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 --output_dir data/maniskill_processed
```

### 3. Visualize Data

```bash
python prepare_data/maniskill/viser_rollout.py --root data/maniskill_processed
```

## Semantic Segmentation IDs

The conversion pipeline automatically groups instance IDs into semantic categories. For the Franka Panda robot, you may notice multiple IDs for the grippers. This is because the visual model separates the metal finger from the rubber pad.

Typical mapping for Panda robot:
- **`endeffector`**: The hand base (`panda_hand`).
- **`gripper_left_finger`**: The metal link of the left finger.
- **`gripper_left_pad`**: The rubber pad of the left finger.
- **`gripper_right_finger`**: The metal link of the right finger.
- **`gripper_right_pad`**: The rubber pad of the right finger.
- **`robotbody`**: All other robot links (links 0-8).
- **`objects`**: Manipulation targets (e.g., cube, debris).
- **`table`**: The workspace surface.

## Supported Environments

Below is a list of ManiSkill environments. Environments marked with **(Downloadable)** have official demonstrations available via `python -m mani_skill.utils.download_demo <env_id>`.

### Common Tabletop Environments (Downloadable)
- `PickCube-v1`
- `StackCube-v1`
- `PushCube-v1`
- `PlugCharger-v1`
- `PushT-v1`
- `PullCube-v1`
- `PokeCube-v1`
- `RollBall-v1`
- `LiftPegUpright-v1`
- `PegInsertionSide-v1`
- `TwoRobotPickCube-v1`
- `TwoRobotStackCube-v1`

### Other Downloadable Environments
- `AnymalC-Reach-v1`
- `DrawTriangle-v1`
- `PullCubeTool-v1`
- `StackPyramid-v1`

### Registered Environments (No official demos yet)
- `PickSingleYCB-v1` (Note: Demos for this are not currently available via the download utility)
- `PickClutterYCB-v1`
- `AssemblingKits-v1`
- `TurnFaucet-v1`
- `OpenCabinetDoor-v1`
- `OpenCabinetDrawer-v1`
- `DrawSVG-v1`
- `TableTopFreeDraw-v1`
- `RotateSingleObjectInHandLevel0-v1` (to Level 3)
- `RotateValveLevel0-v1` (to Level 4)
- `SO100GraspCube-v1`
- `PickCubeSO100-v1`
- `PickCubeWidowXAI-v1`
- `UnitreeG1PlaceAppleInBowl-v1`
- `UnitreeG1TransportBox-v1`
- `UnitreeGo2-Reach-v1`
- `FMBAssembly1Easy-v1`
