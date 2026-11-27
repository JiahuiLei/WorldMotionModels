# lib_human - SMPL-X Body Model Wrapper

```
┌─────────────────────────────────────────────────────────────────┐
│                        lib_human Package                         │
│                                                                  │
│  Primary Interface: src/lib_human/human_model.py                │
│  Entry Point: from lib_human import HumanModel                  │
└─────────────────────────────────────────────────────────────────┘

ext/lib_human/
├── src/lib_human/              📦 Package source
│   ├── human_model.py          ⭐ PRIMARY INTERFACE (HumanModel class)
│   ├── __init__.py             
│   └── smplx_customized/       🔧 Backend (body_models, lbs, utils)
├── tests/                      🧪 Test suite
├── pyproject.toml              ⚙️  Package config
└── README.md                   📚 This file
```

A clean Python wrapper around [smplx](https://github.com/vchoutas/smplx) supporting SMPL, SMPL-H, SMPL-X, and MANO models.

## Installation

```bash
cd ext/lib_human
pip install -e .  # Development mode (editable)
# or
pip install .     # Standard installation
```

**With dev tools** (pytest, black, isort, flake8):
```bash
pip install -e ".[dev]"
```

## Download SMPL-X Models

- **SMPL-X**: https://smpl-x.is.tue.mpg.de/
- **SMPL**: https://smpl.is.tue.mpg.de/
- **MANO**: https://mano.is.tue.mpg.de/

Extract to a directory (e.g., `~/models/smplx/`)

## Quick Start

```python
import torch
from lib_human import HumanModel

# Initialize model
model = HumanModel(
    model_path='/path/to/models',
    model_type='smplx',     # 'smpl', 'smplh', 'smplx', 'mano'
    gender='neutral',       # 'neutral', 'male', 'female'
    canonical_pose='zero',  # 'zero', 't-pose', 'a-pose'
    device='cpu'
)

# Get canonical mesh
shape_params = torch.zeros(1, 10)
canonical_vtx, faces, skinning_weights = model.get_src_mesh(shape_params)

print(f"Vertices: {canonical_vtx.shape}")  # [B, N, 3]
print(f"Faces: {faces.shape}")             # [F, 3]
```

## Forward Pass

```python
# Forward pass with pose
shape_params = torch.zeros(1, 10)
pose_params = torch.zeros(1, 165)  # SMPL-X: 165 dims

Q, vertices, faces, ret = model.forward(shape_params, pose_params)
print(f"Posed vertices: {vertices.shape}")  # [B, N, 3]
```

## API

### HumanModel

**`__init__(model_path, model_type='smplx', gender='neutral', canonical_pose='zero', device='cpu', ...)`**

Initialize the model.

**`get_src_mesh(shape_params)`** → `(vertices, faces, weights)`

Get canonical mesh with shape parameters.

**`forward(shape_params, pose_params)`** → `(Q, vertices, faces, ret)`

Forward pass to get posed mesh.

**`animate_with_Q(shape_params, Q)`** → `(vertices, faces)`

Animate mesh with local joint transformations.

## Testing

```bash
pytest                      # Run all tests
pytest tests/test_smplx.py  # Specific test
```

Update `model_path` in test files before running.

## Model Parameters

| Model | Pose Dims | Joints | Notes |
|-------|-----------|--------|-------|
| SMPL | 72 | 24 | Body only |
| SMPL-H | 156 | 52 | Body + hands |
| SMPL-X | 165 | 55 | Full body + face |
| MANO | 48 | 16 | Hand only |

**Shape parameters (betas)**: Default 10, controls body shape (height, weight, proportions)

## Package Structure

```
ext/lib_human/
├── src/lib_human/           # Package source
│   ├── human_model.py      # Main interface
│   └── smplx_customized/   # Backend
├── tests/                   # Test suite
├── pyproject.toml          # Package config
├── README.md               # This file
└── LICENSE                 # MIT License
```

## License

MIT License. Note: SMPL-X models have their own licenses.

## References

- [SMPL-X GitHub](https://github.com/vchoutas/smplx)
- [SMPL-X Website](https://smpl-x.is.tue.mpg.de/)

