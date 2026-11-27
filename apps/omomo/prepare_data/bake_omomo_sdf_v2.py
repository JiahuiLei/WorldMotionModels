"""Bake metric signed-distance grids of the OMOMO objects from ``rest_object_geo/*.ply``.

    cd release_data/omomo/L2_eval
    python <repo>/apps/omomo/prepare_data/bake_omomo_sdf_v2.py \
        --ply_dir data_wmm/omomo_v2/rest_object_geo --out_dir data_wmm/omomo_v3/rest_object_sdf_v2

``rest_object_sdf/`` (from OMOMO) stores SDFs in voxel units of the original meshes; the model needs them in metres
in the frame of ``rest_object_geo/<object>.ply``. Written per object (13 objects, 26 files):

    <out_dir>/<object>.npy    (256, 256, 256) float32, signed distance in metres, axis order Z, Y, X,
                              negative inside the mesh
    <out_dir>/<object>.json   {centroid, extents, grid_dim, padding_m, axis_order, sign_convention, ply_source}

The grid covers the mesh's bounding box plus ``--padding_m`` (0.5 m) on every side. ``ply_source`` records
``--ply_dir`` as given; the paper data was baked with the relative path shown above.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pysdf
import trimesh

PADDING_M_DEFAULT = 0.50
GRID_DIM_DEFAULT = 256


def bake_one(ply_path: str, out_dir: str, grid_dim: int, padding_m: float, force: bool = False) -> str:
    stem = Path(ply_path).stem
    out_npy = Path(out_dir) / f"{stem}.npy"
    out_json = Path(out_dir) / f"{stem}.json"
    if out_npy.exists() and out_json.exists() and not force:
        print(f"[skip] {stem}: already baked", flush=True)
        return stem

    mesh = trimesh.load(ply_path, force="mesh", process=False)
    bounds = mesh.bounds
    centroid = bounds.mean(axis=0).astype(np.float64)
    extents = (bounds[1] - bounds[0]) + 2.0 * padding_m
    aabb_min = centroid - extents / 2.0
    aabb_max = centroid + extents / 2.0

    print(f"[bake] {stem}: extents={extents}, centroid={centroid}, {grid_dim**3} SDF queries", flush=True)

    lin = [np.linspace(aabb_min[i], aabb_max[i], grid_dim, dtype=np.float64) for i in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # pysdf returns positive inside; store negative inside.
    f = pysdf.SDF(mesh.vertices, mesh.faces.astype(np.uint32))
    sd = f(pts)
    del pts, gx, gy, gz
    sdf = (-sd).astype(np.float32).reshape(grid_dim, grid_dim, grid_dim)
    del sd
    sdf = np.transpose(sdf, (2, 1, 0))  # (Z, Y, X)

    np.save(out_npy, sdf)
    json.dump(
        {
            "centroid": centroid.tolist(),
            "extents": extents.tolist(),
            "grid_dim": grid_dim,
            "padding_m": padding_m,
            "axis_order": "ZYX",
            "sign_convention": "negative=inside",
            "ply_source": str(ply_path),
        },
        open(out_json, "w"),
        indent=2,
    )
    print(f"[bake] {stem}: range [{sdf.min():.3f}, {sdf.max():.3f}] m -> {out_npy.name}", flush=True)
    return stem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid_dim", type=int, default=GRID_DIM_DEFAULT)
    p.add_argument("--padding_m", type=float, default=PADDING_M_DEFAULT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--num_workers", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plys = sorted(Path(args.ply_dir).glob("*.ply"))
    print(f"[scan] {len(plys)} PLY files in {args.ply_dir}", flush=True)
    work = [(str(pp), args.out_dir, args.grid_dim, args.padding_m, args.force) for pp in plys]
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            pool.starmap(bake_one, work)
    else:
        for w in work:
            bake_one(*w)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
