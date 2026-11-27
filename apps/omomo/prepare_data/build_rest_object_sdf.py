"""Build ``rest_object_sdf/`` (97 files) from the object SDFs of the OMOMO download.

    python apps/omomo/prepare_data/build_rest_object_sdf.py --omomo_dir <OMOMO data/> --out <dir>

OMOMO ships ``data/rest_object_sdf_256_npy_files/`` (63 files): for each object a signed-distance grid
``<mesh>.npy`` (256, 256, 256) float32, its grid metadata ``<mesh>.json`` and the mesh itself. The CHOIS code and the
contact scorer look an object's SDF up as ``<object>.ply.npy`` / ``<object>.ply.json`` (the name of its
``rest_object_geo/<object>.ply``). This script copies the 63 files unchanged and adds those 17 object names as
copies of the matching simplified mesh's SDF (``chair`` has only ``chair_cleaned.obj``).
"""

import argparse
import shutil
from pathlib import Path

OBJECTS = ["clothesstand", "floorlamp", "large_table", "largebox", "largetable", "monitor", "mop", "plasticbox",
           "smallbox", "smalltable", "suitcase", "trashcan", "tripod", "vacuum", "whitechair", "woodchair"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omomo_dir", required=True, type=Path, help="data/ folder of the OMOMO download")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    src = args.omomo_dir / "rest_object_sdf_256_npy_files"
    files = sorted(p for p in src.iterdir() if p.is_file())
    if len(files) != 63:
        raise SystemExit(f"{src}: expected the 63 files of the OMOMO download, found {len(files)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copyfile(p, args.out / p.name)
    aliases = {name: f"{name}_cleaned_simplified.obj" for name in OBJECTS}
    aliases["chair"] = "chair_cleaned.obj"
    for name, mesh in aliases.items():
        for ext in ("npy", "json"):
            shutil.copyfile(src / f"{mesh}.{ext}", args.out / f"{name}.ply.{ext}")
    n = sum(1 for _ in args.out.iterdir())
    if n != 97:
        raise SystemExit(f"{args.out}: expected 97 files, found {n}")
    print(f"wrote {n} files to {args.out}")


if __name__ == "__main__":
    main()
