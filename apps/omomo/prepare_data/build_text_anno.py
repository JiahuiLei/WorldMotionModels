"""Unpack the OMOMO text annotations and split them by subject.

    python apps/omomo/prepare_data/build_text_anno.py --zip omomo_text_anno.zip --out <dir>

``omomo_text_anno.zip`` (from the OMOMO repository, see ``download.sh``) holds 4,912 files
``omomo_text_anno_json_data/<seq_name>.json``, each ``{"<seq_name>": "<instruction>"}``. Written:

    <out>/train/<seq_name>.json   4,354 files, subjects sub1..sub15
    <out>/test/<seq_name>.json      558 files, subjects sub16, sub17 (the held-out test subjects)

File contents are copied unchanged.
"""

import argparse
import zipfile
from pathlib import Path

TEST_SUBJECTS = ("sub16", "sub17")
PREFIX = "omomo_text_anno_json_data/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    counts = {"train": 0, "test": 0}
    with zipfile.ZipFile(args.zip) as zf:
        for name in sorted(zf.namelist()):
            if not name.startswith(PREFIX) or not name.endswith(".json"):
                continue
            fname = name[len(PREFIX):]
            split = "test" if fname.split("_")[0] in TEST_SUBJECTS else "train"
            dest = args.out / split / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            counts[split] += 1
    if counts != {"train": 4354, "test": 558}:
        raise SystemExit(f"unexpected annotation counts {counts}; expected 4354 train and 558 test")
    print(f"wrote {counts['train']} train and {counts['test']} test annotations to {args.out}")


if __name__ == "__main__":
    main()
