"""Download the third-party weights that TraceGen Layer 2 and Layer 3 need, pinned and checksum-verified.

    python apps/tracegen/prepare_data/download_weights.py --dinov3-url "<URL from Meta's DINOv3 download e-mail>"
    python apps/tracegen/prepare_data/download_weights.py --skip-dinov3 --skip-authors   # only the HF encoders

Written under ``<data>/tracegen/checkpoints/`` (``<data>`` = ``$WMM_DATA`` or ``<repo>/release_data``):

| Folder | What | Source (licence) |
|---|---|---|
| ``hf_home/hub/models--t5-base`` | T5 text encoder (WMM) | Hugging Face ``t5-base`` (Apache-2.0) |
| ``hf_home/hub/models--google--siglip-base-patch16-384`` | SigLIP image encoder (TraceGen baseline) | ``google/siglip-base-patch16-384`` (Apache-2.0) |
| ``hf_home/hub/models--timm--vit_large_patch16_dinov3.lvd1689m`` | DINOv3 ViT-L (TraceGen baseline) | ``timm/vit_large_patch16_dinov3.lvd1689m`` (DINOv3 License) |
| ``tracegen_authors/{Droid,EpicKitchen}`` | TraceGen authors' released checkpoints | ``JayLee131/TraceGen_{Droid,EpicKitchen}`` (Apache-2.0) |
| ``dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`` | DINOv3 ViT-S/16 image encoder (WMM) | Meta, after accepting the DINOv3 License |

The DINOv3 ViT-S/16 file is distributed by Meta only after you accept the DINOv3 License at
https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ ; Meta then e-mails a download URL. Pass that
URL with ``--dinov3-url``, or a file you already have with ``--dinov3-file``. Every file is checked against the
SHA-256 listed below; a mismatch stops the script.
"""

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[1]

# (repo, revision, file, sha256)
HF_CACHE = [
    ("t5-base", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1", "config.json",
     "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033"),
    ("t5-base", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1", "model.safetensors",
     "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4"),
    ("t5-base", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1", "spiece.model",
     "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
    ("t5-base", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1", "tokenizer.json",
     "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066"),
    ("google/siglip-base-patch16-384", "41aec1c83b32e0a6fca20ad88ba058aa5b5ea394", "config.json",
     "bd23a8a92607ff1ebdd84b625772246d9b0160d0a7f4b63bde2bd3ae1baa21de"),
    ("google/siglip-base-patch16-384", "41aec1c83b32e0a6fca20ad88ba058aa5b5ea394", "model.safetensors",
     "f273e98edc393ec5ffea71518b0c9ab4b0e8dd2be43affb22d49ff659fb28605"),
    ("google/siglip-base-patch16-384", "41aec1c83b32e0a6fca20ad88ba058aa5b5ea394", "preprocessor_config.json",
     "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8"),
    ("timm/vit_large_patch16_dinov3.lvd1689m", "30c1109559f65dea34316b0d4842d35c5771fe11", "model.safetensors",
     "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"),
]
# (repo, revision, local folder, file, sha256)
AUTHORS = [
    ("JayLee131/TraceGen_Droid", "2f23a7ecdc52d4d56e55c1c39fddc44bbeb2a507", "Droid", "README.md",
     "4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7"),
    ("JayLee131/TraceGen_Droid", "2f23a7ecdc52d4d56e55c1c39fddc44bbeb2a507", "Droid", "train.yaml",
     "527c831dbd2ff76e2271735cce3deab60e4755468b9f7b6fa9dcd86e8e890c9d"),
    ("JayLee131/TraceGen_Droid", "2f23a7ecdc52d4d56e55c1c39fddc44bbeb2a507", "Droid", "tracegen_droid.pth",
     "6c2be516547f1f2ee0766e752718fe51fbb229238d9b33a192fd213e6efaff6d"),
    ("JayLee131/TraceGen_EpicKitchen", "08299f4a2b01c116c925f69e39c47b285ada2da3", "EpicKitchen", "README.md",
     "4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7"),
    ("JayLee131/TraceGen_EpicKitchen", "08299f4a2b01c116c925f69e39c47b285ada2da3", "EpicKitchen", "train.yaml",
     "527c831dbd2ff76e2271735cce3deab60e4755468b9f7b6fa9dcd86e8e890c9d"),
    ("JayLee131/TraceGen_EpicKitchen", "08299f4a2b01c116c925f69e39c47b285ada2da3", "EpicKitchen",
     "tracegen_epickitchen.pth", "127acd7cb74b23a18feb882209fadef94d8d2638946323ada4612056f741ee00"),
]
DINOV3_FILE = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINOV3_SHA256 = "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d"
# Optional tokenizer files that t5-base does not have. The code loads the encoders offline (HF_HUB_OFFLINE=1), so the
# cache must record them as absent, as an online load would.
ABSENT = {"t5-base": ["added_tokens.json", "chat_template.jinja", "special_tokens_map.json", "tokenizer_config.json"]}


def check(path: Path, digest: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != digest:
        raise RuntimeError(f"{path}: sha256 {h.hexdigest()}, expected {digest}")
    print(f"  ok  {path}  sha256 {digest[:12]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.environ.get("WMM_DATA", str(APP.parents[1] / "release_data")),
                        help="release_data root (default: $WMM_DATA or <repo>/release_data)")
    parser.add_argument("--dinov3-url", help="DINOv3 ViT-S/16 download URL from Meta's e-mail")
    parser.add_argument("--dinov3-file", help="an existing copy of " + DINOV3_FILE)
    parser.add_argument("--skip-dinov3", action="store_true")
    parser.add_argument("--skip-hf", action="store_true", help="skip the Hugging Face encoders")
    parser.add_argument("--skip-authors", action="store_true", help="skip the TraceGen authors' checkpoints (baseline)")
    args = parser.parse_args()
    ckpt = Path(args.data).expanduser().resolve() / "tracegen" / "checkpoints"

    from huggingface_hub import hf_hub_download

    if not args.skip_hf:
        hub = ckpt / "hf_home" / "hub"
        for repo, revision, name, digest in HF_CACHE:
            check(Path(hf_hub_download(repo, name, revision=revision, cache_dir=hub)), digest)
            # Point the cache's `main` at the pinned revision, so that offline loads by repo name find these files.
            folder = hub / ("models--" + repo.replace("/", "--"))
            (folder / "refs").mkdir(parents=True, exist_ok=True)
            (folder / "refs" / "main").write_text(revision)
            for absent in ABSENT.get(repo, []):
                marker = folder / ".no_exist" / revision / absent
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
    if not args.skip_authors:
        for repo, revision, folder, name, digest in AUTHORS:
            check(Path(hf_hub_download(repo, name, revision=revision, local_dir=ckpt / "tracegen_authors" / folder)),
                  digest)
    if not args.skip_dinov3:
        dst = ckpt / "dinov3" / DINOV3_FILE
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.dinov3_file:
            shutil.copyfile(args.dinov3_file, dst)
        elif args.dinov3_url:
            urllib.request.urlretrieve(args.dinov3_url, dst)
        elif not dst.is_file():
            raise SystemExit("DINOv3 ViT-S/16: pass --dinov3-url (from Meta's e-mail) or --dinov3-file, "
                             "or --skip-dinov3")
        check(dst, DINOV3_SHA256)


if __name__ == "__main__":
    main()
