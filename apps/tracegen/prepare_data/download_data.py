"""Download the TraceGen Droid and EpicKitchen splits and unpack them into the layout the code reads.

    python apps/tracegen/prepare_data/download_data.py                      # all six splits (about 500 GB)
    python apps/tracegen/prepare_data/download_data.py --splits droid_test epic_test   # Layer 2 only (30 GB)

Source: the TraceGen authors' Hugging Face datasets, pinned to one revision each:
``yoonkyojung/TraceGenDroid`` (CC BY 4.0, derived from DROID) and ``yoonkyojung/TraceGenEpickitchen``
(CC BY-NC 4.0, derived from EPIC-KITCHENS; non-commercial use only). Every tar shard is checked against its
SHA-256 before it is unpacked, and every split is checked against the expected number of episodes and files.

Layout written under ``<data>/tracegen/`` (``<data>`` = ``$WMM_DATA`` or ``<repo>/release_data``)::

    L2_eval/data_wmm/tracegen/data/Droid/droid_test/<episode>/{images,depth,samples}/...
    L2_eval/data_wmm/tracegen/data/Epickitchen/epic_test/<episode>/...
    L3_train/data_wmm/tracegen/data/Droid/droid/<episode>/...
    L3_train/data_wmm/tracegen/data/Epickitchen/epic_{1,2,3}/<episode>/...
    L3_train/data_wmm/tracegen/data/<Env>/<test split>  -> symlink to the L2_eval copy

Shards are downloaded to ``--tar-dir`` (default ``<data>/tracegen/_hf_tars``) and deleted after a successful
unpack unless ``--keep-tars`` is given. ``--from-dir`` unpacks shards that are already on disk (same
``<split>/<split>-NNNNN.tar`` names) instead of downloading them.
"""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

APP = Path(__file__).resolve().parents[1]

REPOS = {
    "Droid": ("yoonkyojung/TraceGenDroid", "c35bc5237bc28bc2ee91d4df589f04232e17ba4c"),
    "Epickitchen": ("yoonkyojung/TraceGenEpickitchen", "234a97a6ac15877ec610f627e55e67f6bd6eb5e7"),
}
TEST_SPLITS = ("droid_test", "epic_test")
# split -> (env, episodes, files)
SPLITS = {
    "droid": ("Droid", 9006, 788399),
    "droid_test": ("Droid", 500, 43682),
    "epic_1": ("Epickitchen", 5210, 178328),
    "epic_2": ("Epickitchen", 4366, 178714),
    "epic_3": ("Epickitchen", 248, 13034),
    "epic_test": ("Epickitchen", 500, 24284),
}
# repo file -> (size in bytes, sha256) at the pinned revision
SHARDS = {
    "droid/droid-00000.tar": (17402757120, "0fe63ce0699aab903fd3c53a3dc93fde18157f0d8ece43b3ae748b753c98c6b6"),
    "droid/droid-00001.tar": (17313413120, "da15abfeea68ed2a45a7aed99908dcc75ce2bfca447282fb8aa05d8ee4202f59"),
    "droid/droid-00002.tar": (17347041280, "775c5150c1475d52ddf5a10e2fe3140bd27259729decd9c6ca97f46a6786b149"),
    "droid/droid-00003.tar": (17543536640, "2116206983304c964cb4ec7e8ae734a69fcfcfb6fc3b58db068eda7fe599bbcc"),
    "droid/droid-00004.tar": (17444986880, "0ff3e29a2cdf359c022ba009b92bd1a5b47aafa51c976ecaaefe344ddc428394"),
    "droid/droid-00005.tar": (17484636160, "efb51862c6103c91a640b5d51c8a7f8bedf77e2342d69388a5b9643663a073fb"),
    "droid/droid-00006.tar": (17446512640, "2a4e4d1d9bb6ab266d63f21bbeba1d97367b2f45c52340e5571463cded9a19f4"),
    "droid/droid-00007.tar": (17525217280, "20db009e2d781e37f03b5eed1c8180c302c1b18481a6ca96ad57c52941a342fa"),
    "droid/droid-00008.tar": (17554616320, "a031faaa0c78d2a015242455236adef402715dc315469978e8448462cb2cdf4e"),
    "droid/droid-00009.tar": (17527029760, "068157446e7c5cd15db3c9eb3ea262ba64da734b632e57fbdae6f18db11af43c"),
    "droid/droid-00010.tar": (17278443520, "c02f2db207178aaea6cac1956f041f1213cc0b3e4011485f33ba02f587899019"),
    "droid/droid-00011.tar": (17388257280, "81a859cf5dbc76398e1ed537a3bfdd9c3e00d336575b8132f0af6942ea778386"),
    "droid/droid-00012.tar": (17490800640, "97f47f4f6f02a1047ff0a1ef3d41b7916493927d1b926d129c5b621921916a07"),
    "droid/droid-00013.tar": (17504542720, "ab4f63d147381f70d12beb78e14f611a29ca9489baf6895bdc553ea2b5f6bccc"),
    "droid/droid-00014.tar": (17441259520, "986f4e6a89bf7061daf89cdc21487cc57c9228b969b733a384218ede2462975c"),
    "droid/droid-00015.tar": (17347020800, "bd94b6a45dedca5d9fdea60ec762b7332f26c45e13af28d73845e775bfd9150a"),
    "droid/droid-00016.tar": (17380444160, "9337ff4c72691687ab7c93de333abea64798e1a4295eb19f5fe9befd3a2fa31b"),
    "droid/droid-00017.tar": (17517475840, "5b0e08b44c438c0e12343d9bc28ce319cf4fb154f8751e86938843e7ce331cf8"),
    "droid/droid-00018.tar": (215572480, "a8d23cb25c1324e37c20bbc30e8f09c6bf55a11f9ba5b63d543d7f282684f46f"),
    "droid_test/droid_test-00000.tar": (17390315520, "684b6685dfac70bbeff969902e1cd1b4168f69bb788b53944d68b3cc5ba26bb3"),
    "epic_1/epic_1-00000.tar": (10529914880, "fa046a4ddac9ff3c3c6ba8398e48ca65046e0ba5818cbddba23775e04524df23"),
    "epic_1/epic_1-00001.tar": (9254666240, "22b1522ab9b1bae79af8ab4c445da0400157a38eaaa713bd7f799a29d948c916"),
    "epic_1/epic_1-00002.tar": (9170012160, "1f3656f3afc6dd0b9dc19de6ad8f612bd4ae9f7b99b57eeac540fe8562141729"),
    "epic_1/epic_1-00003.tar": (7460833280, "8036c9ca298dd82a6b8d55b3867c10ab9f36b1ecf0f37602fb25641104f7ffae"),
    "epic_1/epic_1-00004.tar": (8837140480, "ccda1f0b5004204339480288ae3c2b15d559d1467d55fa66dea38e024d390f63"),
    "epic_1/epic_1-00005.tar": (9494128640, "9adbde1c7cfde6a163d5079561b61373fa6d7b4c8bece9fb2c98ec4bb10ec8f7"),
    "epic_1/epic_1-00006.tar": (7276810240, "ac73f31e16a1c18661d9e5a99fbcf3ddbc036effb28a9ac6573a7dd8e6dd8426"),
    "epic_1/epic_1-00007.tar": (6839767040, "dd9e8c287544caa2a8f152d9a51a6969f6d8f192b71ed5f04d004d9dd8d92281"),
    "epic_1/epic_1-00008.tar": (6749757440, "952a00aee397c0bab7130ba45c8a534ad8813ed9b0ee0008aea3725ef8c530ba"),
    "epic_1/epic_1-00009.tar": (10197944320, "54957e5d005f318b07dbfccf5d9b99ab5c9336edd322fdeee62283cc38c77cc2"),
    "epic_1/epic_1-00010.tar": (4293519360, "5a49bbefac238edfece7f5948e6ff25c3217245e798ef6c8129e86ccfb685ff0"),
    "epic_2/epic_2-00000.tar": (10332385280, "ac5caf6b59a13fc9b97249f7be03d38b34318247647d7433e4a8ff38766890bd"),
    "epic_2/epic_2-00001.tar": (12172851200, "9a039a2101a060fd1420401f3b644f372ba5f371081f3ef284ab5e91cdc06cbd"),
    "epic_2/epic_2-00002.tar": (9648588800, "4a4c9873db51cb1b926c931d9c71241d33295f0f88ed333b3740c2c61c3c99ae"),
    "epic_2/epic_2-00003.tar": (10445557760, "cb921c52fe20c5c61d9e4d344e0355d255bee537abd90985001d271856f375ee"),
    "epic_2/epic_2-00004.tar": (10187704320, "788d6c409860c8be72e6d74e59e2cf813ced8efa58c697415d90bae129849780"),
    "epic_2/epic_2-00005.tar": (8656414720, "a99236e9f0a96fedfb299d9150ef6250eda7e50e500aee26dfe3adb059c6cc5a"),
    "epic_2/epic_2-00006.tar": (10263982080, "a9fbd850fc3fda1211520b2a59674ecacb61f4e49e6b7fd6bff8ebf7cc6757da"),
    "epic_2/epic_2-00007.tar": (11338526720, "13cac81ca84a193ce9f30b7265cfffa29d1aa3649ec0690f86ec43f8fd7926a6"),
    "epic_2/epic_2-00008.tar": (8328294400, "d3893b6532e9d04c34475a9645331699b379d033b2f48e5dcd459a82be2dad88"),
    "epic_3/epic_3-00000.tar": (6855301120, "a05ca258f49fe20da350a9dbea47ea470d6cac4410badd65afbc1f83d3d7277b"),
    "epic_test/epic_test-00000.tar": (12427827200, "aac018e7263d9e0054e962076378a1b8a6c99f4a53f1aebd9942bb1e2a162311"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def split_dir(root: Path, split: str) -> Path:
    env = SPLITS[split][0]
    layer = "L2_eval" if split in TEST_SPLITS else "L3_train"
    return root / layer / "data_wmm" / "tracegen" / "data" / env / split


def fetch(name: str, tar_dir: Path, from_dir: Path) -> Path:
    """Return a verified local copy of one shard (downloaded unless --from-dir is given)."""
    size, digest = SHARDS[name]
    if from_dir is not None:
        path = from_dir / name
    else:
        from huggingface_hub import hf_hub_download

        repo, revision = REPOS[SPLITS[name.split("/")[0]][0]]
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=tar_dir))
    if path.stat().st_size != size:
        raise RuntimeError(f"{path}: size {path.stat().st_size}, expected {size}")
    got = sha256(path)
    if got != digest:
        raise RuntimeError(f"{path}: sha256 {got}, expected {digest}")
    print(f"  ok  {name}  sha256 {digest[:12]}", flush=True)
    return path


def count(dst: Path):
    episodes = sum(1 for p in dst.iterdir() if p.is_dir())
    files = sum(len(fs) for _, _, fs in os.walk(dst))
    return episodes, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.environ.get("WMM_DATA", str(APP.parents[1] / "release_data")),
                        help="release_data root (default: $WMM_DATA or <repo>/release_data)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--tar-dir", default=None, help="where shards are downloaded (default <data>/tracegen/_hf_tars)")
    parser.add_argument("--from-dir", default=None, help="unpack shards already on disk instead of downloading")
    parser.add_argument("--keep-tars", action="store_true", help="keep downloaded shards after unpacking")
    args = parser.parse_args()

    root = Path(args.data).expanduser().resolve() / "tracegen"
    tar_dir = Path(args.tar_dir).expanduser().resolve() if args.tar_dir else root / "_hf_tars"
    from_dir = Path(args.from_dir).expanduser().resolve() if args.from_dir else None
    for split in args.splits:
        dst = split_dir(root, split)
        expected = SPLITS[split][1:]
        if dst.is_dir() and count(dst) == expected:
            print(f"{split}: already complete at {dst}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        print(f"{split} -> {dst}", flush=True)
        for name in sorted(n for n in SHARDS if n.startswith(split + "/")):
            path = fetch(name, tar_dir, from_dir)
            subprocess.run(["tar", "-xf", str(path), "-C", str(dst)], check=True)
            if from_dir is None and not args.keep_tars:
                path.unlink()
        got = count(dst)
        if got != expected:
            raise RuntimeError(f"{split}: {got[0]} episodes / {got[1]} files, expected {expected[0]} / {expected[1]}")
        print(f"{split}: {got[0]} episodes, {got[1]} files (as expected)", flush=True)

    # The training configs also read the test splits from L3_train/; link them to the Layer-2 copies.
    for split in TEST_SPLITS:
        env = SPLITS[split][0]
        link = root / "L3_train" / "data_wmm" / "tracegen" / "data" / env / split
        if split_dir(root, split).is_dir() and link.parent.is_dir() and not link.exists():
            link.symlink_to(Path("../../../../../L2_eval/data_wmm/tracegen/data") / env / split)
            print(f"linked {link} -> L2_eval copy")


if __name__ == "__main__":
    main()
