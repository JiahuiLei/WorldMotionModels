"""Argparse-based option parsers for CHOIS t2m_eval adapted to OMOMO.

Provides TrainDecompOptions (Stage 1) and TrainTexMotMatchOptions (Stage 2).
Field names match what networks/trainers.py expects (self.opt.X).
"""
import argparse
import os


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name", type=str, default="omomo")
    parser.add_argument("--name", type=str, default="decomp_v3")
    parser.add_argument("--checkpoints_dir", type=str, default="./ckpt")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_latest", type=int, default=500)
    parser.add_argument("--save_every_e", type=int, default=5)
    parser.add_argument("--eval_every_e", type=int, default=5)
    parser.add_argument("--is_continue", action="store_true")
    parser.add_argument("--is_train", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)

    # motion / classifier hparams (mirror CHOIS defaults)
    parser.add_argument("--dim_pose", type=int, default=72,
                        help="24 joints x 3 for OMOMO")
    parser.add_argument("--dim_movement_enc_hidden", type=int, default=512)
    parser.add_argument("--dim_movement_dec_hidden", type=int, default=512)
    parser.add_argument("--dim_movement_latent", type=int, default=512)
    parser.add_argument("--dim_word", type=int, default=300)
    parser.add_argument("--dim_pos_ohot", type=int, default=15)
    parser.add_argument("--dim_text_hidden", type=int, default=512)
    parser.add_argument("--dim_coemb_hidden", type=int, default=512)
    parser.add_argument("--dim_motion_hidden", type=int, default=1024)
    parser.add_argument("--max_text_len", type=int, default=30)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument("--max_motion_length", type=int, default=120)

    # GloVe meta root for the WordVectorizer .npy/.pkl files.
    parser.add_argument("--glove_meta_root", type=str,
                        default="./glove")
    parser.add_argument("--glove_prefix", type=str, default="our_vab")


class TrainDecompOptions(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        _add_common(self.parser)
        # Stage 1 specific losses
        self.parser.add_argument("--lambda_sparsity", type=float, default=0.001)
        self.parser.add_argument("--lambda_smooth", type=float, default=0.01)

    def parse(self):
        opt = self.parser.parse_args()
        return opt


class TrainTexMotMatchOptions(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        _add_common(self.parser)
        # Stage 2 specific: contrastive margin and pointer to Stage 1 ckpt dir
        self.parser.add_argument("--negative_margin", type=float, default=3.0)
        self.parser.add_argument("--decomp_name", type=str, default="decomp_v3",
                                 help="Stage-1 checkpoint subdir under ckpt/<dataset>/")
        # Override name default so Stage 2 writes to a different dir
        for a in self.parser._actions:
            if a.dest == "name":
                a.default = "text_motion_features"

    def parse(self):
        opt = self.parser.parse_args()
        return opt
