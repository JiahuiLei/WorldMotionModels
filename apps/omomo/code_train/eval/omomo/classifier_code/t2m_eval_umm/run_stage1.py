"""Stage 1: train MovementConvEncoder + Decoder autoencoder on OMOMO joints24.

Minimal wrapper — does NOT call plotting (CHOIS original uses a SAPIEN viewer
which is not needed for the FID/R_prec classifier). Saves latest.tar in
ckpt/<dataset>/<name>/model/.
"""
import os
import sys
import logging
from os.path import join as pjoin

# Ensure the t2m_eval_umm dir is on sys.path so `utils`, `networks`, `data`,
# `options` all resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch.utils.data import DataLoader

from options.train_options import TrainDecompOptions
from networks.modules import MovementConvEncoder, MovementConvDecoder
from networks.trainers import DecompTrainerV3
from utils.word_vectorizer import WordVectorizer
from data.omomo_dataset import CanoObjectTrajDataset, stage1_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_stage1")


def _noop_plot(*args, **kwargs):
    """Stage 1 trainer calls plot_eval every `eval_every_e` epochs. CHOIS uses
    SAPIEN viewer which we don't need for the classifier."""
    return


def main() -> None:
    parser = TrainDecompOptions()
    parser.parser.add_argument(
        "--data_root_folder", type=str,
        required=True,
    )
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.meta_dir = pjoin(opt.save_root, "meta")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./logs", opt.dataset_name, opt.name)
    for d in (opt.model_dir, opt.meta_dir, opt.eval_dir, opt.log_dir):
        os.makedirs(d, exist_ok=True)

    dim_pose = opt.dim_pose  # 72 for OMOMO joints24
    logger.info("dim_pose=%d dataset=%s device=%s", dim_pose, opt.dataset_name, opt.device)

    w_vectorizer = WordVectorizer(opt.glove_meta_root, opt.glove_prefix)

    movement_enc = MovementConvEncoder(dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)
    logger.info("movement_enc params: %d", sum(p.numel() for p in movement_enc.parameters()))
    logger.info("movement_dec params: %d", sum(p.numel() for p in movement_dec.parameters()))

    trainer = DecompTrainerV3(opt, movement_enc, movement_dec)

    train_ds = CanoObjectTrajDataset(
        train=True, data_root_folder=opt.data_root_folder,
        word_vectorizer=w_vectorizer, return_dict=True,
    )
    val_ds = CanoObjectTrajDataset(
        train=False, data_root_folder=opt.data_root_folder,
        word_vectorizer=w_vectorizer, return_dict=True,
    )
    logger.info("train=%d val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True,
        collate_fn=stage1_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=False, pin_memory=True,
        collate_fn=stage1_collate_fn,
    )

    trainer.train(train_loader, val_loader, _noop_plot, train_ds)


if __name__ == "__main__":
    main()
