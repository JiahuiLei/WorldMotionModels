"""Stage 2: text-motion contrastive matching on OMOMO joints24.

Loads Stage-1 movement_enc from ckpt/<dataset>/<decomp_name>/model/latest.tar,
then trains TextEncoderBiGRUCo + MotionEncoderBiGRUCo with ContrastiveLoss.
"""
import os
import sys
import logging
from os.path import join as pjoin

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch.utils.data import DataLoader

from options.train_options import TrainTexMotMatchOptions
from networks.modules import (
    MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo,
)
from networks.trainers import TextMotionMatchTrainer
from utils.word_vectorizer import WordVectorizer
from data.omomo_dataset import CanoObjectTrajDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_stage2")


def build_models(opt):
    dim_pose = opt.dim_pose
    movement_enc = MovementConvEncoder(dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(
        word_size=opt.dim_word, pos_size=opt.dim_pos_ohot,
        hidden_size=opt.dim_text_hidden, output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )
    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt.dim_movement_latent, hidden_size=opt.dim_motion_hidden,
        output_size=opt.dim_coemb_hidden, device=opt.device,
    )
    if not opt.is_continue:
        ckpt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, "model", "latest.tar")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Stage 1 ckpt not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=opt.device)
        movement_enc.load_state_dict(ckpt["movement_enc"])
        logger.info("Loaded Stage 1 movement_enc from %s", ckpt_path)
    return text_enc, motion_enc, movement_enc


def main() -> None:
    parser = TrainTexMotMatchOptions()
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
    opt.log_dir = pjoin("./logs", opt.dataset_name, opt.name)
    opt.eval_dir = pjoin(opt.save_root, "eval")
    for d in (opt.model_dir, opt.eval_dir, opt.log_dir):
        os.makedirs(d, exist_ok=True)

    w_vectorizer = WordVectorizer(opt.glove_meta_root, opt.glove_prefix)

    text_enc, motion_enc, movement_enc = build_models(opt)
    logger.info("text_enc params: %d", sum(p.numel() for p in text_enc.parameters()))
    logger.info("motion_enc params: %d", sum(p.numel() for p in motion_enc.parameters()))

    trainer = TextMotionMatchTrainer(opt, text_enc, motion_enc, movement_enc)

    train_ds = CanoObjectTrajDataset(
        train=True, data_root_folder=opt.data_root_folder,
        word_vectorizer=w_vectorizer, return_dict=False,
    )
    val_ds = CanoObjectTrajDataset(
        train=False, data_root_folder=opt.data_root_folder,
        word_vectorizer=w_vectorizer, return_dict=False,
    )
    logger.info("train=%d val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=True, pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=opt.batch_size, drop_last=True,
        num_workers=opt.num_workers, shuffle=False, pin_memory=True,
        collate_fn=collate_fn,
    )

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
