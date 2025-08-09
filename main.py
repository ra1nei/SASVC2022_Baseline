import argparse
import os
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl
from omegaconf import OmegaConf

from utils import *  # find_gpus, keras_decay, v.v.

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    # --- Load config (.conf) và convert sang dict để System dùng ---
    ocfg = OmegaConf.load(args.config)
    config = OmegaConf.to_container(ocfg, resolve=True)  # <- DICT thuần

    output_dir = Path(args.output_dir)
    pl.seed_everything(config["seed"], workers=True)

    # --- (Tuỳ chọn) Nếu còn dùng spk_meta .pk thì giữ, còn không thì bỏ ---
    # if not (
    #     os.path.exists(config["dirs"]["spk_meta"] + "spk_meta_trn.pk")
    #     and os.path.exists(config["dirs"]["spk_meta"] + "spk_meta_dev.pk")
    #     and os.path.exists(config["dirs"]["spk_meta"] + "spk_meta_eval.pk")
    # ):
    #     generate_spk_meta(ocfg)  # nếu hàm này yêu cầu OmegaConf, truyền ocfg

    # --- Chuẩn bị đường dẫn lưu ---
    model_tag = output_dir / Path(args.config).stem
    model_save_path = model_tag / "weights"
    model_save_path.mkdir(parents=True, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # --- Khởi tạo System (đã sửa để nhận dict) ---
    _system_mod = import_module(f"systems.{config['pl_system']}")
    System = getattr(_system_mod, "System")
    system = System(config)

    # --- Loggers & Callbacks ---
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=model_tag.as_posix(), version=1, name="tsbd_logs"),
        pl.loggers.CSVLogger(save_dir=model_tag.as_posix(), version=1, name="csv_logs"),
    ]

    callbacks = [
        pl.callbacks.ModelSummary(max_depth=3),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_save_path.as_posix(),
            filename="{epoch}-{sasv_eer_dev:.5f}",
            monitor="sasv_eer_dev",     # khớp với validation_epoch_end
            mode="min",
            every_n_epochs=config["val_interval_epoch"],
            save_top_k=config["save_top_k"],
        ),
    ]

    # --- GPU chọn lọc như cũ ---
    gpus = find_gpus(config["ngpus"], min_req_mem=config.get("min_req_mem", None))
    if gpus == -1:
        raise ValueError("Required GPUs are not available")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # --- Trainer (cập nhật tham số theo dict) ---
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["ngpus"],
        callbacks=callbacks,
        logger=logger,
        strategy="ddp",
        sync_batchnorm=True,
        max_epochs=config["epoch"],
        fast_dev_run=config["fast_dev_run"],
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        gradient_clip_val=config.get("gradient_clip", 0) or 0,
        # progress_bar_refresh_rate đã deprecated trong PL mới → bỏ
        reload_dataloaders_every_n_epochs=config["loader"].get("reload_every_n_epoch", 0) or 0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(system)
    #trainer.test(system, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SASVC Baseline (refactored).")
    # parser.add_argument("--config", type=str, required=True,
    #                     help="path to .conf (OmegaConf) file")
    parser.add_argument("--output_dir", type=str, default="./exp_result",
                        help="output directory")
    main(parser.parse_args())
