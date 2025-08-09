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


def main(config, output_dir="./exp_result"):
    output_dir = Path(output_dir)
    pl.seed_everything(config["seed"], workers=True)

    # --- Chuẩn bị đường dẫn lưu ---
    model_tag = output_dir / "my_model"
    model_save_path = model_tag / "weights"
    model_save_path.mkdir(parents=True, exist_ok=True)

    # --- Khởi tạo System ---
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
            monitor="sasv_eer_dev",
            mode="min",
            every_n_epochs=config["val_interval_epoch"],
            save_top_k=config["save_top_k"],
        ),
    ]


    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        callbacks=callbacks,
        logger=logger,
        #strategy="ddp_notebook", # Change to ddp if not using notebook
        sync_batchnorm=True,
        max_epochs=config["epoch"],
        fast_dev_run=config["fast_dev_run"],
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        gradient_clip_val=config.get("gradient_clip", 0) or 0,
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
