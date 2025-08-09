import math
import os
import pickle as pk
from tqdm import tqdm
from importlib import import_module
from typing import Any
import numpy as np
import pytorch_lightning as pl
import schedulers as lr_schedulers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import get_all_EERs
from utils import keras_decay
from datautils import *
from dataloaders.backend_fusion import SASV_Trainset, SASV_DevEvalset




class System(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        """
        config: dict hoặc omegaconf.DictConfig
        """
        super().__init__(*args, **kwargs)
        self.config = config

        # Load model từ config
        _model_module = import_module(f"models.{self.config['model_arch']}")
        _model_class = getattr(_model_module, "Model")
        self.model = _model_class(self.config["model_config"])

        # Cấu hình loss
        self.configure_loss()

        # Lưu hyperparams (nếu config là dict thì convert sang str)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def training_step(self, batch, batch_idx, dataloader_idx=-1):
        embd_asv_enrol, embd_asv_test, embd_cm_test, label = batch
        pred = self.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
        loss = self.loss(pred, label)
        self.log(
            "trn_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        embd_asv_enrol, embd_asv_test, embd_cm_test, key = batch
        pred = self.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
        pred = torch.softmax(pred, dim=-1)

        return {"pred": pred, "key": key}

    def validation_epoch_end(self, outputs):
        log_dict = {}
        preds, keys = [], []
        for output in outputs:
            preds.append(output["pred"])
            keys.extend(list(output["key"]))

        preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)

        log_dict["sasv_eer_dev"] = sasv_eer
        log_dict["sv_eer_dev"] = sv_eer
        log_dict["spf_eer_dev"] = spf_eer

        self.log_dict(log_dict)

    def test_step(self, batch, batch_idx, dataloader_idx=-1):
        res_dict = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        return res_dict

    def test_epoch_end(self, outputs):
        log_dict = {}
        preds, keys = [], []
        for output in outputs:
            preds.append(output["pred"])
            keys.extend(list(output["key"]))

        preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)

        log_dict["sasv_eer_eval"] = sasv_eer
        log_dict["sv_eer_eval"] = sv_eer
        log_dict["spf_eer_eval"] = spf_eer

        self.log_dict(log_dict)


    def configure_optimizers(self):
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.wd,
            )
        elif self.config.optimizer.lowe() == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.config.optim.lr,
                momentum=self.config.optim.momentum,
                weight_decay=self.config.optim.wd,
            )
        else:
            raise NotImplementedError("....")

        if self.config.optim.scheduler.lower() == "sgdr_cos_anl":
            assert (
                self.config.optim.n_epoch_per_cycle is not None
                and self.config.optim.min_lr is not None
                and self.config.optim.warmup_steps is not None
                and self.config.optim.lr_mult_after_cycle is not None
            )
            lr_scheduler = lr_schedulers.CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=len(self.train_dataloader())
                // self.config.ngpus
                * self.config.optim.n_epoch_per_cycle,
                cycle_mult=1.0,
                max_lr=self.config.optim.lr,
                min_lr=self.config.optim.min_lr,
                warmup_steps=self.config.optim.warmup_steps,
                gamma=self.config.optim.lr_mult_after_cycle,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif self.config.optim.scheduler.lower() == "reduce_on_plateau":
            assert (
                self.config.optim.lr is not None
                and self.config.optim.min_lr is not None
                and self.config.optim.factor is not None
                and self.config.optim.patience is not None
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.config.optim.factor,
                patience=self.config.optim.patience,
                min_lr=self.config.optim.min_lr,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                    "monitor": "dev_sasv_eer",
                },
            }
        elif self.config.optim.scheduler.lower() == "keras":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: keras_decay(step)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                },
            }

        else:
            raise NotImplementedError(".....")

    def setup(self, stage: str | None = None):
        """
        Chuẩn bị dữ liệu cho các stage: ["fit", "validate", "test"].
        - Dùng biến trong RAM thay vì đọc file protocol.
        - Hỗ trợ config dạng dict.
        """

        # 1) Nạp meta & embeddings (nếu bạn đã có sẵn các biến này ngoài class
        #    thì có thể bỏ 2 dòng dưới, hoặc để các hàm này chỉ set self.asv_embd, self.cm_embd)
        #self.load_meta_information()   # optional: có thể để trống nếu không dùng
        self.load_embeddings()         # đảm bảo set: self.asv_embd, self.cm_embd

        # 2) Nếu chưa có spk_meta_all thì build từ key của embeddings (đảm bảo đồng bộ ASV/CM)
        if not hasattr(self, "spk_meta_all"):
            self.spk_meta_all = build_spk_meta_from_embeds(
                asv_embd=self.asv_embd,
                cm_embd=self.cm_embd,
                require_in_both=True
            )

        # 3) Stage FIT: tạo split 80/20 + chuẩn bị biến dùng cho train/val
        if stage in ("fit", None):
            # Nếu bạn đã có các biến split sẵn thì bỏ khúc này đi
            if not (hasattr(self, "spk_meta_trn") and hasattr(self, "spk_meta_val")):
                train_spk, val_spk = split_speakers(self.spk_meta_all, train_ratio=0.8, seed=42)
                self.spk_meta_trn = filter_spk_meta(self.spk_meta_all, train_spk)
                self.spk_meta_val = filter_spk_meta(self.spk_meta_all, val_spk)

                self.asv_embd_trn = filter_emb_dict(self.asv_embd, train_spk)
                self.cm_embd_trn  = filter_emb_dict(self.cm_embd,  train_spk)
                self.asv_embd_val = filter_emb_dict(self.asv_embd, val_spk)
                self.cm_embd_val  = filter_emb_dict(self.cm_embd,  val_spk)

                self.spk_model_val = build_spk_model(self.spk_meta_val, self.asv_embd_val)
                self.utt_list_val  = build_val_utt_list(self.spk_meta_val)

            # Gán “hàm tạo dataset” trực tiếp bằng class đã có
            self.ds_func_trn = SASV_Trainset
            self.ds_func_dev = SASV_DevEvalset

        # 4) Stage VALIDATE: dùng dev set từ biến RAM (không đọc file)
        elif stage == "validate":
            # Yêu cầu các biến này đã có từ fit hoặc bạn tự gán trước khi gọi validate()
            assert hasattr(self, "utt_list_val") and hasattr(self, "spk_model_val"), \
                "utt_list_val / spk_model_val chưa có. Hãy build ở fit hoặc gán trước validate."
            self.ds_func_dev = SASV_DevEvalset

        # 5) Stage TEST: tương tự validate nhưng dùng biến eval
        elif stage == "test":
            # Bạn cần tự chuẩn bị self.utt_list_eval / self.spk_model_eval / *_eval embeddings
            # (có thể build giống phần val hoặc đọc từ nơi bạn đã chuẩn hoá)
            assert hasattr(self, "utt_list_eval") and hasattr(self, "spk_model_eval"), \
                "utt_list_eval / spk_model_eval chưa có. Hãy build/gán trước test."
            self.ds_func_eval = SASV_DevEvalset

        else:
            raise NotImplementedError(f"Unsupported stage: {stage}")


    def train_dataloader(self):
        self.train_ds = SASV_Trainset(
            self.cm_embd_trn, 
            self.asv_embd_trn, 
            self.spk_meta_trn
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=self.config["loader"]["n_workers"],
        )

    def val_dataloader(self):
        self.dev_ds = SASV_DevEvalset(
            self.utt_list_val,        # biến có sẵn, không đọc file
            self.spk_model_val,
            self.asv_embd_val,
            self.cm_embd_val
        )
        return DataLoader(
            self.dev_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=self.config["loader"]["n_workers"],
        )

    def test_dataloader(self):
        with open(self.config.dirs.sasv_eval_trial, "r") as f:
            sasv_eval_trial = f.readlines()
        self.eval_ds = self.ds_func_eval(
            sasv_eval_trial, self.cm_embd_eval, self.asv_embd_eval, self.spk_model_eval)
        return DataLoader(
            self.eval_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
        )

    def configure_loss(self):
        if self.config.loss.lower() == "bce":
            self.loss = F.binary_cross_entropy_with_logits
        if self.config.loss.lower() == "cce":
            self.loss = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(self.config.loss_weight)
            )
        else:
            raise NotImplementedError("!")

    def load_meta_information(self):
        with open(self.config.dirs.spk_meta + "spk_meta_trn.pk", "rb") as f:
            self.spk_meta_trn = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_dev.pk", "rb") as f:
            self.spk_meta_dev = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_eval.pk", "rb") as f:
            self.spk_meta_eval = pk.load(f)




    def load_embeddings(self):
        """
        Gán embeddings vào class.
        Nếu đã load sẵn ngoài code thì truyền vào self trước khi gọi setup().
        """
        if hasattr(self, "asv_embd") and hasattr(self, "cm_embd"):
            # Đã có embeddings trong RAM → không cần load lại
            return

        # Nếu chưa có, bạn có thể load từ đường dẫn trong config
        from pathlib import Path

        emb_dir_asv = Path(self.config["dirs"]["asv_embedding"])
        emb_dir_cm  = Path(self.config["dirs"]["cm_embedding"])

        self.asv_embd = load_embedding_dict(str(emb_dir_asv))
        self.cm_embd  = load_embedding_dict(str(emb_dir_cm))

