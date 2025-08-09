import random
from typing import Dict, List

from torch.utils.data import Dataset


class SASV_Trainset(Dataset):
    def __init__(self, cm_embd, asv_embd, spk_meta):
        self.cm_embd = cm_embd
        self.asv_embd = asv_embd
        self.spk_meta = spk_meta

    def __len__(self):
        return len(self.cm_embd.keys())

    def __getitem__(self, index):

        ans_type = random.randint(0, 1)
        if ans_type == 1:  # target
            spk = random.choice(list(self.spk_meta.keys()))
            enr, tst = random.sample(self.spk_meta[spk]["bonafide"], 2)

        elif ans_type == 0:  # nontarget
            nontarget_type = random.randint(1, 2)

            if nontarget_type == 1:  # zero-effort nontarget
                spk, ze_spk = random.sample(list(self.spk_meta.keys()), 2)
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[ze_spk]["bonafide"])

            if nontarget_type == 2:  # spoof nontarget
                spk = random.choice(list(self.spk_meta.keys()))
                if len(self.spk_meta[spk]["spoof"]) == 0:
                    while True:
                        spk = random.choice(list(self.spk_meta.keys()))
                        if len(self.spk_meta[spk]["spoof"]) != 0:
                            break
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[spk]["spoof"])

        return self.asv_embd[enr], self.asv_embd[tst], self.cm_embd[tst], ans_type


import os
from torch.utils.data import Dataset
import torch

def uid(p): return os.path.splitext(os.path.basename(p))[0]

def uid(p: str) -> str:
    # path -> UID (basename không đuôi); nếu đã là UID thì giữ nguyên
    return os.path.splitext(os.path.basename(p))[0] if ("/" in p or "\\" in p or p.endswith(".wav")) else p

class SASV_Evalset(Dataset):
    def __init__(self, utt_list, spk_model, asv_embd, cm_embd, return_key=True):
        self.utt_list = utt_list
        self.spk_model = spk_model
        self.asv_embd = asv_embd
        self.cm_embd = cm_embd
        self.return_key = return_key  # để ghi submission nếu cần

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, index):
        line = self.utt_list[index].strip()
        parts = line.split()

        # 4: spkmd key _ label
        # 3: spkmd key label  (UID)
        # 3: enroll_path test_path label  (path -> UID)
        if len(parts) == 4:
            spkmd, key, _, label = parts
        elif len(parts) == 3:
            a, b, label = parts
            spkmd, key = uid(a), uid(b)
        else:
            raise ValueError(f"Bad line format: {line}")

        # Chuẩn hoá label (nhận non-target, non_target, số…)
        lab = label.strip().lower().replace("-", "").replace("_", "")
        mapping = {
            "1": "target", "target": "target", "bonafide": "target", "genuine": "target",
            "0": "nontarget", "nontarget": "nontarget", "nonmatch": "nontarget", "impostor": "nontarget",
            "2": "spoof", "spoof": "spoof", "attack": "spoof"
        }
        if lab not in mapping:
            raise ValueError(f"Unknown label {label} (normalized='{lab}')")
        key_type = mapping[lab]

        try:
            spk = torch.as_tensor(self.spk_model[spkmd], dtype=torch.float32)
            asv = torch.as_tensor(self.asv_embd[key],   dtype=torch.float32)
            cm  = torch.as_tensor(self.cm_embd[key],    dtype=torch.float32)
        except KeyError as e:
            raise KeyError(
                f"Missing {e}. have_spk={spkmd in self.spk_model} "
                f"have_asv={key in self.asv_embd} have_cm={key in self.cm_embd} "
                f"(spkmd='{spkmd}', key='{key}')"
            )

        if self.return_key:
            # tiện cho test_epoch_end ghi submit theo key
            return {"spk": spk, "asv": asv, "cm": cm, "key": key, "label": key_type}
        else:
            return spk, asv, cm, key_type

    
class SASV_DevEvalset(Dataset):
    def __init__(self, utt_list, spk_model, asv_embd, cm_embd):
        self.utt_list = utt_list
        self.spk_model = spk_model
        self.asv_embd = asv_embd
        self.cm_embd = cm_embd
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, index):
        line = self.utt_list[index].strip()
        parts = line.split()

        if len(parts) == 4:
            spkmd, key, _, label = parts
        elif len(parts) == 3:
            spkmd, key, label = parts
        else:
            raise ValueError(f"Bad line format: {line}")

        # chấp nhận cả số lẫn chữ
        mapping = {"1": "target", "0": "nontarget", "2": "spoof",
                   "target": "target", "nontarget": "nontarget", "spoof": "spoof"}
        if label not in mapping:
            raise ValueError(f"Unknown label {label}")

        key_type = mapping[label]
        return self.spk_model[spkmd], self.asv_embd[key], self.cm_embd[key], key_type
    





def get_trnset(
    cm_embd_trn: Dict, asv_embd_trn: Dict, spk_meta_trn: Dict
) -> SASV_DevEvalset:
    return SASV_Trainset(
        cm_embd=cm_embd_trn, asv_embd=asv_embd_trn, spk_meta=spk_meta_trn
    )


def get_dev_evalset(
    utt_list: List, cm_embd: Dict, asv_embd: Dict, spk_model: Dict
) -> SASV_DevEvalset:
    return SASV_DevEvalset(
        utt_list=utt_list, cm_embd=cm_embd, asv_embd=asv_embd, spk_model=spk_model
    )
