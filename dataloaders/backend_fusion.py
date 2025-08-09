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

class SASV_Evalset(Dataset):
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

        # Hỗ trợ 3 dạng:
        # 4: spkmd key _ label
        # 3: spkmd key label      (đã là UID)
        # 3: enroll_path test_path label  (sẽ convert sang UID)
        if len(parts) == 4:
            spkmd, key, _, label = parts
        elif len(parts) == 3:
            a, b, label = parts
            # nếu a/b có '/' hoặc '.wav' -> coi như path, convert sang UID
            if ("/" in a or "\\" in a or a.endswith(".wav")) or ("/" in b or "\\" in b or b.endswith(".wav")):
                spkmd, key = uid(a), uid(b)
            else:
                spkmd, key = a, b
        else:
            raise ValueError(f"Bad line format: {line}")

        # map label (chuỗi) để dùng lại get_all_EERs sau này
        mapping = {"1": "target", "0": "nontarget", "2": "spoof",
                   "target": "target", "nontarget": "nontarget", "spoof": "spoof"}
        if label not in mapping:
            raise ValueError(f"Unknown label {label}")
        key_type = mapping[label]

        try:
            spk = torch.as_tensor(self.spk_model[spkmd], dtype=torch.float32)
            asv = torch.as_tensor(self.asv_embd[key], dtype=torch.float32)
            cm  = torch.as_tensor(self.cm_embd[key], dtype=torch.float32)
        except KeyError as e:
            raise KeyError(
                f"Missing {e}. have_spk={spkmd in self.spk_model} "
                f"have_asv={key in self.asv_embd} have_cm={key in self.cm_embd}"
            )

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
