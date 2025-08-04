import os
from pathlib import Path
import random

DATA_ROOT = "VSASV-Dataset/vlsp2025/train"
SAVE_DIR = "protocols_vsasv"
os.makedirs(SAVE_DIR, exist_ok=True)

utt_list = []

for speaker in os.listdir(DATA_ROOT):
    for cm_label in ["bonafide", "spoof"]:
        folder = Path(DATA_ROOT) / speaker / cm_label
        if not folder.exists():
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".wav"):
                utt_id = f"{speaker}_{fname}"
                utt_list.append({
                    "spk": speaker,
                    "utt": utt_id,
                    "wav_path": str(folder / fname),
                    "cm_label": cm_label,
                    "attack_id": "-" if cm_label == "bonafide" else "A00",  # gán tạm
                })

# Shuffle và chia train/dev/eval theo tỉ lệ
random.seed(42)
random.shuffle(utt_list)

N = len(utt_list)
train_utt = utt_list[:int(0.6 * N)]
dev_utt = utt_list[int(0.6 * N):int(0.8 * N)]
eval_utt = utt_list[int(0.8 * N):]

splits = {"train": train_utt, "dev": dev_utt, "eval": eval_utt}


def write_cm_protocol(split, utts):
    fname = f"{SAVE_DIR}/vsasv.cm.{split}.trl.txt" if split != "train" else f"{SAVE_DIR}/vsasv.cm.train.trn.txt"
    with open(fname, "w") as f:
        for u in utts:
            f.write(f"{u['spk']} {u['utt']} {u['attack_id']} {u['cm_label']}\n")


def write_asv_sasv_protocol(split, utts):
    fname = f"{SAVE_DIR}/vsasv.asv.{split}.gi.trl.txt"
    sasv_fname = f"{SAVE_DIR}/vsasv.sasv.{split}.trl.txt"
    with open(fname, "w") as fasv, open(sasv_fname, "w") as fsasv:
        for u in utts:
            asv_label = "target" if u["cm_label"] == "bonafide" else "spoof"
            fasv.write(f"{u['spk']} {u['utt']} {u['cm_label']} {asv_label}\n")
            fsasv.write(f"{u['spk']} {u['utt']} {u['cm_label']} {asv_label}\n")


for split, items in splits.items():
    write_cm_protocol(split, items)
    write_asv_sasv_protocol(split, items)

print("finish", SAVE_DIR)
