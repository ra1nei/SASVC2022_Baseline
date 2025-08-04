import os
import random

TRAIN_META = "/kaggle/input/vsasv-train/train_vlsp_2025_metadata.txt"
EVAL_META  = "/kaggle/input/public-test-vsasv/public_test_vlsp.txt"
SAVE_DIR   = "/kaggle/working/protocols"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Đọc toàn bộ train metadata ---
with open(TRAIN_META, "r") as f:
    lines = f.readlines()

# Gom theo speaker
spk_to_lines = {}
for line in lines:
    spk = line.strip().split()[0]
    spk_to_lines.setdefault(spk, []).append(line)

# Shuffle speaker list
speakers = list(spk_to_lines.keys())
random.seed(42)
random.shuffle(speakers)

# Chia 80% train, 20% dev
split = int(0.8 * len(speakers))
train_spk = speakers[:split]
dev_spk   = speakers[split:]

# Lấy dòng theo speaker
train_lines = [l for spk in train_spk for l in spk_to_lines[spk]]
dev_lines   = [l for spk in dev_spk for l in spk_to_lines[spk]]

# --- Ghi train/dev list ---
with open(f"{SAVE_DIR}/cm_trn_list.txt", "w") as f:
    f.writelines(train_lines)

with open(f"{SAVE_DIR}/cm_dev_list.txt", "w") as f:
    f.writelines(dev_lines)

# --- Eval giữ nguyên từ file riêng ---
os.system(f"cp {EVAL_META} {SAVE_DIR}/cm_eval_list.txt")

print(f"✅ Train speakers: {len(train_spk)}, Dev speakers: {len(dev_spk)}")
print(f"Files saved in {SAVE_DIR}")
