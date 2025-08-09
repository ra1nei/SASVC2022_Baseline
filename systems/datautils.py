import os, random, torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def build_spk_meta_from_embeds(asv_embd, cm_embd=None, require_in_both=True):
    """
    Tạo spk_meta theo dạng:
      { spk: {"bonafide":[...], "spoof":[...]} }
    - Nếu require_in_both=True: chỉ nhận key có mặt ở cả ASV và CM (an toàn khi train fusion)
    """
    keys = set(asv_embd.keys())
    if cm_embd is not None and require_in_both:
        keys = keys & set(cm_embd.keys())

    spk_meta = defaultdict(lambda: {"bonafide": [], "spoof": []})
    for k in keys:
        spk, cls, _ = k.split("/", 2)  # "idxxxx/bonafide/xxx.wav"
        if cls not in ("bonafide", "spoof"): 
            continue
        spk_meta[spk][cls].append(k)
    return dict(spk_meta)


def split_speakers(spk_meta, train_ratio=0.8, seed=42, require_min_bona=2):
    random.seed(seed)
    speakers = [spk for spk, v in spk_meta.items() if len(v["bonafide"]) >= require_min_bona]
    random.shuffle(speakers)
    k = int(len(speakers) * train_ratio)
    train_spk = set(speakers[:k])
    val_spk   = set(speakers[k:])
    return train_spk, val_spk


def filter_spk_meta(spk_meta, speakers):
    out = {}
    for spk in speakers:
        if spk in spk_meta:
            out[spk] = {
                "bonafide": list(spk_meta[spk]["bonafide"]),
                "spoof":    list(spk_meta[spk]["spoof"])
            }
    return out

def filter_emb_dict(emb_dict, speakers):
    out = {}
    for k, v in emb_dict.items():
        spk = k.split("/")[0]  # "idxxxx/bonafide/xxx.wav"
        if spk in speakers:
            out[k] = v
    return out


def build_val_utt_list(spk_meta_val):
    utt_list = []
    speakers = list(spk_meta_val.keys())
    for spk, v in spk_meta_val.items():
        # target (bona cùng speaker)
        for k in v["bonafide"]:
            utt_list.append(f"{spk} {k} - 1")
        # spoof nontarget (spoof cùng speaker)
        for k in v["spoof"]:
            utt_list.append(f"{spk} {k} - 0")
        # zero-effort (bona khác speaker, chọn ngẫu nhiên 1-2 mỗi speaker)
        import random
        other_spk = speakers.copy()
        if spk in other_spk: other_spk.remove(spk)
        random.shuffle(other_spk)
        for zspk in other_spk[:2]:  # mỗi spk lấy 2 mẫu zero-effort
            if spk_meta_val[zspk]["bonafide"]:
                k = random.choice(spk_meta_val[zspk]["bonafide"])
                utt_list.append(f"{spk} {k} - 0")
    return utt_list


def build_spk_model(spk_meta_split, asv_embd):
    spk_model = {}
    for spk, v in spk_meta_split.items():
        bona_keys = [k for k in v["bonafide"] if k in asv_embd]
        if not bona_keys:
            continue
        vecs = [asv_embd[k] for k in bona_keys]
        mean = np.mean(np.stack(vecs, axis=0), axis=0)
        spk_model[spk] = mean.astype(np.float32)
    return spk_model

def load_embedding_dict(root_dir):
    emb_dict = {}
    speakers = os.listdir(root_dir)

    for spk in tqdm(speakers, desc=f"Loading embeddings from {root_dir}"):
        for label_type in ["bonafide", "spoof"]:
            sub_dir = os.path.join(root_dir, spk, label_type)
            if not os.path.exists(sub_dir):
                continue
            for fname in os.listdir(sub_dir):
                if fname.endswith(".npy"):
                    key = f"{spk}/{label_type}/{fname.replace('.npy', '.wav')}"
                    emb_dict[key] = np.load(os.path.join(sub_dir, fname))
    return emb_dict