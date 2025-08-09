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
        # target
        for k in v["bonafide"]:
            utt_list.append(f"{spk} {k} target")

        # spoof
        for k in v["spoof"]:
            utt_list.append(f"{spk} {k} spoof")

        # zero-effort nontarget
        other_spk = speakers.copy()
        if spk in other_spk:
            other_spk.remove(spk)
        random.shuffle(other_spk)
        for zspk in other_spk[:2]:
            if spk_meta_val[zspk]["bonafide"]:
                k = random.choice(spk_meta_val[zspk]["bonafide"])
                utt_list.append(f"{spk} {k} nontarget")
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

def load_protocol_lines(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines  # mỗi dòng dạng "spkmd key ... label"

def build_spk_model_from_trials(utt_list, asv_embd, use_bonafide_only=True):
    """
    utt_list: list dòng trial. Hỗ trợ 3 kiểu:
      1) "spkID key - 1|0|2"
      2) "spkID key label"           (label: 1/0/2 hoặc target/nontarget/spoof)
      3) "enrol_key test_key label"  (không có spkID, sẽ lấy spk từ enrol_key)
    """
    from collections import defaultdict
    import numpy as np

    # gom danh sách enrol theo speaker
    spk2keys = defaultdict(list)

    for line in utt_list:
        parts = line.strip().split()
        if not parts:
            continue

        if len(parts) == 4:
            # spkID key _ label
            spk, key, _, lab = parts
        elif len(parts) == 3:
            a, b, c = parts
            # TH1: "spk key label"
            if "/" in b and ("/bonafide/" in b or "/spoof/" in b):
                spk, key, lab = a, b, c
            else:
                # TH2: "enrol_key test_key label"
                enrol_key, test_key, lab = a, b, c
                # lấy spk từ enrol_key (tiền tố trước dấu '/')
                spk = enrol_key.split("/", 1)[0]
                key = enrol_key
        else:
            # fallback tối giản: lấy speaker từ key ở cột 1
            key = parts[0]
            spk = key.split("/", 1)[0]
            lab = parts[-1]

        lab_map = {"1":"target","0":"nontarget","2":"spoof",
                   "target":"target","nontarget":"nontarget","spoof":"spoof"}
        lab = lab_map.get(lab, None)

        # chỉ dùng bonafide nếu cần
        if use_bonafide_only and "/bonafide/" not in key:
            continue

        if key in asv_embd:
            spk2keys[spk].append(key)

    # tính mean embedding per speaker
    spk_model = {}
    for spk, keys in spk2keys.items():
        if not keys: 
            continue
        vecs = [asv_embd[k] for k in keys if k in asv_embd]
        if not vecs:
            continue
        mean = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        spk_model[spk] = mean

    return spk_model


