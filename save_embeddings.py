import argparse
import json
import os
import pickle as pk
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from aasist.data_utils import Dataset_ASVspoof2019_devNeval
from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

# list of dataset partitions
SET_PARTITION = ["trn", "dev", "eval"]

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "trn": "protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

# directories of each dataset partition
SET_DIR = {
    "trn": "./LA/ASVspoof2019_LA_train/",
    "dev": "./LA/ASVspoof2019_LA_dev/",
    "eval": "./LA/ASVspoof2019_LA_eval/",
}

# enrolment data list for speaker model calculation
# each speaker model comprises multiple enrolment utterances
SET_TRN = {
    "dev": [
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt",
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt",
    ],
    "eval": [
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt",
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt",
    ],
}


def save_embeddings(
    set_name, cm_embd_ext, asv_embd_ext, device
):
    meta_lines = open(SET_CM_PROTOCOL[set_name], "r").readlines()
    utt2spk = {}
    utt_list = []
    for line in meta_lines:
        tmp = line.strip().split(" ")

        spk = tmp[0]
        utt = tmp[1]

        if utt in utt2spk:
            print("Duplicated utt error", utt)

        utt2spk[utt] = spk
        utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )

    cm_emb_dic = {}
    asv_emb_dic = {}

    print("Getting embeddings from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb, _ = cm_embd_ext(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for k, cm_emb, asv_emb in zip(key, batch_cm_emb, batch_asv_emb):
            cm_emb_dic[k] = cm_emb
            asv_emb_dic[k] = asv_emb

    os.makedirs("embeddings", exist_ok=True)
    with open( "embeddings/cm_embd_%s.pk" % (set_name), "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open("embeddings/asv_embd_%s.pk" % (set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)

def save_embeddings_from_vsasv(protocol_path, data_root, cm_embd_ext, asv_embd_ext, device):
    utt2spk = {}
    utt2path = {}

    with open(protocol_path, "r") as f:
        for line in f:
            spk_id, utt_name, _, cm_label = line.strip().split()

            # Xây dựng đường dẫn đầy đủ đến file .wav
            audio_path = os.path.join(data_root, spk_id, cm_label, utt_name)
            utt_id = f"{spk_id}_{utt_name}"

            if utt_id in utt2spk:
                print("Duplicated utt_id", utt_id)

            utt2spk[utt_id] = spk_id
            utt2path[utt_id] = audio_path

    # Dataset custom đơn giản
    class VSASVDataset(torch.utils.data.Dataset):
        def __init__(self, utt2path):
            self.keys = list(utt2path.keys())
            self.paths = [utt2path[k] for k in self.keys]

        def __getitem__(self, idx):
            path = self.paths[idx]
            wav, _ = torchaudio.load(path)
            return wav.squeeze(0), self.keys[idx]  # mono

        def __len__(self):
            return len(self.keys)

    dataset = VSASVDataset(utt2path)
    loader = DataLoader(dataset, batch_size=30, shuffle=False, drop_last=False)

    cm_emb_dic = {}
    asv_emb_dic = {}

    print(f"Extracting embeddings from {protocol_path}...")

    for batch_x, keys in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb, _ = cm_embd_ext(batch_x)
            batch_asv_emb = asv_embd_ext(batch_x, aug=False)

        for k, cm_emb, asv_emb in zip(keys, batch_cm_emb.cpu(), batch_asv_emb.cpu()):
            cm_emb_dic[k] = cm_emb.numpy()
            asv_emb_dic[k] = asv_emb.numpy()

    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/cm_embd_vsasv.pk", "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open("embeddings/asv_embd_vsasv.pk", "wb") as f:
        pk.dump(asv_emb_dic, f)

def save_models(set_name, asv_embd_ext, device):
    utt2spk = {}
    utt_list = []

    for trn in SET_TRN[set_name]:
        meta_lines = open(trn, "r").readlines()

        for line in meta_lines:
            tmp = line.strip().split(" ")

            spk = tmp[0]
            utts = tmp[1].split(",")

            for utt in utts:
                if utt in utt2spk:
                    print("Duplicated utt error", utt)

                utt2spk[utt] = spk
                utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )
    asv_emb_dic = {}

    print("Getting embedgins from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for k, asv_emb in zip(key, batch_asv_emb):
            utt = k
            spk = utt2spk[utt]

            if spk not in asv_emb_dic:
                asv_emb_dic[spk] = []

            asv_emb_dic[spk].append(asv_emb)

    for spk in asv_emb_dic:
        asv_emb_dic[spk] = np.mean(asv_emb_dic[spk], axis=0)

    with open("embeddings/spk_model.pk_%s" % (set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-aasist_config", type=str, default="./aasist/config/AASIST.conf"
    )
    parser.add_argument(
        "-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth"
    )
    parser.add_argument(
        "-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model"
    )

    return parser.parse_args()

def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.aasist_config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]

    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.eval()

    asv_embd_ext = ECAPA_TDNN(C=1024).to(device)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.eval()

    protocol_path = "vsasv_train.txt"
    data_root = "VSASV-Dataset/vlsp2025/train"

    save_embeddings_from_vsasv(protocol_path, data_root, cm_embd_ext, asv_embd_ext, device)

if __name__ == "__main__":
    main()
