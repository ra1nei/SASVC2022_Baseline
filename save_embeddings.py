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

from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters


def save_embeddings_from_vsasv(protocol_path, data_root, cm_embd_ext, asv_embd_ext, device):
    utt2path = {}

    with open(protocol_path, "r") as f:
        for line in f:
            
            ### DEBUGGING
            parts = line.strip().split()
            if len(parts) == 4:
                spk_id, utt_name, _, cm_label = parts
            elif len(parts) == 3:
                spk_id, utt_name, cm_label = parts
            else:
                raise ValueError(f"Invalid line format: {line}")

            # audio_path = os.path.join(data_root, spk_id, cm_label, utt_name)
            audio_path = os.path.join(data_root, utt_name)
            utt_id = f"{spk_id}_{utt_name}"
            utt2path[utt_id] = audio_path

    class VSASVDataset(torch.utils.data.Dataset):
        def __init__(self, utt2path):
            self.keys = list(utt2path.keys())
            self.paths = [utt2path[k] for k in self.keys]

        def __getitem__(self, idx):
            wav, _ = torchaudio.load(self.paths[idx])
            return wav.squeeze(0), self.keys[idx]  # mono

        def __len__(self):
            return len(self.keys)

    dataset = VSASVDataset(utt2path)
    loader = DataLoader(dataset, batch_size=30, shuffle=False)

    cm_emb_dic, asv_emb_dic = {}, {}
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-aasist_config", type=str, default="./aasist/config/AASIST.conf")
    parser.add_argument("-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth")
    parser.add_argument("-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model")
    parser.add_argument("-protocol_path", type=str, required=True)
    parser.add_argument("-data_root", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.aasist_config, "r") as f_json:
        config = json.load(f_json)
    model_config = config["model_config"]

    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.eval()

    asv_embd_ext = ECAPA_TDNN(C=1024).to(device)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.eval()

    save_embeddings_from_vsasv(args.protocol_path, args.data_root, cm_embd_ext, asv_embd_ext, device)


if __name__ == "__main__":
    print("Running")
    main()
    print("Finish")