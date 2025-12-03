import torch
import torch.utils.data
import torch.nn.functional as F
import torchaudio.transforms as T
import os
import torchaudio
import random
import numpy as np
from torch.utils.data import random_split,DataLoader



class GIS_Dataset(torch.utils.data.Dataset):

    def __init__(self, clean_dir, noise_dir, snr_range, sr, crop_len):
        self.crop_len = crop_len
        self.sr = sr
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.clean = []
        self.noise = []
        self.snr_db = snr_range
        self.resample = T.Resample(48000, sr)
        for label, folder in enumerate(sorted(os.listdir(self.clean_dir))):
            folder_path = os.path.join(self.clean_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        self.clean.append((file_path, label))

        for _, folder in enumerate(sorted(os.listdir(self.noise_dir))):
            folder_path = os.path.join(self.noise_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        self.noise.append(file_path)

    def __len__(self):
        return len(self.clean)


    def __getitem__(self, idx):
        file_path, label = self.clean[idx]
        clean_wav, sr = torchaudio.load(file_path)
        clean_wav = self.resample(clean_wav)
        noise_path = random.choice(self.noise)
        noise_wav, _ = torchaudio.load(noise_path)
        noise_wav = noise_wav[:,0: 2*self.sr]

        if clean_wav.shape[1] > self.crop_len:
            clean_wav = clean_wav[:, :self.crop_len]
        if noise_wav.shape[1] > self.crop_len:
            max_start = noise_wav.shape[1] - self.crop_len
            start = random.randint(0, max_start)
            noise_wav = noise_wav[:, start:start + self.crop_len]

        snr = random.uniform(*self.snr_db)
        clean_pow = torch.mean(torch.pow(clean_wav,2))
        noise_pow = torch.mean(torch.pow(noise_wav,2))
        scale = torch.sqrt(clean_pow / (noise_pow * 10 ** (snr / 10) + 1e-7))
        noisy_wav = clean_wav + noise_wav * scale

        clean_wav = (clean_wav - clean_wav.mean()) / (clean_wav.std() + 1e-8)
        noisy_wav = (noisy_wav - noisy_wav.mean()) / (noisy_wav.std() + 1e-8)

        clean_wav = clean_wav.squeeze(0).to(dtype=torch.float32)
        noisy_wav = noisy_wav.squeeze(0).to(dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return clean_wav, noisy_wav, label


def worker_init_fn(worker_id):  #噪声选择、切片、信噪比皆为随机
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)
    torch.cuda.manual_seed_all(42 + worker_id)


def n_load_data(clean_dir1, clean_dir2, noise_dir1, noise_dir2, snr_range, sr, crop_len, batch_size):
    train_dataset = GIS_Dataset(clean_dir1, noise_dir1, snr_range, sr, crop_len)
    val_dataset = GIS_Dataset(clean_dir2,noise_dir2,snr_range, sr, crop_len)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True, #?
        num_workers= 4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False, #?
        drop_last=True,
        num_workers= 4,
        worker_init_fn = worker_init_fn(0)
    )

    return train_loader, val_loader

def load_test_data(clean, noise, snr, sr, croplen, batch_size):
    val_dataset = GIS_Dataset(clean,noise,snr,sr,croplen)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False, #?
        drop_last=True,
        num_workers= 4
    )
    return val_loader

# 主函数
def main():
    filedir = r""
    noisedir = r""
    # batch_size = 4
    #


if __name__ == "__main__":
    main()







