import torch
from tqdm import tqdm
import numpy as np
from my_model.dataset_ESC50 import load_test_data
from model import resnet18
from resnet_train import TwoChannelInputResNet
import my_model.s_i_tft as ft
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    set_seed(42)
    DEVICE = torch.device("cuda:0")
    clean_dir = ""
    noise_dir = ''

    weight_path = r""
    batch_size = 4

    sample_rate = 8000
    crop_len = 1600
    hop_size = 32
    win_size = 400
    res_model = resnet18(4)
    model = TwoChannelInputResNet(res_model).to(device=DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    # 加载数据
    test_ds =load_test_data(clean_dir,noise_dir,(-10.0,-10.0),sample_rate,crop_len,batch_size)
    test_loader = tqdm(test_ds, desc='Evaluating', unit='batch')


    tatal_acc = 0.0
    with torch.no_grad():
        for clean, noisy, labels in test_loader:
            clean = clean.to(DEVICE)
            noisy = noisy.to(DEVICE)
            labels = labels.to(DEVICE)
            # STFT
            n_mag, n_pha, _, _ = ft.torch_mag_pha_stft(noisy, hop_size=hop_size, win_size=win_size)
            n_input = torch.stack((n_mag.permute(0, 2, 1), n_pha.permute(0, 2, 1)), dim=1)

            # 推理
            cls_out = model(n_input)
            _, est_labels = cls_out.max(1)
            correct_labels = (est_labels == labels).sum().item()
            out_len = labels.size(0)
            acc = correct_labels / out_len
            tatal_acc += acc


            # 清理变量
            del clean, noisy, n_mag, n_pha, n_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    avg_acc = tatal_acc / len(test_ds)
    print(f"平均准确率 : {avg_acc:.4f}")



if __name__ == '__main__':
    main()
