import torch
from tqdm import tqdm
import numpy as np
from my_model.dataset_ESC50 import load_test_data
from my_model.generator import generator_new
import my_model.s_i_tft as ft
import random
import matplotlib.pyplot as plt
import csv
from create_model import load_model

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
    batch_size = 4
    sample_rate = 8000
    crop_len = 1600
    hop_size = 32
    win_size = 400
    #name = ["my_model","my_model_detach",my_model_woCom,"MobileNetv3","EfficientNet_v2","xLSTM","LSTM","RNN","GRU","ConvTransPlus",
    # Resnet18,Resnet34,TCN,TSSequencer]
    modelname = "my_model"
    model = load_model(modelname)
    model.eval()
    snr_range_list = np.arange(-20.0, 5.5, 5)#分别验证每个信噪比下的模型准确率
    acc_list = []

    for snr in snr_range_list:
        print(f"\n=== 正在测试 SNR = {snr} dB ===")

        # 构建测试数据集
        test_ds = load_test_data(clean_dir, noise_dir, (snr, snr), sample_rate, crop_len, batch_size)
        test_loader = tqdm(test_ds, desc=f'Evaluating {snr}dB', unit='batch')

        tatal_acc = 0.0

        with torch.no_grad():
            for _, noisy, labels in test_loader:
                noisy = noisy.to(DEVICE)
                labels = labels.to(DEVICE)

####################不重要#############################
                # STFT
                n_mag, n_pha, _, _ = ft.torch_mag_pha_stft(noisy, hop_size=hop_size, win_size=win_size)
                n_input = torch.stack((n_mag.permute(0, 2, 1), n_pha.permute(0, 2, 1)), dim=1)
                # 推理
                if modelname == "my_model":
                    _, _, cls_out = model(n_input)
                elif modelname == "my_model_detach":
                    _, _, cls_out = model(n_input)
                elif modelname == "my_model_woCom":
                    _, _, cls_out = model(n_input)
                elif modelname == "LSTM":
                    _, _, cls_out = model(n_input)
                elif modelname == "RNN":
                    _, _, cls_out = model(n_input)
                elif modelname == "GRU":
                    _, _, cls_out = model(n_input)
                else:
                    cls_out = model(n_input)

                _, est_labels = cls_out.max(1)
###########################################################

                correct_labels = (est_labels == labels).sum().item()
                out_len = labels.size(0)
                acc = correct_labels / out_len
                tatal_acc += acc

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_acc = tatal_acc / len(test_ds)
        acc_list.append(avg_acc)
        print(f"[SNR={snr} dB] 平均准确率 : {avg_acc:.4f}")

    # === 结果可视化 ===
    plt.figure(figsize=(8, 6))
    plt.plot(snr_range_list, acc_list, marker='o', color='b', linestyle='-')
    plt.title('Accuracy vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy')
    plt.grid(True)
    plt.xticks(snr_range_list)
    # plt.savefig(f'result//{modelname}.png')  # 保存图像
    plt.show()

    # === 保存数据到 CSV ===
    with open(f'result//enc4.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR (dB)', 'Accuracy'])
        for snr, acc in zip(snr_range_list, acc_list):
            writer.writerow([f"{snr:.1f}", acc])


if __name__ == '__main__':
    main()