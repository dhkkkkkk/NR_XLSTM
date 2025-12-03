import torch
from tqdm import tqdm
import numpy as np
from my_model.dataset_ESC50 import load_test_data
import my_model.s_i_tft as ft
import pandas
import random
import matplotlib.pyplot as plt
import csv
from create_model import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torchinfo import summary
from ptflops import get_model_complexity_info
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
    batch_size = 8
    sample_rate = 8000
    crop_len = 1600
    hop_size = 32
    win_size = 400
    #name = ["my_model","my_model_detach",my_model_woCom,"MobileNetv3","EfficientNet_v2","xLSTM","LSTM","RNN","GRU","ConvTransPlus",
    # Resnet18,Resnet34,TCN,TSSequencer]
    modelname = "my_model"
    model = load_model(modelname)
    model.eval()
    snr = 60.0

    print(f"\n=== 正在测试 SNR = {snr} dB ===")

    # 构建测试数据集
    test_ds = load_test_data(clean_dir, noise_dir, (snr, snr), sample_rate, crop_len, batch_size)
    test_loader = tqdm(test_ds, desc=f'Evaluating {snr}dB', unit='batch')

    tatal_acc = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clean, _, labels in test_loader:
            clean = clean.to(DEVICE)
            # noisy = noisy.to(DEVICE)
            labels = labels.to(DEVICE)

            # STFT
            n_mag, n_pha, _, _ = ft.torch_mag_pha_stft(clean, hop_size=hop_size, win_size=win_size)
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

            correct_labels = (est_labels == labels).sum().item()
            out_len = labels.size(0)
            acc = correct_labels / out_len
            tatal_acc += acc

            # 收集预测和标签
            all_preds.extend(est_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_acc = tatal_acc / len(test_ds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)


    print(f"[SNR={snr} dB] 平均准确率 : {avg_acc:.4f}")
    print(f"[SNR={snr} dB] 精确率(Precision): {precision:.4f}")
    print(f"[SNR={snr} dB] 召回率(Recall): {recall:.4f}")
    print(f"[SNR={snr} dB] F1 分数(F1 Score): {f1:.4f}")

    # df = pandas.DataFrame({
    #     "label": all_labels,
    #     "pred": all_preds
    # })
    #
    # save_path = f"pred_label_snr_{snr}dB.csv"
    # df.to_csv(save_path, index=False, encoding="utf-8")
    # # === 混淆矩阵 ===
    # cm = confusion_matrix(all_labels, all_preds)
    # print(f"[SNR={snr} dB] 混淆矩阵：")
    # print(cm)
    #
    # plt.figure(figsize=(6, 5))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap='Blues', values_format='d')
    # plt.title(f"Confusion Matrix (SNR={snr} dB)")
    # plt.show()


    # 统计模型 FLOPs 和参数量
    with torch.cuda.device(0):  # 确保使用 CUDA
        macs, params = get_model_complexity_info(
            model, (2, 51, 201),  # [C, T, F]
            as_strings=True, print_per_layer_stat=False, verbose=False
        )

    print(f"{modelname} 模型统计:")
    print(f"参数量（Parameters）: {params}")
    print(f"FLOPs（MACs）: {macs}")


if __name__ == '__main__':
    main()