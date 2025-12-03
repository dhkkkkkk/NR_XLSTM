import librosa
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

def torch_mag_pha_stft(y, n_fft=400, hop_size=10, win_size=128, compress_factor=0.3, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    # com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1) #压缩后的complex
    real, imag = mag*torch.cos(pha), mag*torch.sin(pha)
    return mag, pha, real, imag


def torch_mag_pha_istft(mag, pha, n_fft=400, hop_size=10, win_size=128, compress_factor=0.3, center=True):
    # Magnitude Decompression

    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)
    return wav


def main():

    n_fft = 400
    win_length = 128
    hop_length = 32
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    resample = T.Resample(48000, 8000)
    wavdir = r''
    data, sr = torchaudio.load(wavdir)

    # data = data[:,0:4000]
    data = resample(data)

    # d_data = d_data[:,0:400]
    # d_data = d_data / d_data.abs().max()
    # d_data = F.normalize(d_data,dim=1)
    mag, pha, real, imag = torch_mag_pha_stft(data, n_fft, hop_length, win_length,compress_factor=0.3)
    # mag = mag.squeeze()
    # pha = pha.squeeze()
    wav = torch_mag_pha_istft(mag, pha,compress_factor=0.3)



    mag = mag.squeeze().cpu()
    mag_np = mag.numpy()

    pha = pha.squeeze().cpu()
    pha_np = pha.numpy()

    print(mag.size())
    plt.figure(figsize=(10, 8))
    plt.imshow(mag_np, aspect='auto', origin='lower', cmap='viridis',
               )
    # plt.colorbar(label='Magnitude (dB)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('STFT Magnitude Spectrum')
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(r"C:\Users\11540\Desktop\fig/mag.jpg", bbox_inches='tight', pad_inches=0)
    plt.show()




    # plt.show()


if __name__ == '__main__':
    main()