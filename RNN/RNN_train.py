import torch
import my_model.s_i_tft as ft
from RNN_model import my_rnn
import torch.nn.functional as F
import argparse
import my_model.dataset_ESC50 as dataloader
from tqdm import tqdm
from my_model.SI_SNR import SISNR_loss, SISNR_score, n_SISNR_loss
import os
from my_model.utils import phase_losses
import torch.nn as nn
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, train_ds, test_ds, args, device):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.args = args
        self.device = device
        self.generator = my_rnn(layer= self.args.layer,
                                       in_channels=2,
                                       in_length=(self.args.crop_len // self.args.hop_size) + 1,
                                       feature_dim=(self.args.n_fft // 2) + 1,
                                       baseline= self.args.baseline
                                       ).to(device=device)
        if self.args.pre_flag == True:
            self.generator.load_state_dict(torch.load(self.args.pre_path))
        self.optimizer_gene = torch.optim.Adam([{'params': self.generator.cov_head.parameters(), 'lr': self.args.init_lr},
                                                 {'params': self.generator.Encoder.parameters(), 'lr': self.args.init_lr},
                                                 {'params': self.generator.TSCB.parameters(), 'lr': self.args.init_lr},
                                                 {'params': self.generator.Mask_Decoder.parameters(), 'lr': self.args.init_lr},
                                                 {'params': self.generator.Com_Decoder.parameters(), 'lr': self.args.init_lr},
                                                 # {'params': self.generator.Classifier.parameters(), 'lr': self.args.init_lr }
                                                 ])
        self.optimizer_cls = torch.optim.Adam(self.generator.Classifier.parameters(),lr=self.args.init_lr)

        self.class_loss = nn.CrossEntropyLoss()

    def forward_generator_step(self, clean, noisy):
        n_mag, n_pha, n_real, n_imag = ft.torch_mag_pha_stft(noisy, self.args.n_fft, self.args.hop_size,
                                                             self.args.win_length,self.args.compress_factor)    # 4*(b,f,t)
        c_mag, c_pha, c_real, c_imag = ft.torch_mag_pha_stft(clean, self.args.n_fft, self.args.hop_size,
                                                             self.args.win_length,self.args.compress_factor)    # 4*(b,f,t)
        n_input = torch.stack((n_mag.permute(0,2,1),n_pha.permute(0,2,1)),1)    # (b,c,t,f)
        est_real, est_imag, class_outputs = self.generator(n_input) #(b,f,t)

        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        est_pha = torch.atan2(est_imag+(1e-10), est_real+(1e-5))
        est_audio = ft.torch_mag_pha_istft(est_mag, est_pha, self.args.n_fft, self.args.hop_size,
                                                             self.args.win_length,self.args.compress_factor)

        return {
            "clean_real": c_real,
            "clean_imag": c_imag,
            "clean_mag": c_mag,
            "clean_pha": c_pha,

            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "est_pha": est_pha,

            "est_audio": est_audio,
            "class_outputs": class_outputs
        }

    def calculate_generator_loss(self, generator_outputs):

        sisnr_loss = n_SISNR_loss(generator_outputs["clean"], generator_outputs["est_audio"])

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )
        loss_ip, loss_gd, loss_iaf = phase_losses(generator_outputs["clean_pha"],generator_outputs["est_pha"])
        pha_loss = loss_ip + loss_gd + loss_iaf
        loss = torch.stack([
            self.args.loss_weights[0] * loss_ri,
            self.args.loss_weights[1] * loss_mag,
            self.args.loss_weights[2] * time_loss,
            self.args.loss_weights[3] * sisnr_loss,
        #     self.args.loss_weights[4] * score_loss,
            self.args.loss_weights[5] * pha_loss
        ])
        return loss



    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        labels = batch[2].to(self.device)


        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["clean"] = clean


        loss = (self.calculate_generator_loss(generator_outputs)).sum()
        classloss = self.class_loss(generator_outputs["class_outputs"],labels)
        self.optimizer_gene.zero_grad()
        self.optimizer_cls.zero_grad()
        loss.backward(retain_graph=True)
        classloss.backward()
        self.optimizer_gene.step()
        self.optimizer_cls.step()

        return loss.item(), classloss.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        labels = batch[2].to(self.device)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["clean"] = clean
        score = SISNR_score(generator_outputs["clean"], generator_outputs["est_audio"])
        loss = self.calculate_generator_loss(generator_outputs)
        _, predicted = generator_outputs["class_outputs"].max(1)
        correct_test = (predicted == labels).sum().item()
        total_test = labels.size(0)
        acc = correct_test / total_test
        return score, loss, acc

    def train(self):



        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_gene, step_size=self.args.decay_epoch, gamma=0.5)
        scheduler_C = torch.optim.lr_scheduler.StepLR(
            self.optimizer_cls, step_size=self.args.decay_epoch, gamma=0.5)


        total_params = sum(p.numel() for p in list(self.generator.parameters()))
        print(f"Total parameters: {total_params}")
        for epoch in range(self.args.epochs):

            # self.args.detach_decoder = False    #统一训练
            self.args.detach_decoder = True     #全detach
            # self.args.detach_decoder = ((epoch + 1) % 5 != 0) #每3epoch训练1次decoder

            loop1 = tqdm(enumerate(self.train_ds), total=len(self.train_ds),
                         desc=f'T_Epoch {epoch + 1}/{self.args.epochs}', unit='batch')
            self.generator.train()
            step = 0
            gen_loss = 0.0
            gen_cl_loss = 0.0
            for idx, data in loop1:
                step = idx + 1
                loss, class_loss = self.train_step(data)
                gen_loss += loss
                gen_cl_loss += class_loss
                loop1.set_postfix({
                    'g_loss': '{0:1.5f}'.format(loss),
                    'cls_loss': '{0:1.5f}'.format(class_loss)
                })
            avg_loss = gen_loss / step
            acg_cl_loss = gen_cl_loss /step
            print(f'avg_loss: {avg_loss},  avg_cl_loss: {acg_cl_loss}')

            self.args.detach_decoder = False
            loop2 = tqdm(enumerate(self.test_ds),total=len(self.test_ds),
                         desc=f'V_Epoch {epoch + 1}/{self.args.epochs}', unit='batch')
            self.generator.eval()
            step = 0
            gen_score = 0.0
            gen_acc = 0.0
            gen_loss_list = torch.zeros(5,device=self.device)
            for idx, data in loop2:
                step = idx + 1
                p_si_snr_socre, loss_list, acc = self.test_step(data)
                gen_score += p_si_snr_socre
                gen_loss_list += loss_list
                gen_acc += acc
                loop2.set_postfix({'p_SI-SNR' : '{0:1.5f}'.format(p_si_snr_socre)})
            avg_score = gen_score / step
            avg_loss = gen_loss_list / step
            avg_acc = gen_acc /step
            print(f"avg_score: {avg_score}")
            for i, loss_name in enumerate(['loss_ri', 'loss_mag', 'time_loss', 'score_loss', 'pha_loss']):
                print(f'{loss_name}: {avg_loss[i]:.5f}')
            print(f"class_accuracy: {avg_acc:.5f}")
            # if avg_score > best_score:
            #     best_score = avg_score
            #     weight_path = os.path.join(self.args.save_model_dir,f"best_{int(best_score*10)}.pth")
            #     torch.save(self.generator.state_dict(),weight_path)
            #     print(f"Epoch {epoch + 1}: 保存了最佳模型, s: {best_score}")
            if (epoch + 1) % 10 == 0:
                weight_path =os.path.join(self.args.save_model_dir,f"epoch_{epoch+1}.pth")
                torch.save(self.generator.state_dict(), weight_path)
                print(f"Epoch {epoch + 1}: 保存了模型")

            scheduler_G.step()
            scheduler_C.step()



def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", type=bool, default=False, help="DEBUG MODE")
    parser.add_argument("--pre_flag", type=bool, default=False, help="...")
    parser.add_argument("--pre_path", type=str,
                        default=r"", ####
                        help="..")

    parser.add_argument("--epochs", type=int, default=60, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--detach_decoder",type = bool, default= False)

    parser.add_argument("--clean_dir_train", type=str, default="")
    parser.add_argument("--clean_dir_test", type=str, default="")
    parser.add_argument("--noise_dir_train",type=str, default="")
    parser.add_argument("--noise_dir_test", type=str,default="")
    parser.add_argument("--save_model_dir", type=str, default='', ####
                        help="dir of saved model")


    parser.add_argument("--samplerate", type=int,default=8000)
    parser.add_argument("--crop_len",type=int, default=1600)    #same with sr at 1s

    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--compress_factor",type=float, default=0.3)
    parser.add_argument("--hop_size", type=int, default=32)
    parser.add_argument("--win_length", type=int, default=400)

    parser.add_argument("--baseline",type=str,default="RNN")
    args = parser.parse_args()
    # set_seed(42)
    train_ds, test_ds = dataloader.n_load_data(args.clean_dir_train, args.clean_dir_test,
                                               args.noise_dir_train, args.noise_dir_test,
                                               (-20.0, 15.0), args.samplerate,args.crop_len,args.batch_size)
    trainer = Trainer(train_ds, test_ds, args, DEVICE)
    trainer.train()


if __name__ == "__main__":
    main()
