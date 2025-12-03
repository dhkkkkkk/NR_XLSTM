from tsai.models.TSSequencerPlus import TSSequencerPlus
import torch
import my_model.s_i_tft as ft
import argparse
import my_model.dataset_ESC50 as dataloader
from tqdm import tqdm
import os
import torch.nn as nn


class my_model(nn.Module):
    def __init__(self, model, exp_channel):
        super(my_model, self).__init__()

        self.channel_conv = nn.Conv2d(in_channels=2, out_channels=exp_channel, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(201, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.model = model

    def forward(self, x):
        x = self.channel_conv(x)
        x = torch.squeeze(x, 1)
        x = x.permute(0,2,1)
        x = self.model(x) #(b,f,t)
        out =self.fc(x)#（b,f)
        return out


class Trainer:
    def __init__(self, train_ds, test_ds, args, device):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.args = args
        self.device = device


        self.ori_model = TSSequencerPlus(201,201,51,d_model=512,depth=6,lstm_dropout=0.1,dropout=0.1,mlp_ratio=2,
                        pre_norm=True,use_token=True,fc_dropout=0.1)
        self.model = my_model(self.ori_model,1).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.init_lr)
        self.class_loss = nn.CrossEntropyLoss()

    def forward_generator_step(self, clean, noisy):
        n_mag, n_pha, n_real, n_imag = ft.torch_mag_pha_stft(noisy, self.args.n_fft, self.args.hop_size,
                                                             self.args.win_length,self.args.compress_factor)    # 4*(b,f,t)
        n_input = torch.stack((n_mag.permute(0,2,1),n_pha.permute(0,2,1)),1)    # (b,c,t,f)
        class_outputs = self.model(n_input)
        return {
            "class_outputs": class_outputs
        }


    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        labels = batch[2].to(self.device)
        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )

        classloss = self.class_loss(generator_outputs["class_outputs"],labels)
        self.optimizer.zero_grad()
        classloss.backward()
        self.optimizer.step()

        return classloss.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        labels = batch[2].to(self.device)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )

        _, predicted = generator_outputs["class_outputs"].max(1)
        correct_test = (predicted == labels).sum().item()
        total_test = labels.size(0)
        acc = correct_test / total_test
        return acc

    def train(self):



        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay_epoch, gamma=0.5)


        total_params = sum(p.numel() for p in list(self.model.parameters()))
        print(f"Total parameters: {total_params}")
        for epoch in range(self.args.epochs):


            loop1 = tqdm(enumerate(self.train_ds), total=len(self.train_ds),
                         desc=f'T_Epoch {epoch + 1}/{self.args.epochs}', unit='batch')
            self.model.train()
            step = 0
            gen_cl_loss = 0.0
            for idx, data in loop1:
                step = idx + 1
                class_loss = self.train_step(data)

                gen_cl_loss += class_loss
                loop1.set_postfix({
                    'cls_loss': '{0:1.5f}'.format(class_loss)
                })
            acg_cl_loss = gen_cl_loss /step
            print(f' avg_cl_loss: {acg_cl_loss}')


            loop2 = tqdm(enumerate(self.test_ds),total=len(self.test_ds),
                         desc=f'V_Epoch {epoch + 1}/{self.args.epochs}', unit='batch')
            self.model.eval()
            step = 0
            gen_acc = 0.0
            for idx, data in loop2:
                step = idx + 1
                acc = self.test_step(data)
                gen_acc += acc
                # loop2.set_postfix({'p_SI-SNR' : '{0:1.5f}'.format(p_si_snr_socre)})
            avg_acc = gen_acc /step
            print(f"class_accuracy: {avg_acc:.5f}")
            # if avg_score > best_score:
            #     best_score = avg_score
            #     weight_path = os.path.join(self.args.save_model_dir,f"best_{int(best_score*10)}.pth")
            #     torch.save(self.generator.state_dict(),weight_path)
            #     print(f"Epoch {epoch + 1}: 保存了最佳模型, s: {best_score}")
            if (epoch + 1) % 10 == 0:
                weight_path =os.path.join(self.args.save_model_dir,f"epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), weight_path)
                print(f"Epoch {epoch + 1}: 保存了模型")

            scheduler.step()




def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", type=bool, default=False, help="DEBUG MODE")
    parser.add_argument("--pre_flag", type=bool, default=False, help="...")
    parser.add_argument("--pre_path", type=str,
                        default=r"", ####
                        help="..")

    parser.add_argument("--epochs", type=int, default=60, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--detach_decoder",type = bool, default= False)

    parser.add_argument("--clean_dir_train", type=str, default="")
    parser.add_argument("--clean_dir_test", type=str, default="")
    parser.add_argument("--noise_dir_train",type=str, default="")
    parser.add_argument("--noise_dir_test", type=str,default="")
    parser.add_argument("--save_model_dir", type=str, default=r'', ####
                        help="dir of saved model")

    parser.add_argument("--samplerate", type=int,default=8000)
    parser.add_argument("--crop_len",type=int, default=1600)    #same with sr at 1s

    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--compress_factor",type=float, default=0.3)
    parser.add_argument("--hop_size", type=int, default=32)
    parser.add_argument("--win_length", type=int, default=400)

    args = parser.parse_args()
    # set_seed(42)
    train_ds, test_ds = dataloader.n_load_data(args.clean_dir_train, args.clean_dir_test,
                                               args.noise_dir_train, args.noise_dir_test,
                                               (-20.0, 15.0), args.samplerate,args.crop_len,args.batch_size)
    trainer = Trainer(train_ds, test_ds, args, DEVICE)
    trainer.train()


if __name__ == "__main__":
    main()