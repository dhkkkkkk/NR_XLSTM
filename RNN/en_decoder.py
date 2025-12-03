import torch
import torch.nn as nn
from my_model.utils import LearnableSigmoid2d



class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, baseline):
        super(EncoderLayer, self).__init__()

        if baseline == 'RNN':
            self.RNN = nn.RNN(hidden_size,2*hidden_size,num_layers=4,batch_first=True)
            self.linear_proj = nn.Linear(2*hidden_size, hidden_size)
        elif baseline == 'LSTM':
            self.RNN = nn.LSTM(hidden_size,2*hidden_size,num_layers=4,batch_first=True)
            self.linear_proj = nn.Linear(2*hidden_size, hidden_size)
        elif baseline == 'GRU':
            self.RNN = nn.GRU(hidden_size,2*hidden_size,num_layers=4,batch_first=True)
            self.linear_proj = nn.Linear(2*hidden_size, hidden_size)

        self.preNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.secNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)

    def forward(self, x):
        x = self.preNorm(x)
        y,_ = self.RNN(x)
        y = self.linear_proj(y)
        y = self.dropout(y)
        x = x + y

        y = self.secNorm(x)
        y = self.ffn(y)
        y = self.dropout(y)
        x = x + y
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, baseline):
        super(DecoderLayer, self).__init__()

        if baseline == 'RNN':
            self.RNN = nn.RNN(hidden_size, 2*hidden_size, num_layers=4, batch_first=True)
            self.linear_proj = nn.Linear(2*hidden_size, hidden_size)

        elif baseline == 'LSTM':
            self.RNN = nn.LSTM(hidden_size, 2*hidden_size, num_layers=4, batch_first=True)
            self.linear_proj = nn.Linear(2*hidden_size, hidden_size)

        elif baseline == 'GRU':
            self.RNN = nn.GRU(hidden_size, 2*hidden_size, num_layers=4, batch_first=True)
            self.linear_proj = nn.Linear(2*hidden_size, hidden_size)

        self.preNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.secNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)

    def forward(self, x):
        x = self.preNorm(x)
        y,_ = self.RNN(x)
        y = self.linear_proj(y)
        y = self.dropout(y)
        x = x + y

        y = self.secNorm(x)
        y = self.ffn(y)
        y = self.dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, baseline):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate, baseline)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs):

        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output)
        return self.last_norm(encoder_output)


class MaskDecoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, baseline):
        super(MaskDecoder, self).__init__()
        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate,baseline)
                    for _ in range(n_layers)]
        self.transformer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layers = nn.ModuleList(decoders)
        self.sub_pixel = SPConvTranspose2d(1, 1, (1, 3), 4)
        self.conv_1 = nn.Conv2d(1, 1, (1, 2), (1, 1), (0, 1))
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.prelu = nn.PReLU(1)
        self.final_conv = nn.Conv2d(1, 1, (1, 1))
        self.lsigmoid = LearnableSigmoid2d((4*hidden_size) + 1, beta=2)

    def forward(self, x):
        for dec_layer in self.layers:
            x = dec_layer(x)    #(b,t,f)
        x = self.transformer_norm(x).unsqueeze(1)    # (b,c,t,f)
        x = self.sub_pixel(x)   # (b,c,t,r*f')
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)  # (b,f,t)
        return self.lsigmoid(x)


class ComplexDecoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, baseline):
        super(ComplexDecoder, self).__init__()
        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate, baseline)
                    for _ in range(n_layers)]
        self.transformer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layers = nn.ModuleList(decoders)
        self.sub_pixel = SPConvTranspose2d(1, 1, (1, 3), 4)
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.prelu = nn.PReLU(1)
        self.complex_conv = nn.Conv2d(1, 2, (1, 2),(1, 1), (0, 1))

    def forward(self, x):
        for dec_layer in self.layers:
            x = dec_layer(x)
        x = self.transformer_norm(x).unsqueeze(1)     # (b,c,t,f)
        x = self.sub_pixel(x)   # (b,c,t,r*f')
        x = self.prelu(self.norm(x))
        x = self.complex_conv(x)
        return x.permute(0, 1, 3, 2)


class Classifier(nn.Module):
    def __init__(self,hidden_size,num_classes,baseline):
        super(Classifier, self).__init__()
        if baseline == 'RNN':
            self.RNN = nn.RNN(hidden_size, 2*hidden_size, num_layers=4, batch_first=True)
            self.linear_proj = nn.Linear(2 * hidden_size, hidden_size)

        elif baseline == 'LSTM':
            self.RNN = nn.LSTM(hidden_size, 2*hidden_size, num_layers=4, batch_first=True)
            self.linear_proj = nn.Linear(2 * hidden_size, hidden_size)

        elif baseline == 'GRU':
            self.RNN = nn.GRU(hidden_size, 2*hidden_size, num_layers=4, batch_first=True)
            self.linear_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,num_classes)
        )

    def forward(self,x):
        x,_ = self.RNN(x)
        x = self.linear_proj(x)
        class_token = x[:,-1,:]
        y = self.fc(class_token)
        return y




def main():
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.rand(4, 2, 51, 201).to(device=DEVICE)



if __name__ == '__main__':
    main()
