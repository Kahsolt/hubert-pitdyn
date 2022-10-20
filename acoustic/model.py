import torch
import torch.nn as nn

# 相比原始的 Soft-VC acoustic model，仅修改了Encoder输入宽度 (256->768)


class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(upsample=True)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x, mels)

    @torch.inference_mode()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder.generate(x)


class Encoder(nn.Module):
    def __init__(self, upsample: bool = True):
        super().__init__()
        self.prenet = PreNet(768, 384, 384)         # 768 = units[256] + pit[256] + dyn[256]
        self.convs = nn.Sequential(
            nn.Conv1d(384, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.ConvTranspose1d(512, 512, 4, 2, 1) if upsample else nn.Identity(),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prenet = PreNet(128, 256, 256)
        self.lstm1 = nn.LSTM(512 + 256, 768, batch_first=True)
        self.lstm2 = nn.LSTM(768, 768, batch_first=True)
        self.lstm3 = nn.LSTM(768, 768, batch_first=True)
        self.proj = nn.Linear(768, 128, bias=False)

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        mels = self.prenet(mels)
        x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
        res = x
        x, _ = self.lstm2(x)
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        x = res + x
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, xs: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(xs.size(0), 128, device=xs.device)
        h1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c3 = torch.zeros(1, xs.size(0), 768, device=xs.device)

        mel = []
        for x in torch.unbind(xs, dim=1):
            m = self.prenet(m)
            x = torch.cat((x, m), dim=1).unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)


class PreNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
