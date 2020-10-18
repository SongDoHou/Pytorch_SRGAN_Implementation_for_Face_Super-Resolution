import torch
import torch.nn as nn
import torch.nn.functional as F

# input:   3 x 32 x 32
# output:  3 x 128 x 128

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return x + self.main(x)

class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        residual_blocks = []
        for _ in range(5):
            residual_blocks.append(ResidualBlock())
        self.residual_blocks = nn.Sequential(*residual_blocks)
        # self.block1 = ResidualBlock()
        # self.block2 = ResidualBlock()
        # self.block3 = ResidualBlock()
        # self.block4 = ResidualBlock()
        # self.block5 = ResidualBlock()

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample1 = UpsampleBlock()
        self.upsample2 = UpsampleBlock()

        self.last = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.first(x)
        _skip = x

        x = self.residual_blocks(x)
        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        x = self.mid(x)
        x = x + _skip

        x = self.upsample1(x)
        x = self.upsample2(x)

        x = self.last(x)

        return x

class DBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(DBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, xh=128, xw=128):
        super(Discriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.block2 = DBlock(64, 64, 2)
        self.block3 = DBlock(64, 128, 1)
        self.block4 = DBlock(128, 128, 2)
        self.block5 = DBlock(128, 256, 1)
        self.block6 = DBlock(256, 256, 2)
        self.block7 = DBlock(256, 512, 1)
        self.block8 = DBlock(512, 512, 2)

        self.block9 = nn.Sequential(
            nn.Linear(512 * (xh//16) * (xw//16), 1204),
            nn.LeakyReLU(0.2),
            nn.Linear(1204, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x.view(x.size()[0], -1))

        return x

if __name__ == "__main__":
    generator = Generator().cuda()
    toy_input = torch.randn((1, 3, 32, 32)).cuda()
    generator_result = generator(toy_input)
    print(generator_result.size()) # 1 x 3 x 128 x 128

    discriminator = Discriminator().cuda()
    discriminator_result = discriminator(generator_result)
    print(discriminator_result.size()) # 1 x 1
