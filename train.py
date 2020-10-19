from model import Generator
from model import Discriminator
import torch
import torch.optim as optim
from dataset import FaceData, TrainData
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.models import vgg19


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def main():
    batch_size = 16
    generator = Generator().cuda()
    discriminator = Discriminator(96,96).cuda()
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    # dataset = FaceData('train')
    dataset = TrainData()
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    MSE = nn.MSELoss()
    BCE = nn.BCELoss()

    # content loss, perceptual loss vgg / i,j == 5,4
    vgg_net = vgg19(pretrained=True).features[:36].cuda()
    vgg_net.eval()
    for param in vgg_net.parameters():
        param.requires_grad = False

    discriminator.train()
    generator.train()
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    print("Start Training")
    current_epoch = 0
    for epoch in range(current_epoch, 100):
        for step, (img_Input, img_GT) in tqdm(enumerate(data_loader)):

            img_GT = img_GT.cuda()
            img_Input = img_Input.cuda()
            
            # # Discriminator update
            # img_SR = generator(img_Input)
            # fake = discriminator(img_SR)
            # real = discriminator(img_GT)
            # loss_Dfake = 0.001 * BCE(fake, torch.zeros(batch_size, 1).cuda())
            # loss_Dreal = 0.001 * BCE(real, torch.ones(batch_size, 1).cuda())
            # loss_D = 0.001 * (loss_Dfake + loss_Dreal)
            # # if epoch > 0:
            # discriminator.zero_grad()
            # loss_D.backward(retain_graph=True)
            # optimizer_D.step()

            # # Generator update
            # img_SR = generator(img_Input)
            # loss_content = MSE(img_SR, img_GT)
            # loss_vgg = 0.006 * MSE(vgg_net(img_SR), vgg_net(img_GT))
            # fake = discriminator(img_SR)
            # loss_Dfake = 0.001 * BCE(fake, torch.zeros(batch_size, 1).cuda())

            # loss_G = loss_content + loss_vgg + loss_Dfake
            # generator.zero_grad()
            # loss_G.backward()
            # # loss_Dfake.backward()
            # optimizer_G.step()
            if epoch < 10:
                # SRResnet Initialize Generator update
                generator.zero_grad()
                img_SR = generator(img_Input)
                loss_content = MSE(img_SR, img_GT)
                loss_content.backward()
                optimizer_G.step()
                
                if step%100 == 0:
                    print()
                    print("Loss_content : {}".format(loss_content.item()))
                continue

            # Discriminator update
            discriminator.zero_grad()
            D_real = discriminator(img_GT)
            loss_Dreal = 0.1 * BCE(D_real, torch.ones(batch_size, 1).cuda())
            loss_Dreal.backward()
            D_x = D_real.mean().item()

            img_SR = generator(img_Input)
            D_fake = discriminator(img_SR.detach())
            loss_Dfake = 0.1 * BCE(D_fake, torch.zeros(batch_size, 1).cuda())
            loss_Dfake.backward()
            DG_z = D_fake.mean().item()

            loss_D = (loss_Dfake + loss_Dreal)
            optimizer_D.step()

            # Generator update
            generator.zero_grad()
            loss_content = MSE(img_SR, img_GT)
            loss_vgg = MSE(vgg_net(img_SR), vgg_net(img_GT))

            # img_SR = generator(img_Input)
            G_fake = discriminator(img_SR)
            loss_Gfake = BCE(G_fake, torch.zeros(batch_size, 1).cuda())

            loss_G = loss_content + 0.006* loss_vgg + 0.001 * loss_Gfake
            loss_G.backward()
            # loss_Dfake.backward()
            optimizer_G.step()


            if step%100 == 0:
                # :.10f
                print()
                print("fake out : {}".format(DG_z))
                print("real out : {}".format(D_x))
                print("Loss_Dfake :   {}".format(loss_Dfake.item()))
                print("Loss_Dreal :   {}".format(loss_Dreal.item()))
                print("Loss_D :       {}".format(loss_D.item()))
                print("Loss_content : {}".format(loss_content.item()))
                print("Loss_vgg :     {}".format(0.006*loss_vgg.item()))
                print("Loss_Gfake :   {}".format(0.001*loss_Gfake.item()))
                print("Loss_G :       {}".format(loss_G.item()))
                print("Loss_Total :   {}".format((loss_G + loss_D).item()))
                # print("Loss_D : {:.4f}".format(loss_D.item()))
                # print("Loss : {:.4f}".format(loss_total.item()))

        with torch.no_grad():
            generator.eval()
            save_image(denorm(img_SR[0].cpu()), "./Result/{0}_SR.png".format(epoch))
            save_image(denorm(img_GT[0].cpu()), "./Result/{0}_GT.png".format(epoch))
            generator.train()

if __name__ == "__main__":
    main()