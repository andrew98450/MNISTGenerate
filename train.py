import argparse
import torch
import cv2
from model import NetG, NetD
from utils import load_dataset

parse = argparse.ArgumentParser()
parse.add_argument("--dataset", default="Dataset", type=str)
parse.add_argument("--download", default=True, type=bool)
parse.add_argument("--epoch", default=20, type=int)
parse.add_argument("--gpu", default=False, type=bool)
parse.add_argument("--batchsize", default=600, type=int)
parse.add_argument("--lr", default=0.0002, type=float)
args = parse.parse_args()

batch_size = args.batchsize
lr = args.lr
use_gpu = args.gpu
epoch = args.epoch
n_c = 10
c = 0.01

train_set = load_dataset(root=args.dataset, download=args.download)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, 
    batch_size=batch_size,
    shuffle=True)

netG = NetG()
netD = NetD()

if use_gpu:
    netG = netG.cuda()
    netD = netD.cuda()

g_optim = torch.optim.Adam(params=netG.parameters(), lr=lr, betas=(0.5, 0.9))
d_optim = torch.optim.Adam(params=netD.parameters(), lr=lr, betas=(0.5, 0.9))
loss_f = torch.nn.BCELoss()

for i in range(epoch):
    for iters, (real_data, labels) in enumerate(train_loader, 0):
       
        real_data = real_data.float()
        fake_data = torch.randn(batch_size, 128).float()
        if use_gpu:
            real_data = real_data.cuda()
            fake_data = fake_data.cuda()  
                   
        generate_data = netG(fake_data).detach()
        d_real = netD(real_data)
        d_fake = netD(generate_data)            
        d_real_loss = loss_f(d_real, torch.ones_like(d_real))
        d_fake_loss = loss_f(d_fake, torch.zeros_like(d_fake))
        
        d_optim.zero_grad()
        d_real_loss.backward()
        d_fake_loss.backward()
        d_optim.step()
    
        fake_data = torch.randn(batch_size, 128).float()
        if use_gpu:   
            fake_data = fake_data.cuda()             
        generate_data = netG(fake_data)
        d_fake = netD(generate_data)
        g_loss = loss_f(d_fake, torch.ones_like(d_fake))
        
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()
    
        img = generate_data[0].cpu().squeeze().mul(255).clamp(0, 255)
        img = img.byte().numpy()
        cv2.imshow("Generate Image", img)
        cv2.waitKey(1)
        
        if iters % 10 == 0:
            print("[+] Epoch: [%d/%d] G_Loss: %.4f D_Real_Loss: %.4f D_Fake_Loss: %.4f" % (i+1, epoch, g_loss, d_real_loss, d_fake_loss))

netG = netG.cpu().eval()
netD = netD.cpu().eval()

torch.save(netG, "mnistG_model.pth")  
torch.save(netD, "mnistD_model.pth")       