# -*- coding: utf-8 -*-
import os
import random
from matplotlib import pyplot as plt
from torchmetrics.audio import SignalNoiseRatio
from dataset.dataset_GNP_WGAN import MyDataset
from config_GNP_WGAN import args
from models.GNP_WGAN_GP import GNP_WGAN_GP,PatchGNP_WGAN_Discriminator
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import torch.nn as nn
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS

def val(val_loader, model,epoch):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    total_images = 0
    val_snr = 0.0
    with torch.no_grad():
        val_loop = tqdm(val_loader, total=len(val_loader), desc=f"Epoch [{epoch + 1}/{args.num_epochs}]",leave=True)
        for input, target in val_loop:
            input, target = input.float(), target.float()
            input, target = input.cuda(), target.cuda()
            output = model(input)
            mse_loss = nn.MSELoss()
            mse = mse_loss(output, target)
            val_loss += mse.item()
            SNR = SignalNoiseRatio().cuda()
            snr = SNR(output, target)
            val_snr += snr.item()
            output = output.cpu().numpy()
            target = target.cpu().numpy()
            psnr = peak_signal_noise_ratio(output, target, data_range=1)
            val_psnr += psnr
            for i in range(len(output)):
                output_i = output[i].squeeze(0)
                target_i = target[i].squeeze(0)
                ssim_i = structural_similarity(output_i, target_i, data_range=1)
                val_ssim += ssim_i
                total_images += 1
            val_loop.set_postfix(MSE=mse.item() ,PSNR=psnr,SNR=snr.item(),SSIM=ssim_i)
    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    val_snr /= len(val_loader)
    val_ssim /= total_images
    return val_loss, val_psnr, val_ssim,val_snr

def apply_colormap(image, cmap_name='seismic'):
    cmap = plt.get_cmap(cmap_name)
    mapped_image = cmap(image)
    return mapped_image[..., :3]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def main():
    setup_seed(22)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    feature_path = "data/data_C3/features/"
    structure_path = "data/data_C3/gnp/"
    seismic_dataset = MyDataset(feature_path,structure_path)
    train_size = int(0.8 * len(seismic_dataset))
    test_size = len(seismic_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(seismic_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True
                                               )
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               drop_last=True
                                             )
    print('train_data size:', len(train_dataset))
    print('train_loader:', len(train_loader))
    print('val_dataset size:', len(val_dataset))
    print('val_loader:', len(val_loader))

    generator = GNP_WGAN_GP().cuda()
    discriminator = PatchGNP_WGAN_Discriminator().cuda()
    generator = generator.apply(weights_init)
    discriminator = discriminator.apply(weights_init)
    # epoch
    num_epochs = args.num_epochs
    # Optimizers
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=args.lr_g,betas=(0.5, 0.9))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d,betas=(0.5, 0.9))
    writer = SummaryWriter(log_dir=args.log_file)

    loss_fn1 = MS_SSIM_L1_LOSS()
    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype=torch.float32, device='cuda')
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones_like(d_interpolates, device='cuda')
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    n_critic = 5
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch + 1}/{args.num_epochs}]",leave=True)
        for i,(feature, label) in enumerate(loop):
            feature, label = feature.float(), label.float()
            feature, label = feature.cuda(), label.cuda()
            # === Train Discriminator ===leave=True
            optimizer_d.zero_grad()
            # Generate fake images
            gen_images = generator(feature)
            # Get discriminator outputs
            real_outputs = discriminator(label)
            fake_outputs = discriminator(gen_images.detach())
            # GP
            gradient_penalty = compute_gradient_penalty(discriminator, label, gen_images.detach())
            # loss
            d_loss = (torch.mean(fake_outputs) - torch.mean(real_outputs)) + gradient_penalty * 10
            d_loss.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()
            # === Train Generator ===
            if (i + 1) % n_critic == 0:
                # Get discriminator outputs for fake images
                fake_outputs = discriminator(gen_images)
                #g_loss
                g_loss_1 = loss_fn1(gen_images, label)
                g_loss = -torch.mean(fake_outputs) + g_loss_1*10
                g_loss.backward()
                optimizer_g.step()
                # Updating tqdm progress bar
                loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())
        MSE, PSNR, SSIM, SNR = val(val_loader, generator, epoch)
        if (epoch + 1) % 5 == 0:
            colormap_output = np.array([apply_colormap(img.cpu().detach().numpy()[0]) for img in gen_images])
            colormap_label = np.array([apply_colormap(img.cpu().numpy()[0]) for img in label])
            colormap_output = colormap_output.astype(np.float32)
            colormap_label = colormap_label.astype(np.float32)
            colormap_output = np.transpose(colormap_output, (0, 3, 1, 2))
            colormap_label = np.transpose(colormap_label, (0, 3, 1, 2))
            writer.add_images('val Images', colormap_output, epoch)
            writer.add_images('label Images', colormap_label, epoch)
            # Save models at the end of each epoch
            torch.save(generator.state_dict(), os.path.join(args.save_dir, f'generator_epoch_{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.save_dir, f'discriminator_epoch_{epoch + 1}.pth'))
        # Log the average losses for the current epoch to TensorBoard
        writer.add_scalar('Loss/Discriminator', d_loss, epoch)
        writer.add_scalar('Loss/Generator', g_loss, epoch)
        writer.add_scalar('Indicator/MSE', MSE, epoch)
        writer.add_scalar('Indicator/SNR', SNR, epoch)
        writer.add_scalar('Indicator/PSNR', PSNR, epoch)
        writer.add_scalar('Indicator/SSIM', SSIM, epoch)
    writer.close()


if __name__ == '__main__':
    main()