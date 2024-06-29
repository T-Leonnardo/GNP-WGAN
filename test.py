import os
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from torch import nn
from models.GNP_WGAN_GP import GNP_WGAN_GP
from ssim_map import cal_ssim
from torchmetrics.audio import SignalNoiseRatio


def visualize_data(input_image, output_image, target_image,cmap="seismic",interpolation='bilinear'):

    residual = (output_image-target_image)
    _,ssim_map = cal_ssim(np.squeeze(output_image, axis=0),np.squeeze(target_image, axis=0))
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(input_image.transpose(1, 2, 0), cmap=cmap,interpolation=interpolation,vmin=0,vmax=1)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(output_image.transpose(1, 2, 0), cmap=cmap,interpolation=interpolation,vmin=0,vmax=1)
    axes[1].set_title('Output')
    axes[1].axis('off')

    axes[2].imshow(target_image.transpose(1, 2, 0), cmap=cmap,interpolation=interpolation,vmin=0,vmax=1)
    axes[2].set_title('Target')
    axes[2].axis('off')

    axes[3].imshow(residual.transpose(1, 2, 0), cmap=cmap,interpolation=interpolation,vmin=0,vmax=1)
    axes[3].set_title('Residual')
    axes[3].axis('off')

    axes[4].imshow(ssim_map, cmap=cmap)
    axes[4].set_title('SSIM Map')
    axes[4].axis('off')

    plt.show()

def test_GNP_GAN(model, val_loader, criterion):
    model.eval()
    mse = 0.0
    psnr = 0.0
    ssim_val = 0.0
    total_images = 0
    val_snr = 0.0
    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.float(), target.float()
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)
            mse += loss.item()

            SNR = SignalNoiseRatio().cuda()
            snr = SNR(output, target)
            val_snr += snr.item()

            output = output.cpu().numpy()
            target = target.cpu().numpy()
            psnr += peak_signal_noise_ratio(output, target, data_range=1.0)
            for i in range(len(output)):
                output_i = output[i].squeeze(0)
                target_i = target[i].squeeze(0)
                ssim_i = structural_similarity(output_i, target_i, data_range=1.0)
                ssim_val += ssim_i
                total_images +=1

        index_to_visualize = 0
        input = input.cpu().numpy()[index_to_visualize]
        output_image = output[index_to_visualize]
        target_image = target[index_to_visualize]

        visualize_data(input, output_image, target_image,)

        mse /= len(val_loader)
        psnr /= len(val_loader)
        ssim_val /= total_images
        val_snr /= len(val_loader)

    return mse, psnr, ssim_val,val_snr

def test_GMAR(model, test_loader, criterion, device):
    model.eval()
    mse = 0.0
    psnr = 0.0
    ssim = 0.0
    save_path = "data/data_C3/structure_weak"
    feature_counter = 1
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.float(), target.float()
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            mse += loss.item()
            output = output.cpu().numpy()
            target = target.cpu().numpy()
            psnr += peak_signal_noise_ratio(output, target, data_range=1.0)
            save_name = f"structure{feature_counter}.npy"
            output_np = output.squeeze(axis=0).squeeze(axis=0)
            np.save(os.path.join(save_path, save_name), output_np)
            feature_counter += 1
            for i in range(len(output)):
                output_i = output[i]
                target_i = target[i]
                ssim_i = structural_similarity(output_i, target_i, data_range=1.0, channel_axis=0)
                ssim += ssim_i
        mse /= len(test_loader)
        psnr /= len(test_loader)
        ssim /= (len(test_loader) * len(output))
    return mse, psnr, ssim

def test_PconvUnet(model, test_loader, criterion):
    model.eval()
    mse = 0.0
    psnr = 0.0
    ssim = 0.0
    val_snr = 0.0
    total_images = 0
    with torch.no_grad():
        for i, (input, mask, target) in enumerate(test_loader):
            input, mask, target = input.float(), mask.float(),target.float()
            input, mask, target = input.cuda(), mask.cuda(), target.cuda()

            output,_ = model(input,mask)
            output = mask * input + (1 - mask) * output
            loss = criterion(output, target)
            mse += loss.item()

            SNR = SignalNoiseRatio().cuda()
            snr = SNR(output, target)
            val_snr += snr.item()

            output = output.cpu().numpy()
            target = target.cpu().numpy()
            psnr += peak_signal_noise_ratio(output, target, data_range=1.0)

            for i in range(len(output)):
                output_i = output[i]
                target_i = target[i]
                ssim_i = structural_similarity(output_i, target_i, data_range=1.0, channel_axis=0)
                ssim += ssim_i
                total_images += 1
        index_to_visualize = 0

        input_image = input.cpu().numpy()[index_to_visualize]
        output_image = output[index_to_visualize]
        target_image = target[index_to_visualize]

        visualize_data(input_image, output_image, target_image)

        mse /= len(test_loader)
        psnr /= len(test_loader)
        ssim /= total_images
        val_snr /= len(test_loader)

    return mse, psnr, ssim, val_snr

if __name__ == '__main__':

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU")

    model = GNP_WGAN_GP().cuda()
    model.load_state_dict(torch.load('ckpts/WGAN_generator.pth'))
    from dataset.dataset_GNP_WGAN import MyDataset
    feature_path = "data/data_c3/features/"
    structure_path = "data/data_c3/gnp/"
    seismic_dataset = MyDataset(feature_path,structure_path)
    train_size = int(0.8 * len(seismic_dataset))
    test_size = len(seismic_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(seismic_dataset, [train_size, test_size])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             shuffle=False
                                             )
    print('val_dataset size:', len(val_dataset))
    print('val_loader:', len(val_loader))
    criterion = nn.MSELoss()
    mse, psnr_epoch ,ssim_epoch,snr=test_GNP_GAN(model, val_loader, criterion)
    print(mse)
    print(psnr_epoch)
    print(snr)
    print(ssim_epoch)