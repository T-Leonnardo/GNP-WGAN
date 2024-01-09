# -*- coding: utf-8 -*-
import argparse
parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--lr_g', default=2e-4, type=float, help='Learning rate')
parser.add_argument('--lr_d', default=2e-4, type=float, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=10, help='batch size of trainset')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of iterations')
parser.add_argument('--log_file', default='log_file_path', help='fixed loss root path')
parser.add_argument('--save_dir', default='model_file_path', help='fixed PSNR root path')
args = parser.parse_args()
