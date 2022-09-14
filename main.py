import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
from random import shuffle
import time
import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from dataLoader import CelebA64_Dataset
import cv2
import torchvision.models as models
from models.DCGAN import Generator, Discriminator
from torchvision.utils import save_image
import random
from pytorch_fid import fid_score
from inception_score_pytorch.inception_score import inception_score

full_weight_name = 'CelebA_64'

# Normalize weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(opt):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs('weights', exist_ok=True)
    os.makedirs('images_train', exist_ok=True)
    writer = SummaryWriter('./runs/{}'.format(full_weight_name))

    adversarial_loss = torch.nn.BCELoss().cuda()

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    dataloader_train = DataLoader(dataset=CelebA64_Dataset(opt), batch_size=opt.batch_size, num_workers=11, drop_last=True, shuffle=True)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.lr,
                                   betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=opt.lr,
                                   betas=(opt.b1, opt.b2))
    Tensor = torch.cuda.FloatTensor

    # Load pretrained weights
    if os.path.isfile(opt.weights_path):
        print("=> loading checkpoint '{}'".format(opt.weights_path))
        checkpoint = torch.load(opt.weights_path)
        start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['state_dict_G'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        discriminator.load_state_dict(checkpoint['state_dict_D'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.weights_path, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(opt.weights_path))

    # Start training
    for epoch_index in range(start_epoch, opt.n_epochs):

        print('epoch_index=', epoch_index)
        start = time.time()
        # in each minibatch
        pbar = tqdm(dataloader_train, desc='training')
        # optimizer_D.param_groups[0]['lr'] = opt.lr * 5
        # optimizer_G.param_groups[0]['lr'] = opt.lr 
        for batchIdx, imgs in enumerate(pbar):
            iterNum = epoch_index * len(pbar) + batchIdx

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0),
            # valid = Variable(Tensor(imgs.shape[0], 1).fill_(random.random()*0.3 + 0.7),
                             requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0),
            # fake = Variable(Tensor(imgs.shape[0], 1).fill_(random.random()*0.3),
                            requires_grad=False)

            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 1,
                                        (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)
            outputs = discriminator(gen_imgs)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(outputs, valid)
            # g_loss = - outputs.mean()

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            z = Variable(
                Tensor(np.random.normal(0, 1,
                                        (imgs.shape[0], opt.latent_dim))))
            fake_images = generator(z)

            outputs_fake = discriminator(fake_images)
            outputs_real = discriminator(real_imgs)                

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(outputs_real, valid)
            fake_loss = adversarial_loss(outputs_fake, fake)

            d_loss = (real_loss + fake_loss) / 2

            # real_loss = torch.nn.ReLU()(1.0 - outputs_real).mean()
            # fake_loss = torch.nn.ReLU()(1.0 + outputs_fake).mean()

            # d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # if d_loss < 1:
            #     optimizer_G.param_groups[0]['lr'] = 10 * opt.lr
            #     optimizer_D.param_groups[0]['lr'] = 0
            # else:
            #     optimizer_G.param_groups[0]['lr'] = opt.lr
            #     optimizer_D.param_groups[0]['lr'] = opt.lr

            writer.add_scalar("Loss_G/train", g_loss, iterNum)
            writer.add_scalar("Loss_D/train", d_loss, iterNum)
            # pbar.set_description("[D loss: {}] [G loss: {}]".format(d_loss.item(), g_loss.item()))   
            pbar.set_description("[D_r loss: {}] [D_f loss: {}] [G loss: {}]".format(real_loss.item(), fake_loss.item(), g_loss.item()))   

            if iterNum % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25],
                           "images_train/%d.png" % iterNum,
                           nrow=5,
                           normalize=True)
        endl = time.time()
        print('Costing time:', (endl-start)/60)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        save_info = {
            'epoch': epoch_index + 1,
            'state_dict_G': generator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'state_dict_D': discriminator.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }
        if epoch_index % 5 == 4:
            weight_name = '{}_{}.pkl'.format(opt.weights_path.split('.')[0], epoch_index + 1    )
            torch.save(save_info, weight_name)
        torch.save(save_info, opt.weights_path)

def inference(opt):
    os.makedirs("images_inference", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
    
    generator.eval()

    best_fid = 1000
    best_model = 210

    # DCGAN_soft -> 850
    # SNGAN -> 810
    # SNGAN_complex -> 910 
    for idx in range(910, 920, 10):
    # for idx in range(1):
        np.random.seed(82)

        checkpoint = torch.load(opt.weights_path.replace('.', '_{}.'.format(idx)))
        generator.load_state_dict(checkpoint['state_dict_G'])

        # Inference
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        max_batch_num = 100


        for r in range(opt.inference_num // max_batch_num):
            z = Variable(
                Tensor(np.random.normal(0, 1, (max_batch_num, opt.latent_dim))))
            gen_imgs = generator(z)

            if r == 0:
                save_image(gen_imgs.data[:32],
                           "image4report.png",
                           nrow=8,
                           normalize=True)

            for i in range(0, max_batch_num):
                save_image(gen_imgs.data[i],
                           "images_inference/%d.png" % (i + r * max_batch_num),
                           normalize=True)

        if opt.inference_num % max_batch_num != 0:
            z = Variable(
                Tensor(np.random.normal(0, 1, (max_batch_num, opt.latent_dim))))
            gen_imgs = generator(z)
            for i in range(0, opt.inference_num % max_batch_num):
                save_image(gen_imgs.data[i],
                           "images_inference/%d.png" % (i + opt.inference_num // max_batch_num * max_batch_num),
                           normalize=True)

        fid = fid_score.calculate_fid_given_paths(paths=['../hw2_data/face/test/', 'images_inference/'],batch_size=100,device=torch.device(0),dims=2048,num_workers=12)
        is_score = inception_score(FaceDataset('../GAN/images_inference/'), cuda=True, batch_size=32, resize=True, splits=1)
        
        if fid < best_fid:
            best_fid = fid
            best_model = idx
        print(idx, fid, is_score[0])
    print(best_model,best_fid)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=['train', 'inference', 'valid'],
                        default='train',
                        help="operation mode")
    parser.add_argument("--weights_path",
                        type=str,
                        default='weights/{}.pkl'.format(full_weight_name),
                        help="model path for inference")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=1000,
                        help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="size of the batches")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002,
                        # default=0.00005,
                        help="adam: learning rate")
    parser.add_argument("--b1",
                        type=float,
                        default=0.5,
                        # default=0.0,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2",
                        type=float,
                        default=0.999,
                        # default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim",
                        type=int,
                        default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--sample_interval",
                        type=int,
                        default=400,
                        help="interval between image sampling")
    parser.add_argument("--inference_num",
                        type=int,
                        default=1000,
                        help="number of generated images for inference")
    parser.add_argument("--img_size",
                        type=int,
                        default=64,
                        help="number of generated images for inference")

    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'valid':
        validate(opt)
    else:
        inference(opt)

