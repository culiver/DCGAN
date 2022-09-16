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
from model.networks import Generator, Discriminator
from torchvision.utils import save_image
import random
from pytorch_fid import fid_score
from inception_score_pytorch.inception_score import inception_score
import yaml

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
    weight_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'weights')
    sample_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'samples')
    run_dir    = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'runs')

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(run_dir   , exist_ok=True)

    writer = SummaryWriter(run_dir)

    if config['MODEL_CONFIG']['TYPE'] == 'lsgan':
        adversarial_loss = torch.nn.MSELoss().cuda()
    else:
        adversarial_loss = torch.nn.BCELoss().cuda()

    generator = Generator(config).cuda()
    discriminator = Discriminator(config).cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    dataloader_train = DataLoader(dataset=CelebA64_Dataset(config), batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'], num_workers=11, drop_last=True, shuffle=True)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['TRAINING_CONFIG']['LR'], betas=(config['TRAINING_CONFIG']['BETA1'], config['TRAINING_CONFIG']['BETA2']))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['TRAINING_CONFIG']['LR'], betas=(config['TRAINING_CONFIG']['BETA1'], config['TRAINING_CONFIG']['BETA2']))
    Tensor = torch.cuda.FloatTensor

    weights_path = os.path.join(weight_dir, '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))

    # Load pretrained weights
    if os.path.isfile(weights_path):
        print("=> loading checkpoint '{}'".format(weights_path))
        checkpoint = torch.load(weights_path)
        start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['state_dict_G'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        discriminator.load_state_dict(checkpoint['state_dict_D'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(weights_path, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(weights_path))

    # Start training
    for epoch_index in range(start_epoch, config['TRAINING_CONFIG']['EPOCH']):

        print('epoch_index=', epoch_index)
        start = time.time()
        # in each minibatch
        pbar = tqdm(dataloader_train, desc='training')
        for batchIdx, imgs in enumerate(pbar):
            iterNum = epoch_index * len(pbar) + batchIdx

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0),
                             requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0),
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
                                        (imgs.shape[0], config['MODEL_CONFIG']['LATENT_DIM']))))

            # Generate a batch of images
            gen_imgs = generator(z)
            outputs = discriminator(gen_imgs)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(outputs, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            z = Variable(
                Tensor(np.random.normal(0, 1,
                                        (imgs.shape[0], config['MODEL_CONFIG']['LATENT_DIM']))))
            fake_images = generator(z)

            outputs_fake = discriminator(fake_images)
            outputs_real = discriminator(real_imgs)                

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(outputs_real, valid)
            fake_loss = adversarial_loss(outputs_fake, fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()


            writer.add_scalar("Loss_G/train", g_loss, iterNum)
            writer.add_scalar("Loss_D/train", d_loss, iterNum)
            pbar.set_description("[D_r loss: {}] [D_f loss: {}] [G loss: {}]".format(real_loss.item(), fake_loss.item(), g_loss.item()))   

            if iterNum % config['TRAINING_CONFIG']['SAMPLE_STEP'] == 0:
                save_image(gen_imgs.data[:25],
                           os.path.join(sample_dir,"{}.jpg".format(iterNum)),
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
            weight_name = '{}_{}.pkl'.format(weights_path.split('.')[0], epoch_index + 1    )
            torch.save(save_info, weight_name)
        torch.save(save_info, weights_path)

def val(config):
    record_file = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'FID_score.txt')
    f = open(record_file, 'a')
    
    weight_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'weights')
    weight_path = os.path.join(weight_dir, '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))

    val_folder = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'val')
    GT_folder = os.path.join(val_folder, 'GT')
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(GT_folder, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = Generator(config).cuda()
    discriminator = Discriminator(config).cuda()

    if cuda:
        generator.cuda()
        discriminator.cuda()
    
    generator.eval()
    
    best_fid = np.inf
    best_model = 0
    
    for e in range(config['VAL_CONFIG']['START_EPOCH'], config['VAL_CONFIG']['END_EPOCH'], config['VAL_CONFIG']['EPOCH_STEP']):
        weight_name = '{}_{}.pkl'.format(weight_path.split('.')[0], e)
        np.random.seed(82)

        checkpoint = torch.load(weight_name, map_location='cpu')
        generator.load_state_dict(checkpoint['state_dict_G'])
        epoch_num = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        
        fid_pred_folder = os.path.join(val_folder, '{}'.format(e))
        os.makedirs(fid_pred_folder, exist_ok=True)

        # Inference
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        max_batch_num = 100

        num_left = config['VAL_CONFIG']['GEN_NUM']

        while num_left > 0:
            temp_batch_size = min(max_batch_num, num_left)
            z = Variable(
                Tensor(np.random.normal(0, 1, (temp_batch_size, config['MODEL_CONFIG']['LATENT_DIM']))))
            gen_imgs = generator(z)

            for i in range(0, temp_batch_size):
                save_image(gen_imgs.data[i],
                            "{}/{}.jpg".format(fid_pred_folder, (i + config['VAL_CONFIG']['GEN_NUM']-num_left)),
                            normalize=True)
            num_left += -temp_batch_size

        fid = fid_score.calculate_fid_given_paths(paths=['data/CelebA/Img/img_align_celeba_64x64_val', fid_pred_folder],batch_size=100,device=torch.device(0),dims=2048,num_workers=12)
        # is_score = inception_score(FaceDataset('../GAN/images_inference/'), cuda=True, batch_size=32, resize=True, splits=1)
        
        if fid < best_fid:
            best_fid = fid
            best_model = e
        print(e, fid)
        f.write('epoch:{} fid:{}\n'.format(e, fid))

    print(best_model,best_fid)
    f.write('Best epoch:{} Best fid:{}\n'.format(best_model, best_fid))

def test(config):
    weight_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'weights')
    weight_path = os.path.join(weight_dir, '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))
    output_dir = config['TEST_CONFIG']['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = Generator(config).cuda()
    discriminator = Discriminator(config).cuda()
    
    generator.eval()
    
    weight_name = '{}_{}.pkl'.format(weight_path.split('.')[0], config['TEST_CONFIG']['EPOCH'])
    np.random.seed(82)

    checkpoint = torch.load(weight_name, map_location='cpu')
    generator.load_state_dict(checkpoint['state_dict_G'])
    epoch_num = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
    
    # Inference
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    max_batch_num = 100

    num_left = config['TEST_CONFIG']['GEN_NUM']

    while num_left > 0:
        temp_batch_size = min(max_batch_num, num_left)
        z = Variable(
            Tensor(np.random.normal(0, 1, (temp_batch_size, config['MODEL_CONFIG']['LATENT_DIM']))))
        gen_imgs = generator(z)

        for i in range(0, temp_batch_size):
            save_image(gen_imgs.data[i],
                        "{}/{}.jpg".format(output_dir, (i + config['TEST_CONFIG']['GEN_NUM']-num_left)),
                        normalize=True)
        num_left += -temp_batch_size



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default= "train")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config",
                        type=str,
                        default='configs/config_lsgan.yaml')
    opt = parser.parse_args()

    config = yaml.load(open(opt.config, 'r'), Loader=yaml.FullLoader)
    config['MODE'] = opt.mode
    if opt.output_dir:
        config['TEST_CONFIG']['OUTPUT_DIR'] = opt.output_dir

    if opt.mode == 'train':
        train(config)
    elif opt.mode == 'val':
        val(config)
    elif opt.mode == 'test':
        test(config)
    else:
        print("Unsupport mode!")

