# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from src.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from models.discriminator import Discriminator


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    discriminator = Discriminator(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))
    print('[DEBUG] %s Parameters in Discriminator: %d.' % (dt.now(), utils.network_utils.count_parameters(discriminator)))



    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)
    merger.apply(utils.network_utils.init_weights)
    discriminator.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(),
                                          lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)


        discriminator_solver = torch.optim.Adam(discriminator.parameters(),lr=cfg.TRAIN.DISCRIMINATOR_LR,betas=cfg.TRAIN.BETAS)


    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(refiner.parameters(),
                                         lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(merger.parameters(),
                                        lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                        momentum=cfg.TRAIN.MOMENTUM)

        discriminator_solver = torch.optim.SGD(discriminator.parameters(),
                                           lr=cfg.TRAIN.DISCRIMINATOR_LR,
                                           momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
                                                               milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                               gamma=cfg.TRAIN.GAMMA)



    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
        discriminator = torch.nn.DataParallel(discriminator).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    adversarial_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
              (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()


        # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()
        discriminator.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)


            # Train the encoder, decoder, refiner, and merger
            image_features = encoder(rendering_images)

            raw_features, generated_volumes = decoder(image_features)


            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)

            encoder_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refiner(generated_volumes)
                print(generated_volumes.shape)
                refiner_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10

                g_adv_loss = adversarial_loss(discriminator(generated_volumes), real_labels)
                total_refiner_loss = refiner_loss + cfg.TRAIN.GAN_LOSS_WEIGHT * g_adv_loss
            else:
                refiner_loss = encoder_loss

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            total_refiner_loss.zero_grad()
            merger.zero_grad()
            discriminator.zero_grad()


            print("the shape for the generated volumne: ",generated_volumes.shape) #(batch_size,32,32,32)
            real_labels = torch.ones(generated_volumes.size(0), 1).cuda()
            fake_labels = torch.zeros(generated_volumes.size(0), 1).cuda()
            # print(real_labels.shape)
            # print(fake_labels.shape)

            # Detach generated volumes so we don’t update the generator (refiner) yet

            gt_pred = discriminator(ground_truth_volumes)
            # print(gt_pred.shape)
            pred=discriminator(generated_volumes.detach())
            # print(pred.shape)

            real_loss = adversarial_loss(gt_pred.view(gt_pred.size(0), -1) , real_labels)
            fake_loss = adversarial_loss(pred.view(pred.size(0),-1), fake_labels)

            d_loss = (real_loss + fake_loss) * 0.5


            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
                # total_refiner_loss.backward()
            else:
                encoder_loss.backward()

            d_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()
            discriminator_solver.step()

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)
            train_writer.add_scalar('Discriminator/BatchLoss', d_loss.item(), n_itr)
            # train_writer.add_scalar('Refiner/AdvLoss', g_adv_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f RLoss = %.4f'
                % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                   data_time.val, encoder_loss.item(), refiner_loss.item()))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, encoder_losses.avg,
               refiner_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)),
                                                 epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver,
                                                 refiner, refiner_solver, merger, merger_solver, discriminator, discriminator_solver, best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-ckpt.pth'), epoch_idx + 1, encoder,
                                                 encoder_solver, decoder, decoder_solver, refiner, refiner_solver,
                                                 merger, merger_solver,discriminator,discriminator_solver ,best_iou, best_epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
