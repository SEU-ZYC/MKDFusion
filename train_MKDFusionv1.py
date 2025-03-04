# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from MKDFusionv1 import BaseFeatureExtraction, DetailFeatureExtraction, VMamba_Encoder, VMamba_Decoder, multihead, multihead2, classify_BASE, classify_CNN
from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss
import kornia

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'MKDFusionv1'

# . Set the hyper-parameters for training
num_epochs = 80  # total epoch
epoch_gap = 40  # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.  # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
VMamba_Encoder = nn.DataParallel(VMamba_Encoder()).to(device)
VMamba_Decoder = nn.DataParallel(VMamba_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
multihead = nn.DataParallel(multihead()).to(device)
multihead2 = nn.DataParallel(multihead2()).to(device)
classify_BASE = nn.DataParallel(classify_BASE()).to(device)
classify_CNN = nn.DataParallel(classify_CNN()).to(device)
# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    VMamba_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    VMamba_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(
    multihead.parameters(), lr=lr, weight_decay=weight_decay)
optimizer6 = torch.optim.Adam(
    multihead2.parameters(), lr=lr, weight_decay=weight_decay)
optimizer7 = torch.optim.Adam(
    classify_BASE.parameters(), lr=lr, weight_decay=weight_decay)
optimizer8 = torch.optim.Adam(
    classify_CNN.parameters(), lr=lr, weight_decay=weight_decay)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=optim_step, gamma=optim_gamma)
scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer7, step_size=optim_step, gamma=optim_gamma)
scheduler8 = torch.optim.lr_scheduler.StepLR(optimizer8, step_size=optim_step, gamma=optim_gamma)
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
bceloss = nn.BCELoss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')
# Loss_ssim = kornia.losses.SSIMLoss(5)

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    loss_all=0
    i=0
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        VMamba_Encoder.train()
        VMamba_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()
        multihead.train()
        multihead2.train()
        classify_BASE.train()
        classify_CNN.train()
        VMamba_Encoder.zero_grad()
        VMamba_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        multihead.zero_grad()
        multihead2.zero_grad()
        classify_BASE.zero_grad()
        classify_CNN.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()
        optimizer7.zero_grad()
        optimizer8.zero_grad()

        if epoch < epoch_gap:  # Phase I
            feature_V_B, feature_V_D, _ = VMamba_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = VMamba_Encoder(data_IR)
            data_VIS_hat, _ = VMamba_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = VMamba_Decoder(data_IR, feature_I_B, feature_I_D)

            class_vb = classify_BASE(feature_V_B)
            class_ib = classify_BASE(feature_I_B)
            class_vd = classify_CNN(feature_V_D)
            class_id = classify_CNN(feature_I_D)

            class_loss = bceloss(class_vb,torch.ones_like(class_vb)) + bceloss(class_ib,torch.zeros_like(class_ib)) + bceloss(class_vd,torch.ones_like(class_vd)) + bceloss(class_id,torch.zeros_like(class_id))

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I  + coeff_tv * Gradient_loss + class_loss*0.05
            loss.backward()
            nn.utils.clip_grad_norm_(
                VMamba_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                VMamba_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                classify_CNN.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                classify_BASE.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer7.step()
            optimizer8.step()
        else:  # Phase II
            feature_V_B, feature_V_D, feature_V = VMamba_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = VMamba_Encoder(data_IR)
            feature_VI_B = multihead(feature_V_B,feature_I_B)
            feature_VI_D = multihead(feature_V_D, feature_I_D)
            feature_F_B = BaseFuseLayer(feature_VI_B)
            feature_F_D = DetailFuseLayer(feature_VI_D)
            data_Fuse, feature_F = VMamba_Decoder(data_VIS, feature_F_B, feature_F_D)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss
            loss.backward()
            nn.utils.clip_grad_norm_(
                VMamba_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                VMamba_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                multihead.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                multihead2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            optimizer6.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate
    scheduler1.step()
    scheduler2.step()
    scheduler7.step()
    scheduler8.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()
        scheduler6.step()
    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
    if optimizer6.param_groups[0]['lr'] <= 1e-6:
        optimizer6.param_groups[0]['lr'] = 1e-6
    if optimizer7.param_groups[0]['lr'] <= 1e-6:
        optimizer7.param_groups[0]['lr'] = 1e-6
    if optimizer8.param_groups[0]['lr'] <= 1e-6:
        optimizer8.param_groups[0]['lr'] = 1e-6
if True:
    checkpoint = {
        'VMamba_Encoder': VMamba_Encoder.state_dict(),
        'VMamba_Decoder': VMamba_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
        'multihead': multihead.state_dict(),
        'multihead2': multihead2.state_dict(),
        'classify_BASE': classify_BASE.state_dict(),
        'classify_CNN': classify_CNN.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/MKDFusionv1_" + timestamp + '.pth'))



