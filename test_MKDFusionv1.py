from MKDFusionv1 import BaseFeatureExtraction, DetailFeatureExtraction, VMamba_Encoder, VMamba_Decoder, multihead, multihead2
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/MKDFusionv1.pth"
for dataset_name in ["MSRS"]:
    print("\n"*2+"="*80)
    model_name="MKDFusionv1.pth"
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('test_result',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    VMamba_Encoder = nn.DataParallel(VMamba_Encoder()).to(device)
    VMamba_Decoder = nn.DataParallel(VMamba_Decoder()).to(device)
    BaseFeatureExtraction = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    multihead = nn.DataParallel(multihead()).to(device)
    multihead2 = nn.DataParallel(multihead2()).to(device)

    VMamba_Encoder.load_state_dict(torch.load(ckpt_path)['VMamba_Encoder'])
    VMamba_Decoder.load_state_dict(torch.load(ckpt_path)['VMamba_Decoder'])
    BaseFeatureExtraction.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    multihead.load_state_dict(torch.load(ckpt_path)['multihead'])
    multihead2.load_state_dict(torch.load(ckpt_path)['multihead2'])

    VMamba_Encoder.eval()
    VMamba_Decoder.eval()
    BaseFeatureExtraction.eval()
    DetailFuseLayer.eval()
    multihead.eval()
    multihead2.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            # ycrcb, uint8
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = VMamba_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = VMamba_Encoder(data_IR)
            feature_VI_B = multihead(feature_V_B,feature_I_B)
            feature_VI_D = multihead(feature_V_D, feature_I_D)
            feature_F_B = BaseFeatureExtraction(feature_VI_B)
            feature_F_D = DetailFuseLayer(feature_VI_D)
            data_Fuse, feature_F = VMamba_Decoder(data_VIS, feature_F_B, feature_F_D)

            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())

            # float32 to uint8
            fi = fi.astype(np.uint8)
            # concatnate
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
