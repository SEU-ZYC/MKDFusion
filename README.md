# MKDFusion
Codes for ***MKDFusion： Modality knowledge decoupled for infrared and visible image fusion.***

## Usage

We have now released the v1 version of MKDFusion to the public, in which we replaced the VMamba architecture of the CMF module in the original paper with the Transformer architecture. It is worth looking forward to once our paper is officially accepted, we will immediately upload and update the final version of MKDFusion to ensure that all users can access the most advanced and comprehensive fusion model.

### Network Architecture

Our MKDFusionv1 is implemented in ``MKDFusionv1.py``.

### Training
**1. Virtual Environment**
```
# create virtual environment
conda create -n cddfuse python=3.8.10
conda activate MKDFusion
# select pytorch version yourself
# install cddfuse requirements
pip install -r requirements.txt
```
**2. Install VMamba**

Install VMamba from [this link](https://github.com/MzeroMiko/VMamba)

**3. Data Preparation**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder ``'./MSRS_train/'``.

**4. Pre-Processing**

Run 
```
python dataprocessing.py
``` 
and the processed training dataset is in ``'./data/MSRS_train_imgsize_128_stride_200.h5'``.

**5. MKDFusion Training**

Run 
```
python train_MKDFusionv1.py
``` 
and the trained model is available in ``'./models/'``.

### Testing

**1. Pretrained models**

Download the Pretrained models from [this link](https://pan.baidu.com/s/1eVIraSKv6Kk9xsFDSoNbYw?pwd=vevy) and place it in the folder ``'./models/'``.

**2. Test images**

We have selected several demo test images and stored them in``'./test_img/MSRS'``, ``'./test_img/TNO'``.

**3. Test**

If you want to infer with our  and obtain the fusion results in our paper, please run
```
python test_MKDFusionv1.py
``` 

