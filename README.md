# SwinIR
experiment for reproducing SwinIR result (NYCU VLlab)

# SwinIR: Image Restoration Using Shifted Window Transformer
[paper](https://arxiv.org/abs/2108.10257)
**|** 
[supplementary](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)
**|** 
[visual results](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)
**|** 
[original project page](https://github.com/JingyunLiang/SwinIR)
**|**
[online Colab demo](https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb)

> Image restoration is a long-standing low-level vision problem that aims to restore high-quality images from low-quality images (e.g., downscaled, noisy and compressed images). While state-of-the-art image restoration methods are based on convolutional neural networks, few attempts have been made with Transformers which show impressive performance on high-level vision tasks. In this paper, we propose a strong baseline model SwinIR for image restoration based on the Swin Transformer. SwinIR consists of three parts: shallow feature extraction, deep feature extraction and high-quality image reconstruction. In particular, the deep feature extraction module is composed of several residual Swin Transformer blocks (RSTB), each of which has several Swin Transformer layers together with a residual connection. We conduct experiments on three representative tasks: image super-resolution (including classical, lightweight and real-world image super-resolution), image denoising (including grayscale and color image denoising) and JPEG compression artifact reduction. Experimental results demonstrate that SwinIR outperforms state-of-the-art methods on different tasks by up to 0.14~0.45dB, while the total number of parameters can be reduced by up to 67%.


### Dataset Preparation

Training and testing sets can be downloaded as follows. Please put them in `trainsets` and `testsets` respectively.

| Task                 | Training Set | Testing Set|       
| :---                 | :---:        |     :---:      |
| classical/lightweight image SR          | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) or DIV2K +[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | set5 + Set14 + BSD100 + Urban100 + Manga109 [download all](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) |
| real-world image SR          | SwinIR-M (middle size): [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) +[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [OST](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip) (10324 images, sky,water,grass,mountain,building,plant,animal) <br /> SwinIR-L (large size): DIV2K + Flickr2K + OST + [WED](ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) + [FFHQ](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) (first 2000 images, face) + Manga109 (manga) + [SCUT-CTW1500](https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip) (first 100 training images, texts) <br /><br />  ***We use the first practical degradation model [BSRGAN, ICCV2021  ![GitHub Stars](https://img.shields.io/github/stars/cszn/BSRGAN?style=social)](https://github.com/cszn/BSRGAN) for real-world image SR** | [RealSRSet+5images](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/RealSRSet+5images.zip) | 
| color/grayscale image denoising      | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) |  grayscale: Set12 + BSD68 + Urban100 <br />  color: CBSD68 + Kodak24 + McMaster + Urban100 [download all](https://github.com/cszn/FFDNet/tree/master/testsets) | 
| JPEG compression artifact reduction  | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) |  grayscale: Classic5 +LIVE1 [download all](https://github.com/cszn/DnCNN/tree/master/testsets) |

### Environment
python   : 3.8.11   
pytorch  : 1.9.0 + cu102


### Training
To train SwinIR, run the following commands. You may need to modified the related .json file:  
(EX: classical SR, using `options/swinir/train_swinir_sr_classical.json` ),    
`dataroot_H`            : path for training set, high resolution image(groud truth),  
`dataroot_L`            : path for training set, low resolution image,  
`scale factor`          : setting scale for training (SR: 2,3,4,...),   
`dataloader_batch_size` : set the training batch size,    
and also,  `noisel level`, `JPEG level`, `G_optimizer_lr`, `G_scheduler_milestones`, etc. in the json file could be modified for different experiment scnario.  

??????  
(0921)  
To do the GPU device selection, please use `CUDA_VISIBLE_DEVICES=0,3,....`in`train.sh `, or directly write it into command line like:  
 `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json  --dist True `  
`gpu_ids` in `options/swinir/train_swinir_sr_classical.json` seems doesn't work....still finding the reason.    
??????  

And, modified the args below(you may directly modified it in `main_train_psnr.py`, or write it in the command )    
`--opt`           : path to related .json file,    
`--scale`         : setting scale for testing (SR: 2,3,4,...),    
`--folder_lq`     : path for testing set, low resolution image,  
`--folder_gt`     : path for testing set, high resolution image(groud truth),    
`--model_save_dir`: path for saving model(if model performance boost, do the model saving)  
`--chart_save_dir`: path for saving chart   

checkpoint setting:   
(`checkpoint_test`, `checkpoint_save`, `checkpoint_print` in `options/swinir/train_swinir_sr_classical.json`)  
In original setting, the model would:  
print the training message for every 200 iterations,  
saving model for every 5000 iterations,  
testing with set5 in training process every 5000 iterations,  
Noted that: one iteration means one parameter update  


use `sh train.sh` for classical SR x2 training (distributed training)   
or the command below for training:     

```python
# 001 Classical Image SR (middle size)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json  --dist True

# 002 Lightweight Image SR (small size)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_lightweight.json  --dist True

# 003 Real-World Image SR (middle size)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_realworld_psnr.json  --dist True
# before training gan, put the PSNR-oriented model into superresolution/swinir_sr_realworld_x4_gan/models/
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_realworld_gan.json  --dist True

# 004 Grayscale Image Deoising (middle size)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_denoising_gray.json  --dist True

# 005 Color Image Deoising (middle size)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_denoising_color.json  --dist True

# 006 JPEG Compression Artifact Reduction (middle size)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_car_jpeg.json  --dist True
```

You can also train above models using `DataParallel` as follows, but it will be slower.
```python
# 001 Classical Image SR (middle size)
python main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json

...
```


Note:

1, We fine-tune X3/X4/X8 (or noise=25/50, or JPEG=10/20/30) models from the X2 (or noise=15, or JPEG=40) model, so that total_iteration can be halved to save training time. In this case, we halve the initial learning rate and lr_milestones accordingly. This way has similar performance as training from scratch.

2, For SR, we use different kinds of `Upsampler` in classical/lightweight/real-world image SR for the purpose of fair comparison with existing works.

3, We did not re-train the models after cleaning the codes. Feel free to open an issue if you meet any problems. 

4, validation(or said test in the training process) result would be saved in `SwinIR/set5test_results/swinir_classical_sr_x2/`  

## Testing
Following command will download the [pretrained models](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) and put them in `model_zoo/swinir`. All visual results of SwinIR can be downloaded [here](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0).

use `sh test.sh`  for classical SR x2 testing  
or command below for testing:  

```bash
# 001 Classical Image Super-Resolution (middle size)
# Note that --training_patch_size is just used to differentiate two different settings in Table 2 of the paper. Images are NOT tested patch by patch.
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth --folder_lq testsets/set5/LR_bicubic/X2 --folder_gt testsets/set5/HR
python main_test_swinir.py --task classical_sr --scale 3 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth --folder_lq testsets/set5/LR_bicubic/X3 --folder_gt testsets/set5/HR
python main_test_swinir.py --task classical_sr --scale 4 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --folder_lq testsets/set5/LR_bicubic/X4 --folder_gt testsets/set5/HR
python main_test_swinir.py --task classical_sr --scale 8 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth --folder_lq testsets/set5/LR_bicubic/X8 --folder_gt testsets/set5/HR

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq testsets/set5/LR_bicubic/X2 --folder_gt testsets/set5/HR
python main_test_swinir.py --task classical_sr --scale 3 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth --folder_lq testsets/set5/LR_bicubic/X3 --folder_gt testsets/set5/HR
python main_test_swinir.py --task classical_sr --scale 4 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth --folder_lq testsets/set5/LR_bicubic/X4 --folder_gt testsets/set5/HR
python main_test_swinir.py --task classical_sr --scale 8 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth --folder_lq testsets/set5/LR_bicubic/X8 --folder_gt testsets/set5/HR


# 002 Lightweight Image Super-Resolution (small size)
python main_test_swinir.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/set5/LR_bicubic/X2 --folder_gt testsets/set5/HR
python main_test_swinir.py --task lightweight_sr --scale 3 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth --folder_lq testsets/set5/LR_bicubic/X3 --folder_gt testsets/set5/HR
python main_test_swinir.py --task lightweight_sr --scale 4 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth --folder_lq testsets/set5/LR_bicubic/X4 --folder_gt testsets/set5/HR


# 003 Real-World Image Super-Resolution
# (middle size)
python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq testsets/RealSRSet+5images

# (larger size + trained on more datasets)
python main_test_swinir.py --task real_sr --scale 4 --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq testsets/RealSRSet+5images


# 004 Grayscale Image Deoising (middle size)
python main_test_swinir.py --task gray_dn --noise 15 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth --folder_gt testsets/set12
python main_test_swinir.py --task gray_dn --noise 25 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth --folder_gt testsets/set12
python main_test_swinir.py --task gray_dn --noise 50 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth --folder_gt testsets/set12


# 005 Color Image Deoising (middle size)
python main_test_swinir.py --task color_dn --noise 15 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth --folder_gt testsets/McMaster
python main_test_swinir.py --task color_dn --noise 25 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth --folder_gt testsets/McMaster
python main_test_swinir.py --task color_dn --noise 50 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth --folder_gt testsets/McMaster


# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
python main_test_swinir.py --task jpeg_car --jpeg 10 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 20 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 30 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 40 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth --folder_gt testsets/classic5
```
---

Note:
1, test(`test.sh`) result would be saved in `SwinIR/results/swinir_classical_sr_x2/`  

## Results
<details>
<summary>Classical Image Super-Resolution (click me)</summary>
<p align="center">
  <img width="900" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/classic_image_sr.png">
  <img width="900" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/classic_image_sr_visual.png">
</p>
</details>

<details>
<summary>Lightweight Image Super-Resolution</summary>
<p align="center">
  <img width="900" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/lightweight_image_sr.png">
</p>
</details>

<details>
<summary>Real-World Image Super-Resolution</summary>
<p align="center">
  <img width="900" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/real_world_image_sr.png">
</p>
</details>


|&nbsp;&nbsp;&nbsp; Real-World Image (x4)|[BSRGAN, ICCV2021](https://github.com/cszn/BSRGAN)|[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)|SwinIR (ours)|
|      :---      |     :---:        |        :-----:         |        :-----:         | 
|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_LR.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_BSRGAN.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_realESRGAN.jpg">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_SwinIR.png">
|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_LR.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_BSRGAN.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_realESRGAN.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_SwinIR.png">|

<details>
<summary>Grayscale Image Deoising</summary>
<p align="center">
  <img width="900" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/gray_image_denoising.png">
</p>
</details>

<details>
<summary>Color Image Deoising</summary>
<p align="center">
  <img width="900" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/color_image_denoising.png">
</p>
</details>

<details>
<summary>JPEG Compression Artifact Reduction</summary>
<p align="center">
  <img width="900" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/jepg_compress_artfact_reduction.png">
</p>
</details>



Please refer to the [paper](https://arxiv.org/abs/2108.10257) and the [original project page](https://github.com/JingyunLiang/SwinIR)
for more results.


## Citation
    @article{liang2021swinir,
        title={SwinIR: Image Restoration Using Swin Transformer},
        author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
        journal={arXiv preprint arXiv:2108.10257}, 
        year={2021}
    }
