# SwinIR
experiment for reproducing SwinIR result



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

## Testing
Following command will download the [pretrained models](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) and put them in `model_zoo/swinir`. All visual results of SwinIR can be downloaded [here](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0).

If you are too lazy to prepare the datasets, please follow the guide in the [original project page](https://github.com/JingyunLiang/SwinIR#testing-without-preparing-datasets), where you can start testing in a minute. We also provide an [online Colab demo for real-world image SR  <a href="https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb) for comparison with [the first practical degradation model BSRGAN (ICCV2021)  ![GitHub Stars](https://img.shields.io/github/stars/cszn/BSRGAN?style=social)](https://github.com/cszn/BSRGAN) and a recent model [RealESRGAN](https://github.com/xinntao/Real-ESRGAN). Try to test your own images on Colab!

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
