# Old Photo Restoration (Modified Fork)

<img src='imgs/0001.jpg'/>

This is a modified fork of the original [Microsoft's Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) repository, with additional features and improvements.

## Key Differences from Original Repository

1. **Web Interface**: Added a Flask-based web application for easy access through a browser
2. **Colorization**: Integrated automatic colorization of restored grayscale photos
3. **High-Resolution Support**: Enhanced support for processing high-resolution images
4. **User-Friendly Interface**: Added both GUI and web-based interfaces for easier usage
5. **Streamlined Pipeline**: Simplified the restoration process with a single command

## Installation

Follow the same installation steps as the original repository:

```bash
# Clone Synchronized-BatchNorm-PyTorch
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../

cd Global/detection_models
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../

# Download landmark detection model
cd Face_Detection/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ../

# Download pretrained models
cd Face_Enhancement/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
unzip face_checkpoints.zip
cd ../
cd Global/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
unzip global_checkpoints.zip
cd ../

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### 1. Command Line Interface

You can use the command line interface with the following options:

```bash
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              [--with_scratch] \
              [--HR] \
              [--skip_colorization]
```

Options:
- `--input_folder`: Path to folder containing input images
- `--output_folder`: Path to save output images
- `--GPU`: GPU device ID (use -1 for CPU)
- `--with_scratch`: Enable scratch removal
- `--HR`: Enable high-resolution processing
- `--skip_colorization`: Disable automatic colorization

### 2. Web Interface (New!)

The web interface provides an easy-to-use browser-based interface for photo restoration:

```bash
python run_flask.py
```

Then open your browser and navigate to `http://localhost:5000`

Features:
- Upload single or multiple images
- Toggle scratch removal
- Toggle colorization
- Download restored images
- Preview results before downloading

### 3. GUI Interface

The GUI interface provides a desktop application experience:

```bash
python GUI.py
```

Features:
- Browse and select images
- Modify photos with one click
- View results in the application window
- Save restored images automatically

### 4. Docker Container (New!)

A Docker container is provided for easy deployment and usage without installing dependencies.

#### Building the Docker image:

```bash
docker build -t old-photo-restoration .
```

#### Running the Docker container:

1. Web Interface (default):
```bash
docker run -p 5000:5000 -v "$(pwd)/test_images":/app/test_images/old -v "$(pwd)/output":/app/output --name photo-restoration-web old-photo-restoration
```

2. Command Line Interface:
```bash
docker run -v "$(pwd)/test_images":/app/test_images/old -v "$(pwd)/output":/app/output --name photo-restoration-cli old-photo-restoration cli --input_folder /app/test_images/old --output_folder /app/output --GPU -1
```

For Windows PowerShell, replace `$(pwd)` with `${PWD}` or use absolute paths:
```bash
docker run -p 5000:5000 -v "C:\path\to\images":/app/test_images/old -v "C:\path\to\output":/app/output --name photo-restoration-web old-photo-restoration
```

#### Container Management:

Start an existing container:
```bash
docker start photo-restoration-web
```

Stop a running container:
```bash
docker stop photo-restoration-web
```

View container logs:
```bash
docker logs photo-restoration-web
```

Access web interface: Open your browser and go to http://localhost:5000

#### CPU vs GPU:

This Dockerfile is configured for CPU usage by default. For GPU support, you would need to:
1. Use an NVIDIA base image in the Dockerfile (nvidia/cuda)
2. Install the NVIDIA Container Toolkit
3. Add the `--gpus all` flag when running the container

## Additional Features

### Colorization

The fork includes an automatic colorization feature that adds realistic colors to restored grayscale photos. This is enabled by default but can be disabled using the `--skip_colorization` flag or through the web interface.

### High-Resolution Support

Enhanced support for processing high-resolution images with the `--HR` flag or through the web interface's high-resolution option.

## Original Features

All original features from the Microsoft repository are preserved:
- Face enhancement
- Global restoration
- Scratch detection
- Quality restoration

## Requirements

- Python >= 3.6
- CUDA-capable GPU (recommended)
- See requirements.txt for Python dependencies

## License

This project is licensed under the same terms as the original Microsoft repository.

### [Project Page](http://raywzy.com/Old_Photo/) | [Paper (CVPR version)](https://arxiv.org/abs/2004.09484) | [Paper (Journal version)](https://arxiv.org/pdf/2009.07047v1.pdf) | [Pretrained Model](https://hkustconnect-my.sharepoint.com/:f:/g/personal/bzhangai_connect_ust_hk/Em0KnYOeSSxFtp4g_dhWdf0BdeT3tY12jIYJ6qvSf300cA?e=nXkJH2) | [Colab Demo](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing)  | [Replicate Demo & Docker Image](https://replicate.ai/zhangmozhe/bringing-old-photos-back-to-life) :fire:

**Bringing Old Photos Back to Life, CVPR2020 (Oral)**

**Old Photo Restoration via Deep Latent Space Translation, TPAMI 2022**

[Ziyu Wan](http://raywzy.com/)<sup>1</sup>,
[Bo Zhang](https://www.microsoft.com/en-us/research/people/zhanbo/)<sup>2</sup>,
[Dongdong Chen](http://www.dongdongchen.bid/)<sup>3</sup>,
[Pan Zhang](https://panzhang0212.github.io/)<sup>4</sup>,
[Dong Chen](https://www.microsoft.com/en-us/research/people/doch/)<sup>2</sup>,
[Jing Liao](https://liaojing.github.io/html/)<sup>1</sup>,
[Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/)<sup>2</sup> <br>
<sup>1</sup>City University of Hong Kong, <sup>2</sup>Microsoft Research Asia, <sup>3</sup>Microsoft Cloud AI, <sup>4</sup>USTC

<!-- ## Notes of this project
The code originates from our research project and the aim is to demonstrate the research idea, so we have not optimized it from a product perspective. And we will spend time to address some common issues, such as out of memory issue, limited resolution, but will not involve too much in engineering problems, such as speedup of the inference, fastapi deployment and so on. **We welcome volunteers to contribute to this project to make it more usable for practical application.** -->

## :sparkles: News
**2022.3.31**: Our new work regarding old film restoration will be published in CVPR 2022. For more details, please refer to the [project website](http://raywzy.com/Old_Film/) and [github repo](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life).

The framework now supports the restoration of high-resolution input.

<img src='imgs/HR_result.png'>

Training code is available and welcome to have a try and learn the training details. 

You can now play with our [Colab](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing) and try it on your photos. 

## Requirement
The code is tested on Ubuntu with Nvidia GPUs and CUDA installed. Python>=3.6 is required to run the code.

## Installation

Clone the Synchronized-BatchNorm-PyTorch repository for

```
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../
```

```
cd Global/detection_models
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

Download the landmark detection pretrained model

```
cd Face_Detection/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ../
```

Download the pretrained model, put the file `Face_Enhancement/checkpoints.zip` under `./Face_Enhancement`, and put the file `Global/checkpoints.zip` under `./Global`. Then unzip them respectively.

```
cd Face_Enhancement/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
unzip face_checkpoints.zip
cd ../
cd Global/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
unzip global_checkpoints.zip
cd ../
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## :rocket: How to use?

**Note**: GPU can be set 0 or 0,1,2 or 0,2; use -1 for CPU

### 1) Full Pipeline

You could easily restore the old photos with one simple command after installation and downloading the pretrained model.

For images without scratches:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0
```

For scratched images:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              --with_scratch
```

**For high-resolution images with scratches**:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              --with_scratch \
              --HR
```

**For colorization of restored images**:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              --with_scratch \
              --HR
```

By default, the pipeline includes a final colorization step to add realistic colors to the restored grayscale photos. If you want to disable colorization, add the `--skip_colorization` flag:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              --skip_colorization
```

Note: Please try to use the absolute path. The final results will be saved in `./output_path/final_output/`. You could also check the produced results of different steps in `output_path`.

### 2) Scratch Detection

Currently we don't plan to release the scratched old photos dataset with labels directly. If you want to get the paired data, you could use our pretrained model to test the collected images to obtain the labels.

```
cd Global/
python detection.py --test_path [test_image_folder_path] \
                    --output_dir [output_path] \
                    --input_size [resize_256|full_size|scale_256]
```

<img src='imgs/scratch_detection.png'>

### 3) Global Restoration

A triplet domain translation network is proposed to solve both structured degradation and unstructured degradation of old photos.

<p align="center">
<img src='imgs/pipeline.PNG' width="50%" height="50%"/>
</p>

```
cd Global/
python test.py --Scratch_and_Quality_restore \
               --test_input [test_image_folder_path] \
               --test_mask [corresponding mask] \
               --outputs_dir [output_path]

python test.py --Quality_restore \
               --test_input [test_image_folder_path] \
               --outputs_dir [output_path]
```

<img src='imgs/global.png'>


### 4) Face Enhancement

We use a progressive generator to refine the face regions of old photos. More details could be found in our journal submission and `./Face_Enhancement` folder.

<p align="center">
<img src='imgs/face_pipeline.jpg' width="60%" height="60%"/>
</p>


<img src='imgs/face.png'>

> *NOTE*: 
> This repo is mainly for research purpose and we have not yet optimized the running performance. 
> 
> Since the model is pretrained with 256*256 images, the model may not work ideally for arbitrary resolution.

### 5) Colorization

This implementation adds a colorization step that uses a pre-trained deep learning model to colorize the restored grayscale photos. The colorization is based on OpenCV's DNN module with a model trained on the ImageNet dataset.

<p align="center">
<img src='imgs/colorization_sample.jpg' width="80%" height="80%"/>
</p>

The colorization process runs automatically after the restoration pipeline is completed. The colorized images are saved in the `colorized_output` directory within your specified output folder.

You can enable or disable colorization:

- In the command line interface, use the `--skip_colorization` flag to disable it
- In the web interface, toggle the "Colorize Photo" checkbox

#### Key Features:

- Automatically colorizes the restored grayscale photos
- Uses a deep neural network trained on a large dataset of natural images
- Produces realistic and natural-looking colors
- Integrates seamlessly with the existing restoration pipeline

The model files required for colorization are already included in the `Global/models` directory.

### 6) GUI

A user-friendly GUI which takes input of image by user and shows result in respective window.

#### How it works:

1. Run GUI.py file.
2. Click browse and select your image from test_images/old_w_scratch folder to remove scratches.
3. Click Modify Photo button.
4. Wait for a while and see results on GUI window.
5. Exit window by clicking Exit Window and get your result image in output folder.

<img src='imgs/gui.PNG'>

## How to train?

### 1) Create Training File

Put the folders of VOC dataset, collected old photos (e.g., Real_L_old and Real_RGB_old) into one shared folder. Then
```
cd Global/data/
python Create_Bigfile.py
```
Note: Remember to modify the code based on your own environment.

### 2) Train the VAEs of domain A and domain B respectively

```
cd ..
python train_domain_A.py --use_v2_degradation --continue_train --training_dataset domain_A --name domainA_SR_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 100 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder]

python train_domain_B.py --continue_train --training_dataset domain_B --name domainB_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder]  --no_instance --resize_or_crop crop_only --batchSize 120 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [your_output_folder]  --checkpoints_dir [your_ckpt_folder]
```
Note: For the --name option, please ensure your experiment name contains "domainA" or "domainB", which will be used to select different dataset.

### 3) Train the mapping network between domains

Train the mapping without scratches:
```
python train_mapping.py --use_v2_degradation --training_dataset mapping --use_vae_which_epoch 200 --continue_train --name mapping_quality --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 80 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder]
```


Traing the mapping with scraches:
```
python train_mapping.py --no_TTUR --NL_res --random_hole --use_SN --correlation_renormalize --training_dataset mapping --NL_use_mask --NL_fusion_method combine --non_local Setting_42 --use_v2_degradation --use_vae_which_epoch 200 --continue_train --name mapping_scratch --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 36 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder] --irregular_mask [absolute_path_of_mask_file]
```

Traing the mapping with scraches (Multi-Scale Patch Attention for HR input):
```
python train_mapping.py --no_TTUR --NL_res --random_hole --use_SN --correlation_renormalize --training_dataset mapping --NL_use_mask --NL_fusion_method combine --non_local Setting_42 --use_v2_degradation --use_vae_which_epoch 200 --continue_train --name mapping_Patch_Attention --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 36 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder] --irregular_mask [absolute_path_of_mask_file] --mapping_exp 1
```


## Citation

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@inproceedings{wan2020bringing,
title={Bringing Old Photos Back to Life},
author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={2747--2757},
year={2020}
}
```

```bibtex
@article{wan2020old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={arXiv preprint arXiv:2009.07047},
  year={2020}
}
```

If you are also interested in the legacy photo/video colorization, please refer to [this work](https://github.com/zhangmozhe/video-colorization).

## Maintenance

This project is currently maintained by Ziyu Wan and is for academic research use only. If you have any questions, feel free to contact raywzy@gmail.com.

## License

The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
