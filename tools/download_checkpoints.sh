#!/bin/bash

# Instructions for Manual Download:
#
# Please, download the [MiT weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)
# pretrained on ImageNet-1K provided by the official
# [SegFormer repository](https://github.com/NVlabs/SegFormer) and put them in a
# folder `pretrained/` within this project. Only mit_b5.pth is necessary.
#
# Please, download the checkpoint of HRDA on GTA->Cityscapes from
# [here](https://drive.google.com/file/d/1O6n1HearrXHZTHxNRWp8HCMyqbulKcSW/view?usp=sharing).
# and extract it to `work_dirs/`

# Automatic Downloads:
set -e  # exit when any command fails
mkdir -p pretrained/
cd pretrained/
gdown --id 1d7I50jVjtCddnhpf-lqj8-f13UyCzoW1  # MiT-B5 weights
cd ../

mkdir -p work_dirs/
cd work_dirs/
gdown --id 1O6n1HearrXHZTHxNRWp8HCMyqbulKcSW  # HRDA on GTA->Cityscapes
tar -xzf gtaHR2csHR_hrda_246ef.tar.gz
rm gtaHR2csHR_hrda_246ef.tar.gz
cd ../
