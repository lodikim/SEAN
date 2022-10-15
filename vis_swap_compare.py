import cv2
import torch
import os
from glob import glob
import numpy as np
from util import util
import utils
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from PIL import Image
import matplotlib.pyplot as plt
from data.base_dataset import get_params, get_transform
from vis import visualize
from vis2 import visualize2

# A1 ~ A5의 이미지에 B의 style만 vs style+mask 빅 코드

# 아래 실행 코드로 실행
## python vis_swap_compare.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0

###
source_image_list = ["28000.jpg", "28001.jpg", "28002.jpg", "28003.jpg", "28004.jpg"]     # Ex. ["28000.jpg", "28001.jpg", "28002.jpg", "28003.jpg", "28004.jpg"]
target_image_name = "28009.jpg"     # Ex. "28009.jpg"
swap_target = [4, 5]                # Ex. [4, 5]
save_img_name = "swap_eyes_5p_compare"    # Ex. "test_swap_eyes"
###

full_image = Image.new("RGB", (256*(len(source_image_list)+1), 256*4), "white")     # image size: 256 x 256

count = 1
for source_image_name in source_image_list:
    source_label_name = source_image_name[:-4] + ".png"
    target_label_name = target_image_name[:-4] + ".png"
    
    source_image = Image.open("datasets/CelebA-HQ/test/images/" + source_image_name).resize((256, 256))
    full_image.paste(source_image, (256*count, 0))

    recons_image = visualize(source_image_name, source_label_name, source_image_name, swap_target)
    full_image.paste(recons_image, (256*count, 256))

    edited_image = visualize(source_image_name, source_label_name, target_image_name, swap_target)
    full_image.paste(edited_image, (256*count, 256*2))

    edited_image2 = visualize2(source_image_name, source_label_name, target_image_name, target_label_name, swap_target)
    full_image.paste(edited_image2, (256*count, 256*3))

    count = count+1

target_image = Image.open("datasets/CelebA-HQ/test/images/" + target_image_name).resize((256, 256))
full_image.paste(target_image, (0, 256*2+128))

os.makedirs('./vis', exist_ok = True)
full_image.save('./vis/' + save_img_name + '.png')    # save to png