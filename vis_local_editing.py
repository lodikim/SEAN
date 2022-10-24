from pickletools import uint8
import cv2
import os
from PIL import Image
import vis
import numpy as np
import torch
import torch.nn.functional as F
#import visualize_shade


# 아래 실행 코드로 실행
## python vis_local_editing.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0

###
source_image_name = "28009.jpg"     # Ex. "28009.jpg"
target_image_name = "28024.jpg"     # Ex. "28019.jpg"
swap_target = [2]                # Ex. [4, 5]
save_img_name = "local_editing_nose_shade"    # Ex. "test_swap_eyes"
###

full_image = Image.new("RGB", (256*3, 256*1), "white")     # image size: 256 x 256

source_label_name = source_image_name[:-4] + ".png"
target_label_name = target_image_name[:-4] + ".png"

source_image = vis.visualize(source_image_name, source_label_name, source_image_name, swap_target)
#### code for masking swap target
source_label = cv2.imread("datasets/CelebA-HQ/test/labels/" + source_label_name)
source_masks = np.in1d(source_label, swap_target)   # (786432, )
source_masks = source_masks.reshape((512, 512, 3)).astype('float32')
source_masks = cv2.resize(source_masks, (256, 256))
source_final = cv2.addWeighted(np.array(source_image, dtype='uint8'), 1, source_masks.astype('uint8'), 80, 0)
source_final = Image.fromarray(source_final, 'RGB')
#cv2.imwrite('./vis/local-editing/test-image.png', cv2.cvtColor(np.array(source_image, dtype='uint8'), cv2.COLOR_BGR2RGB))
#cv2.imwrite('./vis/local-editing/test-mask.png', source_masks*100)
####
full_image.paste(source_final, (0, 0))

target_image = vis.visualize(target_image_name, target_label_name, target_image_name, swap_target)
#### code for masking everything except swap target
target_label = cv2.imread("datasets/CelebA-HQ/test/labels/" + target_label_name)
target_masks = np.in1d(target_label, swap_target)   # (786432, )
target_masks = target_masks*(-1) + 1
target_masks = target_masks.reshape((512, 512, 3)).astype('float32')
target_masks = cv2.resize(target_masks, (256, 256))
target_final = cv2.addWeighted(np.array(target_image, dtype='uint8'), 1, target_masks.astype('uint8'), 150, 0)
target_final = Image.fromarray(target_final, 'RGB')
####
full_image.paste(target_final, (256*1, 0))

sean_image = vis.visualize(source_image_name, source_label_name, target_image_name, swap_target)
full_image.paste(sean_image, (256*2, 0))

os.makedirs('./vis/local-editing', exist_ok = True)
full_image.save('./vis/local-editing/' + save_img_name + '.png')    # save to png