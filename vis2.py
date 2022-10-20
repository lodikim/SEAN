import cv2
import torch
import os
from glob import glob
import numpy as np
from util import util
from utils import utils
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from PIL import Image
from data.base_dataset import get_params, get_transform

# 마스크까지 바꾸는 코드
## python vis2.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0

### 
source_image_name = "28002.jpg"     # Ex. "28000.jpg"
source_label_name = "28002.png"     # Ex. "28000.png"
target_image_name = "28002.jpg"     # Ex. "28083.jpg"
target_label_name = "scale_28002.png"  
swap_target = [4, 5]                # Ex. [4, 5]
save_img_name = "scale_28002_image"    # Ex. "test_swap_eyes"
### 

def input_style_codes(source_image_name, target_image_name, swap_target):

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    source_style_code_folder = 'styles_test/style_codes/' + source_image_name
    target_style_code_folder = 'styles_test/style_codes/' + target_image_name
    input_style_dic = {}
    label_count = []

    #style_img_mask_dic = {}

    for i in range(19):
        if i in swap_target:
            input_style_code_folder = target_style_code_folder
        else:
            input_style_code_folder = source_style_code_folder

        input_style_dic[str(i)] = {}

        input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
        input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in average_category_folder_list]

        for style_code_path in average_category_list:
            if style_code_path in input_category_list:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(input_style_code_folder, str(i), style_code_path+'.npy'))).cuda()

                if style_code_path == 'ACE':
                    #style_img_mask_dic[str(i)] = "datasets/CelebA-HQ/test/images/28083.jpg"
                    label_count.append(i)

            else:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
    
    return input_style_dic

def change_label(source_label_name, target_label_name, swap_target):
    source_label = cv2.imread("datasets/CelebA-HQ/test/labels/" + source_label_name)
    #target_label = cv2.imread("datasets/CelebA-HQ/test/labels/" + target_label_name)
    target_label = cv2.imread("vis/labels/" + target_label_name)

    source_tensor = torch.tensor(source_label)
    target_tensor = torch.tensor(target_label)

    source_stylevector = torch.ones((4,19,512))
    target_stylevector = torch.zeros((4,19,512))

    _ , result_seg = utils.swap_components(source_stylevector, target_stylevector, source_tensor, target_tensor, swap_target_idxes=swap_target, swap_mode='both')
    result_label = np.array(result_seg)
    return result_label


def visualize2(source_image_name, source_label_name, target_image_name, target_label_name, swap_target):

    opt = TestOptions().parse()
    opt.status = 'UI_mode'
    model = Pix2PixModel(opt)
    model.eval()

    torch.manual_seed(0)

    image_path = "datasets/CelebA-HQ/test/images/" + source_image_name
    result_label = change_label(source_label_name, target_label_name, swap_target)
    label_img = result_label[:, :, 0]

    label = Image.fromarray(label_img)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc  # 'unknown' is opt.label_nc
    label_tensor.unsqueeze_(0)

    image_tensor = torch.zeros([1, 3, 256, 256])


    # if using instance maps
    if opt.no_instance:
        instance_tensor = torch.Tensor([0])

    data = {'label': label_tensor,
        'instance': instance_tensor,
        'image': image_tensor,
        'path': image_path,
    }

    data['obj_dic'] = input_style_codes(source_image_name, target_image_name, swap_target)

    generated = model(data, mode='UI_mode') # UI mode일 때만 obj_dic = data['obj_dic'] 적용

    tile = opt.batchSize > 8
    generated_img = util.tensor2im(generated, tile=tile)[0]
    image_pil = Image.fromarray(generated_img)

    return image_pil
    


if __name__ == "__main__":

    image = visualize2(source_image_name, source_label_name, target_image_name, target_label_name, swap_target)

    os.makedirs('./vis', exist_ok = True)
    image.save('./vis/' + save_img_name + '.png')    # save to png