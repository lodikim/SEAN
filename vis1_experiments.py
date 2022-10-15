# Mask: A
# Style Codes: 눈 제외 - A, 눈 - B
# 구성: Img A , Edited Img A, A Mask

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
from data.base_dataset import get_params, get_transform
from collections import OrderedDict
from PyQt5.QtGui import *

def input_style_codes():

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    source_style_code_folder = 'styles_test/style_codes/28000.jpg'
    target_style_code_folder = 'styles_test/style_codes/28083.jpg'
    input_style_dic = {}
    label_count = []
    swap_target = [4, 5]

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

if __name__ == "__main__":

    opt = TestOptions().parse()
    opt.status = 'UI_mode'
    model = Pix2PixModel(opt)
    model.eval()

    torch.manual_seed(0)

    image_path = "datasets/CelebA-HQ/test/images/28000.jpg"
    print(image_path)
    mat_img_path = "datasets/CelebA-HQ/test/labels/28000.png"
    print(mat_img_path)
    mat_img = cv2.imread(mat_img_path)
    label_img = mat_img[:, :, 0]

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

    data['obj_dic'] = input_style_codes()

    generated = model(data, mode='UI_mode') # UI mode일 때만 obj_dic = data['obj_dic'] 적용
    #print(generated)    # synthesized image
    #print(data['label'])    # input label

    tile = opt.batchSize > 8
    generated_img = util.tensor2im(generated, tile=tile)[0]
    image_pil = Image.fromarray(generated_img)
    image_pil.save('./test_swap_eyes.png')    # save to png

    # style: load_input_feature

    '''
    visuals = OrderedDict([('input_label', data['label']),
                           ('synthesized_image', generated)])

    for key, t in visuals.items():
        tile = opt.batchSize > 8
        if 'input_label' == key:
            t = util.tensor2label(t, opt.label_nc + 2, tile=tile)
        else:
            t = util.tensor2im(t, tile=tile)
        visuals[key] = t

    for label, image_numpy in visuals.items():
        #image_name = os.path.join(label, '%s.png' % (name))
        #save_path = os.path.join(image_dir, image_name)
        #util.save_image(image_numpy, save_path, create_dir=True)
        image_pil = Image.fromarray(image_numpy)
        image_pil.save('./test_vis1.png')
    '''



# python run_UI.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0
# python vis1.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0


'''
if __name__ == "__main__":
    # img1_label = cv2.imread('/home/compu/Downloads/JY/datasets/small_CelebA-HQ/labels/00000.png')
    # img2_label = cv2.imread('/home/compu/Downloads/JY/datasets/small_CelebA-HQ/labels/00001.png')
    img1_label = cv2.imread('C:/Users/bryan/Desktop/face-editing/SEAN/datasets/small_CelebA-HQ/labels/00000.png')
    img2_label = cv2.imread('C:/Users/bryan/Desktop/face-editing/SEAN/datasets/small_CelebA-HQ/labels/00001.png')

    img1_tensor = torch.tensor(img1_label)
    img2_tensor = torch.tensor(img2_label)

    img1_stylevector = torch.ones((4,19,512))
    img2_stylevector = torch.zeros((4,19,512))

    # result_stylevector: img1_stylevector에서 target_idx의 stylevector만 바뀜
    result_stylevector = utils.swap_components(img1_stylevector, img2_stylevector, img1_tensor, img2_tensor, swap_target_idxes=[4,5], swap_mode='matrix')

    # label + stylevector로 img 생성?
'''