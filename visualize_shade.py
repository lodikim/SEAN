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

# 1. source_image_name, source_label_name 선택 (둘은 같아야 함)
# 2. target_image_name과 바꾸고 싶은 얼굴 요소 (swap_target) 선택
# 3. 저장할 이름 선택 (save_img_name), 'vis' 폴더 안에 저장됨
# 4. 아래 실행 코드로 실행
## python vis.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0

### 
source_image_name = "28002.jpg"     # Ex. "28000.jpg"
source_label_name = "28002.png"     # Ex. "28000.png"
target_image_name = "28002.jpg"     # Ex. "28083.jpg"
swap_target = [4, 5]                # Ex. [4, 5]
save_img_name = "28002_recon"    # Ex. "test_swap_eyes"
### 

def input_style_codes(source_image_name, target_image_name, swap_target, isSource):

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    source_style_code_folder = 'styles_test/style_codes/' + source_image_name
    target_style_code_folder = 'styles_test/style_codes/' + target_image_name
    input_style_dic = {}
    label_count = []

    #style_img_mask_dic = {}

    for i in range(19):
        shade = False
        if i in swap_target:
            input_style_code_folder = target_style_code_folder
            if isSource:
                shade = True
        else:
            input_style_code_folder = source_style_code_folder
            if not isSource:
                shade = True

        input_style_dic[str(i)] = {}

        input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
        input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in average_category_folder_list]

        for style_code_path in average_category_list:
            if style_code_path in input_category_list:
                style_numpy = np.load(os.path.join(input_style_code_folder, str(i), style_code_path+'.npy'))
                if shade:
                    style_numpy = style_numpy/100
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(style_numpy).cuda()

                if style_code_path == 'ACE':
                    #style_img_mask_dic[str(i)] = "datasets/CelebA-HQ/test/images/28083.jpg"
                    label_count.append(i)

            else:
                style_numpy = np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))
                if shade:
                    style_numpy = style_numpy/100
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(style_numpy).cuda()
    
    return input_style_dic


def visualize(source_image_name, source_label_name, target_image_name, swap_target, isSource):

    opt = TestOptions().parse()
    opt.status = 'UI_mode'
    model = Pix2PixModel(opt)
    model.eval()

    torch.manual_seed(0)

    image_path = "datasets/CelebA-HQ/test/images/" + source_image_name
    mat_img_path = "datasets/CelebA-HQ/test/labels/" + source_label_name
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

    data['obj_dic'] = input_style_codes(source_image_name, target_image_name, swap_target, isSource)

    generated = model(data, mode='UI_mode') # UI mode일 때만 obj_dic = data['obj_dic'] 적용

    tile = opt.batchSize > 8
    generated_img = util.tensor2im(generated, tile=tile)[0]
    image_pil = Image.fromarray(generated_img)

    return image_pil
    


if __name__ == "__main__":

    image = visualize(source_image_name, source_label_name, target_image_name, swap_target)

    os.makedirs('./vis', exist_ok = True)
    image.save('./vis/' + save_img_name + '.png')    # save to png