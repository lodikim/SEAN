import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# RGB
# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

part_colors = {
        '0' : [0, 0, 0],
        '1' : [204, 0, 0], 
        '2' : [0, 255, 255], 
        '3' : [51, 255, 255],
        '4' : [51, 51, 255],
        '5' : [204, 0, 204],
        '6' : [204, 204, 0], 
        '7' : [102, 51, 0], 
        '8' : [255, 0, 0],
        '9' : [0, 204, 204],
        '10' : [76, 153, 0],
        '11' : [102, 204, 0],
        '12' : [255, 255, 0],
        '13' : [0, 0, 153],
        '14' : [255, 153, 51],
        '15' : [0, 51, 0],
        '16' : [0, 204, 0], 
        '17' : [0, 0, 204],
        '18' : [255, 51, 153],
    }

# input : (3, h, w)
# output : (1, h, w)
def ColorToLabel(seg_image):
    seg_image_ = np.array(seg_image)
    h, w, _ = seg_image_.shape

    canvas = np.zeros((h, w))
    for i in range(len(part_colors)):
        canvas = np.where(np.all(seg_image_==part_colors[str(i)],axis=-1),i,canvas)

    canvas_ = torch.tensor(canvas, dtype=torch.int64)
    return canvas_

# input : (h, w , 3)
# output : (1, 19, 256, 256)
def LabelToOnehot(Xt):
    Xt_ = torch.tensor(np.array(Xt)[:,:,0], dtype=torch.int64)
    h,w = Xt_.shape
    Xt_ = torch.reshape(Xt_,(1,1,h,w))
    one_hot_Xt = torch.FloatTensor(1, 19, h, w).zero_()
    one_hot_Xt_ = one_hot_Xt.scatter_(1, Xt_, 1.0)
    one_hot_Xt_ = F.interpolate(one_hot_Xt_,(256,256))
    return one_hot_Xt_

# img1 -> target
# img2 -> source
# inject source attribute to target
# matrix(b,19,512) seg(b,1,w,h)

# face parsing result label indexes
# 'background':0
# atts = {
# 'skin':1,
# 'l_brow':2,
# 'r_brow':3,
# 'l_eye':4,
# 'r_eye':5,
# 'eye_g':6,
# 'l_ear':7,
# 'r_ear':8,
# 'ear_r':9,
# 'nose':10,
# 'mouth':11,
# 'u_lip':12,
# 'l_lip':13,
# 'neck':14,
# 'neck_l':15,
# 'cloth':16,
# 'hair':17,
# 'hat':18
# }
atts = {
'skin':1,
'nose':2,
'eye_g':3,
'l_eye':4,
'r_eye':5,
'l_brow':6,
'r_brow':7,
'l_ear':8,
'r_ear':9,
'mouth':10,
'u_lip':11,
'l_lip':12,
'hair':13,
'hat':14,
'ear_r':15,
'neck_l':16,
'neck':17,
'cloth':18
}

def swap_components(img1_matrix, img2_matrix, img1_seg, img2_seg, swap_target_idxes=[2,4,5,6,7,10,11,12], swap_mode='both'):
    """
    matrix.size() >> [b,19,512]
    """
    def swap_matrix(img1_matrix, img2_matrix, swap_target_index):
        img1_matrix[:, swap_target_index, : ] = img2_matrix[:, swap_target_index, : ]
        return img1_matrix

    """
    seg.size() >> [b,1,w,h]
    ex) if a pixel is in the skin region, the value of the pixel is 1 
    if you want to change to other attribute, edit 'swap_target_idxes'. also you can swap multi attributes in one time
    """
    def swap_seg(img1_seg, img2_seg, swap_target_index):
        swap_target_index_ = torch.tensor(swap_target_index, dtype=img1_seg.dtype, device = img1_seg.device)
        img1_seg_ = torch.where(img1_seg==swap_target_index_, torch.tensor(atts["skin"], dtype=img1_seg.dtype, device = img1_seg.device), img1_seg) # skin
        img1_seg_ = torch.where(img2_seg==swap_target_index_, swap_target_index_, img1_seg_)
        return img1_seg_

    if swap_mode == 'both':
        for swap_target_index in swap_target_idxes:
            img1_matrix = swap_matrix(img1_matrix, img2_matrix, swap_target_index)
            img1_seg = swap_seg(img1_seg, img2_seg, swap_target_index)
        return img1_matrix, img1_seg

    elif swap_mode == 'matrix':
        for swap_target_index in swap_target_idxes:     # 2022-10-08 추가
            img1_matrix = swap_matrix(img1_matrix, img2_matrix, swap_target_index)
        return img1_matrix

    elif swap_mode == 'seg':
        for swap_target_index in swap_target_idxes:     # 2022-10-08 추가
            img1_seg = swap_seg(img1_seg, img2_seg, swap_target_index)
        return img1_seg


if __name__ == "__main__":
    # img1_label = cv2.imread('/home/compu/Downloads/JY/datasets/small_CelebA-HQ/labels/00000.png')
    # img2_label = cv2.imread('/home/compu/Downloads/JY/datasets/small_CelebA-HQ/labels/00001.png')
    img1_label = cv2.imread('C:/Users/bryan/Desktop/face-editing/SEAN/datasets/small_CelebA-HQ/labels/00000.png')
    img2_label = cv2.imread('C:/Users/bryan/Desktop/face-editing/SEAN/datasets/small_CelebA-HQ/labels/00001.png')


    # img2_label = cv2.flip(img2_label, 1)
    # img1_vis = cv2.imread('/home/compu/Downloads/JY/datasets/small_CelebA-HQ/vis/00000.png')
    # img2_vis = cv2.imread('/home/compu/Downloads/JY/datasets/small_CelebA-HQ/vis/00001.png')

    img1_tensor = torch.tensor(img1_label)
    img2_tensor = torch.tensor(img2_label)

    img1_stylevector = torch.ones((4,19,512))
    img2_stylevector = torch.zeros((4,19,512))

    result_stylevector, result_seg = swap_components(img1_stylevector, img2_stylevector, img1_tensor, img2_tensor, swap_mode='both')
    
    result_ = np.array(result_seg)
    result_ = np.concatenate((img1_label, img2_label, result_), axis=1)
    cv2.imwrite('./test.png',result_*10)

    # # 별개
    # result = ColorToLabel(img1_vis)

    # result = LabelToOnehot(img1_label)