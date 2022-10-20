import os
import cv2
import torch
import numpy as np
from utils import utils

### 
source_image_name = "28002.jpg"     # Ex. "28000.jpg"
source_label_name = "28002.png"     # Ex. "28000.png"
swap_target = [4, 5]                # Ex. [4, 5]
save_img_name = "scale_28002"    # Ex. "test_swap_eyes"
### 

if __name__ == "__main__":

    # 1. Input label
    source_label = cv2.imread("datasets/CelebA-HQ/test/labels/" + source_label_name)    # (512, 512, 3)

    # 2. Get one-hot vector of target component
    one_hot_vecs = utils.LabelToOnehot_noInter(source_label)    # (1, 19, 512, 512)

    for idx in swap_target:
    # 3. Find contours & center
        target_eye = one_hot_vecs[0][idx]
        target_eye = target_eye.cpu().detach().numpy()
        target_eye = target_eye.astype(np.uint8)
        cv2.imshow('Image', target_eye*100)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        contours, hierarchy = cv2.findContours(target_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        M = cv2.moments(contours[0])
        center_x = round(M['m10'] / M['m00'])
        center_y = round(M['m01'] / M['m00'])
        print("center X : '{}'".format(center_x))
        print("center Y : '{}'".format(center_y))
        

        # 4. Scale
        # 4-1. Resize
        scale_factor = 1.5
        scale = int(512*scale_factor)
        eye_resized = cv2.resize(target_eye, (scale, scale))
        print("resized eye shape: ", eye_resized.shape)
        # 4-2. Find new center
        contours_resized, hierarchy_resized = cv2.findContours(eye_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        M = cv2.moments(contours_resized[0])
        center_x_resized = round(M['m10'] / M['m00'])
        center_y_resized = round(M['m01'] / M['m00'])
        print("center X resized : '{}'".format(center_x))
        print("center Y resized : '{}'".format(center_y))
        # 4-3. Crop accordingly
        crop_eye = eye_resized[ int(center_y_resized-center_y):int(center_y_resized-center_y+512), int(center_x_resized-center_x):int(center_x_resized-center_x+512) ]
        print("cropped eye shape: ", crop_eye.shape)
        cv2.imshow('Image', crop_eye*100)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 5. Turn back into label
        crop_eye = torch.tensor(np.array(crop_eye)).float()
        # print('equal?', torch.equal(crop_eye, one_hot_vecs[0][4]))
        changes = one_hot_vecs[0][idx]-crop_eye
        crop_eye = torch.where(changes > 0, one_hot_vecs[0][idx], crop_eye) # Old eye -> must be eye
        one_hot_vecs[0][idx] = crop_eye   # one_hot_vecs는 확실하게 바뀜 (torch.equal로 확인)
        # print('equal?', torch.equal(crop_eye, one_hot_vecs[0][4]))
        # Problem: Now not one-hot vectors
        # Solution: New eye -> others>eye>skin
        new_eye = (changes == -1).nonzero(as_tuple=False)

        for num in range(new_eye.size()[0]):
            i = new_eye[num][0]
            j = new_eye[num][1]
            max_found = False
            max_index = 0
            for k in range(19):
                if max_found:
                    if one_hot_vecs[0][2][i][j] == 1:
                        one_hot_vecs[0][idx][i][j] = 0
                    else:
                        one_hot_vecs[0][18-k][i][j] = 0
                if one_hot_vecs[0][18-k][i][j]==1:
                    max_found = True
        print('sum: ', torch.sum(one_hot_vecs)) # 512 * 512 =262144 -> correct
    output_label = utils.OnehotToLabel(one_hot_vecs)
    
    

    # 6. Visualize
    os.makedirs('./vis/labels', exist_ok = True)
    cv2.imwrite('./vis/labels/' + save_img_name + '.png', output_label.cpu().detach().numpy())
    result = np.concatenate((source_label, output_label.cpu().detach().numpy()), axis=1)
    cv2.imwrite('./vis/labels/test.png', result*10)