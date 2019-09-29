import os, cv2
from tqdm import tqdm
import numpy as np

def make_person_classes(dir_P):    
    dir_C = dir_P + '_classes'
    if not os.path.isdir(dir_C):
        os.makedirs(dir_C)
        
    img_items = os.listdir(dir_P)
    img_items.sort()
    cur_person_name = 'None'
    cur_bundle = []
    loop = tqdm(img_items)
    for item in loop:
        person_name = item[:item.rfind('_')] #something like 'fashionMENDenimid0000056501_1front.jpg'
        img = cv2.imread(os.path.join(dir_P, item))
        loop.set_description(person_name)
        if person_name == cur_person_name:
            cur_bundle.append(img)
        else:
            if cur_person_name != 'None':
                npy_bundle = np.stack(cur_bundle, axis=0)
                np.save(os.path.join(dir_C, cur_person_name + '.npy'), npy_bundle)
            cur_person_name = person_name
            cur_bundle = [img]
            

if __name__ == '__main__':
    make_person_classes('./fashion_data/train')
    make_person_classes('./fashion_data/test')