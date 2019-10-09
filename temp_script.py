import os, cv2
import PIL
import skimage.io
import jpeg4py
from tqdm import tqdm
from time import time
import numpy as np
import pickle
import torchvision.transforms as transforms
import torch

def make_person_classes(dir_P):    
    dir_C = dir_P + '_classes'
    if not os.path.isdir(dir_C):
        os.makedirs(dir_C)

    # class_labels
    class_labels = {}
    class_no = 0
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
            class_labels[cur_person_name] = class_no
            class_no += 1
    return class_labels, class_no

def test_speed():
    # We load all images into torch.FloatTensor!
    
    directory = 'fashion_data/train'
    items = [os.path.join(directory,s) for s in os.listdir(directory)]
    print('Image num: %d' % len(items))
    
    start = time()
    for item in tqdm(items):
        jpeg4py_img = jpeg4py.JPEG(item).decode().transpose(2,0,1).astype(np.float) / 127.5 - 1
        jpeg4py_torch = torch.from_numpy(jpeg4py_img)
    print('jpeg4py time:  %.4f s' % (time() - start))
    
    start = time()
    for item in tqdm(items):
        CV2_img = cv2.imread(item).transpose(2,0,1).astype(np.float) / 127.5 - 1
        CV2_torch = torch.from_numpy(CV2_img)
    
    print('CV2 time:  %.4f s' % (time() - start))
    
    transform_list = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    start = time()
    for item in tqdm(items):
        PIL_img = PIL.Image.open(item).convert('RGB')
        PIL_torch = transform(PIL_img)
    
    print('PIL time:  %.4f s' % (time() - start))
    print(torch.max(torch.abs(jpeg4py_torch - PIL_torch)))

if __name__ == '__main__':
    
    #train_labels, train_class_no = make_person_classes('./fashion_data/train')
    #with open('./fashion_data/train_classes.pickle', 'wb') as ftrain:
    #    pickle.dump(train_labels, ftrain)
    
    test_labels, test_class_no = make_person_classes('./fashion_data/test')
    with open('./fashion_data/test_classes.pickle', 'wb') as ftest:
        pickle.dump(test_labels, ftest)
    #print('train_no: %d, test_no: %d' % (train_class_no, test_class_no))
    #test_speed()