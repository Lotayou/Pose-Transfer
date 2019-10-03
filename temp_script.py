import os, cv2
from tqdm import tqdm
import numpy as np
import pickle

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



if __name__ == '__main__':
    train_labels, train_class_no = make_person_classes('./fashion_data/train')
    test_labels, test_class_no = make_person_classes('./fashion_data/test')
    print('train_no: %d, test_no: %d' % (train_class_no, test_class_no))
    with open('./fashion_data/train_classes.pickle', 'wb') as ftrain:
        pickle.dump(train_labels, ftrain)
    with open('./fashion_data/test_classes.pickle', 'wb') as ftest:
        pickle.dump(test_labels, ftest)

