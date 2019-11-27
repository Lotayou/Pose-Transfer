import os.path
from cv2 import imread
import random
import pandas as pd
import numpy as np
import torch
import pickle
from jpeg4py import JPEG

try:
    from base_dataset import BaseDataset, get_transform
except:
    from .base_dataset import BaseDataset, get_transform


def make_person_classes(dir_P, dir_C):
    import os, tqdm, cv2

    os.makedirs(dir_C)
    items = os.listdir(dir_P)
    items.sort()
    cur_person_name = 'None'
    cur_bundle = []
    for item in tqdm.tqdm(items):
        person_name = item[:item.rfind('_')]  # something like 'fashionMENDenimid0000056501_1front.jpg'
        img = cv2.imread(os.path.join(dir_P, item))
        if person_name == cur_person_name:
            cur_bundle.append(img)
        elif cur_person_name != 'None':
            npy_bundle = np.stack(cur_bundle, axis=0)
            np.save(os.path.join(dir_C, person_name + '.npy'), npy_bundle)
            cur_person_name = person_name
            cur_bundle = [img]


class KeyFUNITDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase)  # person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K')  # keypoints
        self.dir_C = os.path.join(opt.dataroot, opt.phase + '_classes')  # class bundle, each person a class
        self.dir_L = os.path.join(opt.dataroot, opt.phase + '_body_part_labels')
        if not os.path.isdir(self.dir_C):
            make_person_classes(self.dir_P, self.dir_C)

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)
        
        # LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
        # 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
        self.pose_swapping_index = torch.tensor([
            0,1, 5,6,7, 2,3,4, 11,12,13, 8,9,10, 15,14, 17,16
        ], dtype=torch.long)

        with open(os.path.join(opt.dataroot, opt.phase + '_classes.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

        self.size = min(self.size, opt.max_dataset_size)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        # WTF?
        # if self.opt.phase == 'train':
        #     index = random.randint(0, self.size - 1)
        
        # 20191004 Use jpeg4py to replace PIL
        
        P1_name, P2_name = self.pairs[index]
        # 20190930: Add labels for FUNIT
        class_name = P1_name[:P1_name.rfind('_')]
        
        label = torch.tensor([self.labels[class_name]], dtype=torch.float)

        P1_path = os.path.join(self.dir_P, P1_name)  # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy')  # bone of person 1
        P2_path = os.path.join(self.dir_P, P2_name)  # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy')  # bone of person 2
        
        #P1_img = Image.open(P1_path).convert('RGB')
        #P2_img = Image.open(P2_path).convert('RGB')
        P1_img = JPEG(P1_path).decode().transpose(2,0,1)
        P2_img = JPEG(P2_path).decode().transpose(2,0,1)

        BP1_img = np.load(BP1_path)  # h, w, c
        BP2_img = np.load(BP2_path)
        
        PLW1_path = os.path.join(self.dir_L, P1_name.split('.')[0] + '.npy') # part labels of person 1
        PLW2_path = os.path.join(self.dir_L, P2_name.split('.')[0] + '.npy') # part labels of person 2
        PLW1_img = np.load(PLW1_path)
        PLW2_img = np.load(PLW2_path)
        
        # use flip
        if self.opt.phase == 'train':
            if not self.opt.no_flip:
                flip_random = random.uniform(0, 1)

                if flip_random > 0.5:
                    #print('fliped ...')
                    P1_img = np.array(P1_img[:,:,::-1])
                    P2_img = np.array(P2_img[:,:,::-1])

                    # 20191020: This is a potential bug.                     
                    # Left-right channels should be swapped along with spatial coordinates.
                    BP1_img = np.array(BP1_img[:, ::-1, :])
                    BP2_img = np.array(BP2_img[:, ::-1, :])
                    BP1_img = BP1_img[:, :, self.pose_swapping_index]   # 20191022: Modification successful
                    BP2_img = BP2_img[:, :, self.pose_swapping_index]
                    
                    PLW1_img = np.array(PLW1_img[:, ::-1])
                    PLW2_img = np.array(PLW2_img[:, ::-1])
                    
            # 20191024 Bug fix: numpy to torch after flipping.
            
            P1 = torch.from_numpy(P1_img).float() / 127.5 - 1
            P2 = torch.from_numpy(P2_img).float() / 127.5 - 1
            #print('P shape: ', P1.shape) 
            
            BP1 = torch.from_numpy(BP1_img).float().permute(2,0,1)
            BP2 = torch.from_numpy(BP2_img).float().permute(2,0,1)
            #print('BP shape: ', BP1.shape)
            
            PLW1 = torch.from_numpy(PLW1_img).float()
            PLW1 = torch.stack((PLW1,) * 3, dim=0) # expand to 3 channel
            PLW2 = torch.from_numpy(PLW2_img).float()
            PLW2 = torch.stack((PLW2,) * 3, dim=0)
            #print('PLW shape: ', PLW1.shape)
            
            return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'PLW1': PLW1, 'PLW2': PLW2,
                'P1_path': P1_name, 'P2_path': P2_name, 'label': label}
        else:
            
            # funit bundle (Notation following FUNIT paper)
            # FIXME: Ys has different size in dimension 0, need further guidance...
            # Ys = np.load(os.path.join(self.dir_C, class_name + '.npy'))
            # Ys = Ys[:k]
            P1 = torch.from_numpy(P1_img).float() / 127.5 - 1
            P2 = torch.from_numpy(P2_img).float() / 127.5 - 1
             
            BP1 = torch.from_numpy(BP1_img).float().permute(2,0,1)
            BP2 = torch.from_numpy(BP2_img).float().permute(2,0,1)
            
            PLW1 = torch.from_numpy(PLW1_img).float()
            PLW1 = torch.stack((PLW1,) * 3, dim=0) # expand to 3 channel
            PLW2 = torch.from_numpy(PLW2_img).float()
            PLW2 = torch.stack((PLW2,) * 3, dim=0)
            
            return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name, 
                'PLW1': PLW1, 'PLW2': PLW2,
                # 'Ys': Ys,
                'label': label}
        

    def __len__(self):
        return self.size

    def name(self):
        return 'KeyFUNITv2Dataset'


if __name__ == '__main__':
    # make_person_classes('../fashion_data/train')
    # make_person_classes('../fashion_data/test')
    
    im = imread(r'../fashion_data/small_body_part_labels/fashionMENDenimid0000056501_7additional_gray.png')
    #print(im)
    print(im.shape, im.dtype, im.max())

