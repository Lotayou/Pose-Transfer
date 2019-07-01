import os
from tqdm import tqdm

def convert(input_str):
    # input be like: fashionMENShirts_Polosid0000180202_4full.jpg
    # output should be like: fashion/MEN/Shirts_Polos/id_00001802/02_4_full.jpg
    idx0 = 7  # 'fashion'
    sub0 = input_str[:idx0]
    idx1 = input_str.find('MEN')+3
    sub1 = input_str[idx0: idx1]
    idx2 = input_str.find('id000')
    sub2 = input_str[idx1: idx2]
    idx3 = idx2 + 10
    sub3 = 'id_' + input_str[idx2+2: idx3]
    sub4 = input_str[idx3:idx3+4] + '_' + input_str[idx3+4:]
    return '/'.join(['img', sub1, sub2, sub3, sub4])


# path for downloaded fashion images
root_fashion_dir = '/backup1/Datasets/DeepFashion'
assert len(root_fashion_dir) > 0, 'please give the path of raw deep fashion dataset!'

train_images = []
train_f = open('fashion_data/train.lst', 'r')
for lines in train_f:
    lines = lines.strip()
    if lines.endswith('.jpg'):
        train_images.append(lines)

test_images = []
test_f = open('fashion_data/test.lst', 'r')
for lines in test_f:
    lines = lines.strip()
    if lines.endswith('.jpg'):
        test_images.append(lines)

train_path = 'fashion_data/train'
if not os.path.exists(train_path):
    os.mkdir(train_path)

for item in tqdm(train_images):
    from_ = os.path.join(root_fashion_dir, convert(item))
    to_ = os.path.join(train_path, item)
    os.system('cp %s %s' %(from_, to_))


test_path = 'fashion_data/test'
if not os.path.exists(test_path):
    os.mkdir(test_path)

for item in tqdm(test_images):
    from_ = os.path.join(root_fashion_dir, convert(item))
    to_ = os.path.join(test_path, item)
    os.system('cp %s %s' %(from_, to_))
