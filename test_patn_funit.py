from tqdm import tqdm
import os
import numpy as np
from options.test_funit_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage.io import imsave
from skimage.measure import compare_ssim
	
from torch.backends import cudnn
cudnn.enabled = True
cudnn.benchmark = True

def padding(x):
    _h, _w, _c = x.shape
    _im = (np.ones((_h, _h, _c)) * 255).astype(np.uint8)
    _left = (_h - _w) // 2
    _im[:, _left: _left + _w, :] = x
    return _im
    

opt = TestOptions().parse(
	#use_debug_mode=True  # debug
    use_debug_mode=False
)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(('#testing images = %d' % dataset_size))

testing_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.isdir(testing_dir):
    os.makedirs(testing_dir)

model = create_model(opt)
model = model.eval()
h, w = 256, 176
opt.how_many = 99999
ssim_sum1, ssim_sum2 = 0, 0
no = 0

file = open(os.path.join(opt.results_dir, opt.name+'.txt'), 'w')

for data in dataset:
    model.set_input(data)
    model.test()
    test_img = model.get_current_visuals()
    imsave(os.path.join(testing_dir, '%6d.png' % no), test_img)
    no += 1
    
    if no > opt.how_many:
        break
    
    gt = padding(test_img[h:,:w])
    gen1 = padding(test_img[:h,2*w:])
    gen2 = padding(test_img[h:,2*w:])
    
    # imsave(os.path.join(testing_dir, 'gt.png'), gt)
    # imsave(os.path.join(testing_dir, 'g1.png'), gen1)
    # imsave(os.path.join(testing_dir, 'g2.png'), gen2)
    
    ssim_score1 = compare_ssim(gt, gen1, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, multichannel=True,
        data_range=gen1.max() - gen1.min()
    )
    ssim_score2 = compare_ssim(gt, gen2, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, multichannel=True,
        data_range=gen2.max() - gen2.min()
    )
    file.write('[%.6f / %.6f]\n' % (ssim_score1, ssim_score2))
    file.flush()
    ssim_sum1 += ssim_score1
    ssim_sum2 += ssim_score2
    
file.write('Stage I: Mean SSIM = %.6f\n' % (ssim_sum1 / (no - 1)))
file.write('Stage II: Mean SSIM = %.6f\n' % (ssim_sum2 / (no - 1)))
file.close()
