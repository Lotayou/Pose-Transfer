import time
from options.train_funit_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage.io import imsave
	
from random import seed
seed(2333)
    
from torch.backends import cudnn
cudnn.enabled = True
cudnn.benchmark = True

opt = TrainOptions().parse(
	# use_debug_mode=True  # debug
    use_debug_mode=False
)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(('#training images = %d' % dataset_size))

model = create_model(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.train_one_step()

        if total_steps % opt.print_freq == 0:
            im_npy = model.get_current_visuals()
            im_name = '%s/images/%03d_%06d.png' % (model.save_dir, epoch, i)
            imsave(im_name, im_npy)
            print(model.get_error_log(total_steps))

        if total_steps % opt.save_latest_freq == 0:
            print(('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps)))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print(('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps)))
        model.save('latest')
        model.save(epoch)

    print(('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)))
    model.update_learning_rate()
