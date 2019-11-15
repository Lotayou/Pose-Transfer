# nohup python train_patn_funit.py --name TWO_STAGE_FIX_PATN_tier2 --nThreads 8 --batchSize 16 --gpu_ids 0,2 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --no_flip --max_dataset_size 4000 --which_model_netG PATN --norm instance --niter 10 --niter_decay 20 --save_epoch_freq 5  > train_log_fix_patn_20191007.txt &
# nohup python train_patn_funit.py --name TWO_STAGE_FUNIT_GEN_ONLY --model PATN_FUNIT_NO_GAN --nThreads 4 --batchSize 16 --gpu_ids 0,2 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --no_flip --which_model_netG PATN --norm instance --niter 10 --niter_decay 20 --save_epoch_freq 1 > train_log_funit_gen_only_20191005.txt &
 # --no_global_res --continue_train 
# nohup python train_patn_funit.py --name TWO_STAGE_FULL_v2_20191024 --model PATN_FUNIT_FULL_v2 --dataset_mode 'keypoint_funit_v2' --nThreads 4 --batchSize 8 --gpu_ids 0,1 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --which_model_netG PATN --norm instance --DG_ratio 3 --niter 2 --niter_decay 10 --save_epoch_freq 5 > train_log_two_stage_v2_20191024.txt &

# fine-tune 1: modify vgg weights (20191115: network broke, data overflow for one pose channel, terminated)
# nohup python train_patn_funit.py --name TWO_STAGE_FULL_II_modified_vgg_wgts_20191113 --model PATN_FUNIT_FULL --nThreads 4 --batchSize 8 --gpu_ids 0,1 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --which_model_netG PATN --norm instance --DG_ratio 3 --niter 2 --niter_decay 10 --save_epoch_freq 1  --G1_lr 0.0001 --G2_lr 0.0001 --D_lr 0.0001 --use_custom_vgg_weights > train_log_two_stage_20191113.txt&

# fine-tune 2: using multiscale discriminator
nohup python train_patn_funit.py --name TWO_STAGE_FULL_II_multiscale_D_20191115 --model PATN_FUNIT_FULL --nThreads 4 --batchSize 8 --gpu_ids 0,1 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --which_model_netG PATN --norm instance --DG_ratio 3 --niter 2 --niter_decay 10 --save_epoch_freq 1  --G1_lr 0.0001 --G2_lr 0.0001 --D_lr 0.0001 --which_model_netD multiscale > train_log_two_stage_20191115.txt &