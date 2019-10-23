nohup python train_patn_funit.py --name TWO_STAGE_FULL_CORRECT_FLIP_20191022 --model PATN_FUNIT_FULL --nThreads 4 --batchSize 8 --gpu_ids 0,1 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --which_model_netG PATN --norm instance --DG_ratio 3 --niter 10 --niter_decay 20 --lr_policy plateau --save_epoch_freq 5 > train_log_two_stage_20191022.txt &
#nohup python train_patn_funit.py --name TWO_STAGE_FIX_PATN_tier2 --nThreads 8 --batchSize 16 --gpu_ids 0,2 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --no_flip --max_dataset_size 4000 --which_model_netG PATN --norm instance --niter 10 --niter_decay 20 --save_epoch_freq 5  > train_log_fix_patn_20191007.txt &
# nohup python train_patn_funit.py --name TWO_STAGE_FUNIT_GEN_ONLY --model PATN_FUNIT_NO_GAN --nThreads 4 --batchSize 16 --gpu_ids 0,2 --dataroot ./fashion_data --pairLst ./fashion_data/fasion-resize-pairs-train.csv --resize_or_crop no --no_flip --which_model_netG PATN --norm instance --niter 10 --niter_decay 20 --save_epoch_freq 1 > train_log_funit_gen_only_20191005.txt &
