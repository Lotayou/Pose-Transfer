import torch
import os

ckp_path = './checkpoints/fashion_PATN/latest_net_netG.pth'
save_path = './checkpoints/fashion_PATN_v1.0/latest_net_netG.pth'
if not os.path.isdir('./checkpoints/fashion_PATN_v1.0'):
  os.makedirs('./checkpoints/fashion_PATN_v1.0')
states_dict = torch.load(ckp_path)
states_dict_new = states_dict.copy()
for key in list(states_dict.keys()):
	if "running_var" in key or "running_mean" in key:
		del states_dict_new[key]

torch.save(states_dict_new, save_path)