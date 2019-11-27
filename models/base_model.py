import os
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.isdir(os.path.join(self.save_dir, 'images')):
            os.makedirs(os.path.join(self.save_dir, 'images'))

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            torch.save(network.module.state_dict(), save_path)
        else:
            torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = torch.load(save_path)
        try:
            network.load_state_dict(state_dict)
        except:
            # Could be a DataParallel model, remove 'module.' prefix from all keys
            print('Net %s could belong to a DataParallel object, manually removing key prefix...' % network_label)
            from collections import OrderedDict
            correct_dict = OrderedDict()
            for k, v in state_dict.items():
                kk = k.replace('module.', '')
                correct_dict[kk] = v
            network.load_state_dict(correct_dict)
        finally:
            print('Loading successful')
                

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        print('Learning rate updated')
        # lr = self.optimizers[0].param_groups[0]['lr']
        # print(('learning rate = %.7f' % lr))
