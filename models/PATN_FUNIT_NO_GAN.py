import numpy as np
import os
import yaml
from collections import OrderedDict
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from shutil import copyfile
import sys
import torch

from .PATN import TransferModel
from .FUNIT_module.funit_model_nogan import FUNITModel
from .radam import RAdam

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
            
            
class PatnFunitModel(BaseModel):
    def name(self):
        return 'PATH + FUNIT Model'
       
    def initialize(self, opt):
        super(PatnFunitModel, self).initialize(opt)
        self.patn_model = TransferModel()
        self.patn_model.initialize(opt)        
        # force loading
        pretrained_patn_dir = './checkpoints/pretrained_patn_fashion/latest_net_netG.pth'
        copyfile(pretrained_patn_dir, 
            os.path.join(self.save_dir, 'latest_net_netG.pth'))
        self.load_network(self.patn_model.netG, 'netG', 'latest')
        
        with open(opt.funit_options, 'r') as fin:
            self.funit_opt = yaml.load(fin, Loader=yaml.FullLoader)
        
        # load checkpoints        
        self.funit_model = FUNITModel(self.funit_opt)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.funit_model.gen_test, 'net_G_test(funit)', which_epoch)
            self.load_network(self.funit_model.gen, 'net_G(funit)', which_epoch)

        # set optimizers
        if self.isTrain:
            funit_gen_params = list(self.funit_model.gen.parameters())
            # 20191004: Let's play with RAdam (LOL)
            '''
            self.funit_gen_opt = torch.optim.RMSprop(
                [p for p in funit_gen_params if p.requires_grad],
                lr=self.funit_opt['lr_dis'], 
                weight_decay=self.funit_opt['weight_decay'])
            '''
            self.funit_gen_opt = RAdam(
                [p for p in funit_gen_params if p.requires_grad],
                lr=self.funit_opt['lr_gen'], betas=(0.5, 0.999),
                weight_decay=self.funit_opt['weight_decay']
            )
            # 20191003: Use patn scheduler options to guide funit, since we fix patn for now.
            self.funit_gen_scheduler = networks.get_scheduler(self.funit_gen_opt, self.opt)

            # NOTE: We only attach scheduler to funit modules for now.
            self.schedulers = [
                self.funit_gen_scheduler,
            ]
            if opt.continue_train:
                which_epoch = opt.which_epoch
                self.load_network(self.funit_gen_scheduler, 'net_G_scheduler(funit)', which_epoch)

            self.fix_patn = True
            
        # print basic info
        print('---------- Networks initialized -------------')
        networks.print_network(self.patn_model.netG)
        networks.print_network(self.funit_model.gen)
        networks.print_network(self.funit_model.gen_test)
        print('-----------------------------------------------')
        
        self.patn_model.netG.cuda()
        self.funit_model.cuda()
        self.patn_model.netG.eval()  # Cool:)
        self.funit_model.train()
        
        if len(self.opt.gpu_ids) > 1:
            # only use net_G for now will suffice
            # note: it seems okay to wrap the whole model as a DataParallel object, but the internal mechanics is unknown...
            self.funit_model = torch.nn.DataParallel(self.funit_model, device_ids=self.opt.gpu_ids)
            
    
    def set_input(self, input):
        # TODO: change dataset structure, automatically load all ground_truth images of the same person
        #   update: loading Ys for testing and training case B only.
        
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']
        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()
        
        if 'Ys' in input:
            self.Ys = input['Ys'].cuda() if len(self.gpu_ids) > 0 else input['Ys']
        
        
    def train_one_step(self):
        '''
        Connection:

                            PATN
            (P1, BP1, BP2) ------> patn_fake_p2 -------|  FUNIT
                                                       |---------> final_p2 --| D
                        Ys = (y1,y2,...yk) ------------|                      |---> real/fake?
                                                                      P2 -----|

            20191001:
            TODO:
                Two different schemes for training
                A: k=1 (Only use P1) as input...
                B: k=3 use all other images of the same person (except P2) as input...
            TODO: For testing, implementing an additional TestModel with a new method self.eval
                that takes multiple tier P as input... Follow FUNIT test_k_shot...

        Input is set with self.set_input(self, input)
        '''
        # stage I: PATN forward
        # note: we fix patn for now
    
        with torch.no_grad():
            G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
            self.stage_I_output = self.patn_model.netG(G_input) 
        
        # FUNIT (and PATN) generator update
        self.funit_gen_opt.zero_grad()
        if not self.fix_patn:
            self.patn_model.optimizer_G.zero_grad()
        # 20191005 use strong reference (ground truth P2)
        self.stage_II_output, self.loss_dict = self.funit_model(
            self.stage_I_output, self.input_P1, self.input_P2, self.funit_opt
        )
        self.funit_gen_opt.step()
        this_model = self.funit_model.module if len(self.opt.gpu_ids) > 1 else self.funit_model
        update_average(this_model.gen_test, this_model.gen)
        if not self.fix_patn:
            self.patn_model.optimizer_G.step()

        torch.cuda.synchronize()

    def get_current_visuals(self):
        # 2 by 3:
        # A, PA, B'
        # B, PB, B''
        P1 = util.tensor2im(self.input_P1.data)
        P2 = util.tensor2im(self.input_P2.data)
        O1 = util.tensor2im(self.stage_I_output.data)
        O2 = util.tensor2im(self.stage_II_output.data)
        BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]
        
        visual_tensor = np.concatenate((
            np.concatenate((P1, BP1, O1), axis=1),
            np.concatenate((P2, BP2, O2), axis=1),
        ), axis=0)
        return visual_tensor
        
    def get_error_log(self, iter_num):
        log = '[Iter %d] ' % iter_num
        for k, v in self.loss_dict.items():
            log += '%s: %.4f, ' % (k, torch.sum(v).item() / self.opt.batchSize)
        return log
        
    def save(self, label):
        # patn
        if not self.fix_patn:
            self.save_network(self.patn_model.netG, 'netG(patn)', label, self.gpu_ids)
            if self.opt.with_D_PB:
                self.save_network(self.patn_model.netD_PB,  'netD_PB(patn)', label, self.gpu_ids)
            if self.opt.with_D_PP:
                self.save_network(self.patn_model.netD_PP, 'netD_PP(patn)', label, self.gpu_ids)
        # funit
        f_model = self.funit_model.module if len(self.opt.gpu_ids) > 1 else self.funit_model
        self.save_network(f_model.gen, 'net_G(funit)', label, self.gpu_ids)
        self.save_network(f_model.gen_test, 'net_G_test(funit)', label, self.gpu_ids)
        self.save_network(self.funit_gen_scheduler, 'net_G_scheduler(funit)', label, self.gpu_ids)

    def activating_patn(self):
        self.fix_patn = False

