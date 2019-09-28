import numpy as np
import torch
import os
import yaml
from collections import OrderedDict
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import sys
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

from .PATN import TransferModel
from .FUNIT_module.funit_model import FUNITModel

class PatnFunitModel(BaseModel):
    def name(self):
        return 'PATH+FUNIT Model'
       
    def initialize(self, opt)
        self.patn_model = TransferModel()
        self.patn_model.initialize(opt)
        
        with open(opt.funit_options, 'r') as fin:
            self.funit_opt = yaml.load(fin, Loader=yaml.FullLoader)
        
        # load checkpoints        
        self.funit_model = FUNITModel(self.funit_opt)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            # Different from FUNIT design, do not save gen_test since it's merely a deepcopy of gen
            self.load_network(self.funit_model.gen, 'net_G(funit)', which_epoch)
            self.load_network(self.funit_model.dis, 'net_D(funit)', which_epoch)
            
            # load schedulers for funit model
            # self.load_network(self.funit_model.gen_opt)
        
        # set optimizers
        if self.isTrain:
            funit_dis_params = list(self.funit_model.dis.parameters())
            funit_gen_params = list(self.funit_model.gen.parameters())
            self.funit_dis_opt = torch.optim.RMSprop(
                [p for p in funit_dis_params if p.requires_grad],
                lr=self.funit_opt['lr_gen'], 
                weight_decay=self.funit_opt['weight_decay'])
            self.funit_gen_opt = torch.optim.RMSprop(
                [p for p in gen_params if p.requires_grad],
                lr=self.funit_opt['lr_dis'], 
                weight_decay=self.funit_opt['weight_decay'])
            self.funit_dis_scheduler = networks.get_scheduler(self.funit_dis_opt, cfg)
            self.funit_gen_scheduler = networks.get_scheduler(self.funit_gen_opt, cfg)
            if opt.continue_train:
                which_epoch = opt.which_epoch
                self.load_network(self.funit_dis_scheduler, 'net_D_scheduler(funit)', which_epoch)
                self.load_network(self.funit_gen_scheduler, 'net_G_scheduler(funit)', which_epoch)
            
        # print info
        print('---------- Networks initialized -------------')
        networks.print_network(self.patn_model.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.patn_model.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.patn_model.netD_PP)
        networks.print_network(self.funit_model.gen)
        networks.print_network(self.funit_model.dis)
        print('-----------------------------------------------')

    
    def save(self, label):
        self.save_network(self.patn_model.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.patn_model.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.patn_model.netD_PP, 'netD_PP', label, self.gpu_ids)
        # funit
        self.save_network(self.funit_model.gen, 'net_G(funit)', label, self.gpu_ids)
        self.save_network(self.funit_model.dis, 'net_D(funit)', label, self.gpu_ids)
        self.save_network(self.funit_model.gen, 'net_G_scheduler(funit)', label, self.gpu_ids)
        self.save_network(self.funit_model.dis, 'net_D_scheduler(funit)', label, self.gpu_ids)
        
    
    def set_inputs(self, input):
        # TODO: change dataset structure, automatically load all ground_truth images of the same person
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]
        # TODO: (There has to be a variable of k shot)   
        # some of the variable contains only 1-shot...
        self.Ys = input['Ys'] 
        
        # check whether funit module accepts variable k-shot

        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()
            self.Ys = self.Ys.cuda()
        
    def forward(self):
        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.patn_fake_p2 = self.patn_model.netG(G_input)
        
        # TODO: append funit
    
    def backward_patn(self):
        pass
    
    def backward_funit(self):
        pass
        
    def train_one_batch(self):
        pass

    def get_current_visuals(self):
        pass
        
    def get_error_log(self):
        pass
        
    