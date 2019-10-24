import numpy as np
import os
import yaml
from collections import OrderedDict
import itertools
import util.util as util
from shutil import copyfile
import sys
import torch
from torch import nn

from .base_model import BaseModel
from . import networks
from .PATN import TransferModel
from .FUNIT_module.networks import FewShotGen
from .image_pool import MyImagePool
from .radam import RAdam
from .losses.alibaba_vgg_loss import WassFeatureLoss
from .losses.part_aware_recon_loss import PartAwareReconLoss


'''
20191018: A brand_new version of PatnFunitModel

@ Network Architecture:

                           PATN_G                                    BP2 -----| D_pose                       
            (P1, BP1, BP2) ------> patn_fake_p2 -------|  FUNIT               |-------> pose aligned?
                                                       |---------> final_p2 --| 
                        Ys = (y1,y2,...yk) ------------|                      |-------> real/fake?
                                                                    P1/P2-----| D_app
                                                                      
    
    PATN generator; FUNIT FewShotGen, appearance and pose conditional discriminators.
    
@ Training strategy (default 2.A):
    For each batch:
        1. Perform forward computation
        2.A: If PATN is randomly initialized:
            - Update PATN network for one-step
            - Update PATN and FUNIT generator for one-step
        2.B: If PATN is pretrained on DeepFashion:
            - Only Update FUNIT generator
            
        3. Perform random swapping with imagepool buffer
        4. Update discriminator for opt.DG_ratio steps 

@ Author: Lingbo Yang
'''

class PatnFunitModel(BaseModel):
    def name(self):
        return 'PATH_FUNIT_FULL Model'
    
    ##########################
    ### Pre-training steps
    ##########################
    def initialize(self, opt):
        super(PatnFunitModel, self).initialize(opt)
        with open(opt.funit_options, 'r') as fin:
            self.funit_opt = yaml.load(fin, Loader=yaml.FullLoader)
        
        copyfile(opt.funit_options,
            os.path.join(self.save_dir, 'funit_options.yaml'))
        
        self.pool = MyImagePool(opt.pool_size)
        
        # Initialize networks
        input_nc = [opt.P_input_nc, opt.BP_input_nc * 2]
        self.G1 = networks.define_G(input_nc, opt.P_input_nc,
                    opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                    n_downsampling=opt.G_n_downsampling)
                    
        self.G2 = FewShotGen(self.funit_opt['gen'])
        self.nets = [self.G1, self.G2]
        
        if opt.isTrain:
            pp_nc = opt.P_input_nc * 2
            pb_nc = opt.P_input_nc + opt.BP_input_nc
            use_sigmoid = opt.no_lsgan
            self.D_PB = networks.define_D(pb_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)
            self.D_PP = networks.define_D(pp_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)
            self.nets += [self.D_PB, self.D_PP]
        
        # Load networks
        if not opt.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.G1, 'G1', which_epoch)
            self.load_network(self.G2, 'G2', which_epoch)
            if opt.isTrain:
                self.load_network(self.D_PB, 'D_PB', which_epoch)
                self.load_network(self.D_PP, 'D_PP', which_epoch)
                
        
        if opt.isTrain:
            # prepare image pool
            self.PP_pool = MyImagePool(opt.pool_size)
            self.PB_pool = MyImagePool(opt.pool_size)
            # set losses
            # self.recon_loss = nn.L1Loss()
            self.recon_loss = PartAwareReconLoss(loss_type='l1')
            self.perceptual_loss = WassFeatureLoss()
            self.gan_loss = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            
            # set optimizers
            self.opt_G1 = RAdam(self.G1.parameters(), lr=opt.G1_lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.opt_G2 = RAdam(self.G2.parameters(), lr=opt.G2_lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.opt_D_PB = RAdam(self.D_PB.parameters(), lr=opt.D_lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.opt_D_PP = RAdam(self.D_PP.parameters(), lr=opt.D_lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            
            # set schedulers
            self.optimizers = [self.opt_G1, self.opt_G2, self.opt_D_PB, self.opt_D_PP]
            self.schedulers = []
            for _optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(_optimizer, opt))
                
        # print info
        print('------ Network initialized and ready to go ------')
        for net in self.nets:
            networks.print_network(net)
            net = net.cuda()
        
        # Only FUNIT G2 need to be wrapped, other networks have inner wrapper
        if len(self.gpu_ids) > 1:
            self.G2 = nn.DataParallel(self.G2, device_ids=self.gpu_ids)
    
    def set_input(self, input):
        # TODO: change dataset structure, automatically load all ground_truth images of the same person
        #   update: loading Ys for testing and training case B only.
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]
        self.input_label = input['label']
        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()
            self.input_label = torch.as_tensor(self.input_label, dtype=torch.long).cuda()
        
        if 'Ys' in input:
            self.Ys = input['Ys'].cuda() if len(self.gpu_ids) > 0 else input['Ys']
            
        if 'PLW1' in input:
            self.PLW1 = input['PLW1'].cuda()
            self.PLW2 = input['PLW2'].cuda()
            
    #################################
    ###   Training Functions
    #################################
    
    def G1_forward(self):
        self.G1.train()
        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.output_G1 = self.G1(G_input)
            
    def G2_forward(self):
        # rb = G2(D(E_co(B1), E_cl(PA))
        self.G1.train()
        self.G2.train()
        # perform G1 forward again, since G1 has been updated once and graph is freed
        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.output_G1 = self.G1(G_input)
        self.res = self.G2(self.output_G1, self.input_P1)
        self.output_G2 = self.output_G1 + self.res
    
    def update_G1(self):
        self.opt_G1.zero_grad()
        # backward l1+vgg loss
        #l1_loss = self.recon_loss(self.output_G1, self.input_P2)
        l1_loss = self.recon_loss(self.output_G1, self.input_P2, self.PLW2)
        vgg_loss = self.perceptual_loss(self.output_G1, self.input_P2)
        total_loss = self.opt.lambda_vgg * vgg_loss + self.opt.lambda_l1 * l1_loss
        
        total_loss.backward()   # save for the next update of G2
        self.opt_G1.step()
        
        # self.G1_loss_dict = {
            # 'total': total_loss.detach(),
            # 'l1': l1_loss.detach(),
            # 'vgg': vgg_loss.detach(),
        # }
        
    def update_G1_and_G2(self):
        self.opt_G1.zero_grad()
        self.opt_G2.zero_grad()
        # backward combined loss
        #l1_loss = self.recon_loss(self.output_G2, self.input_P2)
        l1_loss = self.recon_loss(self.output_G1, self.input_P2, self.PLW2)
        vgg_loss = self.perceptual_loss(self.output_G2, self.input_P2)
        
        pp_pack = torch.cat((self.output_G2, self.input_P1), dim=1)
        pb_pack = torch.cat((self.output_G2, self.input_BP2), dim=1)
        fake_pp_feat = self.D_PP(pp_pack)
        fake_pb_feat = self.D_PB(pb_pack)
        gan_pp_loss = self.gan_loss(fake_pp_feat, True)
        gan_pb_loss = self.gan_loss(fake_pb_feat, True)
        gan_loss = (gan_pp_loss + gan_pb_loss) / 2
        total_loss = self.opt.lambda_l1 * l1_loss + self.opt.lambda_vgg * vgg_loss + self.opt.lambda_gan * gan_loss
        
        total_loss.backward()
        self.opt_G1.step()
        self.opt_G2.step()
        
        self.G2_loss_dict = {
            'total': total_loss.detach(),
            'l1': l1_loss.detach(),
            'vgg': vgg_loss.detach(),
            'gan': gan_loss.detach(),
        }

    def update_D(self):
        # update pose discriminator
        self.opt_D_PB.zero_grad()
        real_pb_pack = torch.cat((self.input_P2, self.input_BP2), dim=1)
        fake_pb_pack = torch.cat((self.output_G2, self.input_BP2), dim=1)
        fake_pb_pack = self.PB_pool.query(fake_pb_pack)
        
        real_pb_feat = self.D_PB(real_pb_pack)
        fake_pb_feat = self.D_PB(fake_pb_pack.detach())
        real_pb_loss = self.gan_loss(real_pb_feat, True)
        fake_pb_loss = self.gan_loss(fake_pb_feat, False)
        
        pb_loss = (real_pb_loss + fake_pb_loss) * self.opt.lambda_gan / 2
        pb_loss.backward()
        self.opt_D_PB.step()
        
        # update appearance discriminator
        self.opt_D_PP.zero_grad()
        real_pp_pack = torch.cat((self.input_P2, self.input_P1), dim=1)
        fake_pp_pack = torch.cat((self.output_G2, self.input_P1), dim=1)
        fake_pp_pack = self.PP_pool.query(fake_pp_pack)
        
        real_pp_feat = self.D_PP(real_pp_pack)
        fake_pp_feat = self.D_PP(fake_pp_pack.detach())
        real_pp_loss = self.gan_loss(real_pp_feat, True)
        fake_pp_loss = self.gan_loss(fake_pp_feat, False)
        
        pp_loss = (real_pp_loss + fake_pp_loss) * self.opt.lambda_gan / 2
        pp_loss.backward()
        self.opt_D_PP.step()
        
        self.D_loss_dict = {
            'pb': pb_loss.detach(), 
            'pp': pp_loss.detach()
        }
    
    def train_one_step(self):
        self.G1_forward()
        self.update_G1()
        self.G2_forward()
        self.update_G1_and_G2()
        for i in range(self.opt.DG_ratio):
            self.update_D()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    def test(self):
        G_input = [self.input_P1, torch.cat((self.input_BP1, self.input_BP2), 1)]
        with torch.no_grad():
            self.output_G1 = self.G1(G_input)
            self.res = self.G2(self.output_G1, self.input_P1)
            self.output_G2 = self.output_G1 + self.res
        
    #############################
    ###  Utility Functions
    #############################

    def get_current_visuals(self):
        # 2 by 3:
        # A, PA, B'
        # B, r, B''
        P1 = util.tensor2im(self.input_P1.data)
        P2 = util.tensor2im(self.input_P2.data)
        O1 = util.tensor2im(self.output_G1.data)
        O2 = util.tensor2im(self.output_G2.data)
        r = util.tensor2im(self.res.data)
        BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        
        visual_tensor = np.concatenate((
            np.concatenate((P1, BP1, O1), axis=1),
            np.concatenate((P2, r, O2), axis=1),
        ), axis=0)
        return visual_tensor
        
    def get_error_log(self, iter_num):
        log = 'Iter %d:\n\t Dis loss: ' % iter_num
        for k, v in self.D_loss_dict.items():
            log += '%s: %.4f, ' % (k, torch.sum(v).item() / self.opt.batchSize)
        log += '\n\t Gen loss: '
        for k, v in self.G2_loss_dict.items():
            log += '%s: %.4f, ' % (k, torch.sum(v).item() / self.opt.batchSize)
        return log
        
    def save(self, label):
        self.save_network(self.G1, 'G1', label, self.gpu_ids)
        self.save_network(self.G2, 'G2', label, self.gpu_ids)  # Could be a bug, self.G2.module?
        self.save_network(self.D_PB, 'D_PB', label, self.gpu_ids)
        self.save_network(self.D_PP, 'D_PP', label, self.gpu_ids)
        
