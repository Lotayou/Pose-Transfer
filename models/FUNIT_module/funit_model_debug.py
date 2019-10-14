"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

funit model without discriminator and gan loss!
"""

import copy

import torch
import torch.nn as nn

from .networks import FewShotGen
from .patch_discriminator import NLayerDiscriminator, GANLoss
from .vgg_loss_layer import VGGLoss


class FUNITModel(nn.Module):
    def __init__(self, hp, gpu_ids):
        super(FUNITModel, self).__init__()
        device = torch.device('cuda')
        self.gen = FewShotGen(hp['gen'])
        self.residual_weight = hp['res_w']
        self.gen_test = copy.deepcopy(self.gen)
        self.dis = NLayerDiscriminator(input_nc=3, n_layers=hp['n_patchgan_layers'], gpu_ids=gpu_ids, use_sigmoid=True)
        self.gen.train()
        self.vgg_criterion = VGGLoss().to(device)
        self.recon_criterion = nn.L1Loss().to(device)
        self.gan_criterion = GANLoss().to(device)
    
    @property
    def _pad_to_square(x):
        n,c,h,w = x.shape
        zeros = torch.zeros((n,c,h,(h-w)//2), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        return torch.cat((zeros, x, zeros), dim=3)
    
    def forward(self, fake_b, real_a, real_b, hp, mode):
        if mode == 'gen':
            self.gen.train()
            self.dis.eval()
            co_b = self.gen.enc_content(fake_b)
            s_a = self.gen.enc_class_model(real_a)
            #s_b = self.gen.enc_class_model(fake_b)
            res_b = self.gen.decode(co_b, s_a)  # translation
            
            # 20191010 It's weird, residual lies in [-1,1] and adding fake_b
            # could lead to overflow of out_b (up to 1.9)
            # Try manually tune down res_b
            
            out_b = res_b + fake_b
            dis_b_fake = self.dis(out_b)
            
            l_recon = self.recon_criterion(out_b, real_b) # + self.recon_criterion(out_a, real_a)
            l_perceptual = self.vgg_criterion(out_b, real_b) # + self.vgg_criterion(out_a, real_a)
            l_gan = self.gan_criterion(dis_b_fake, True)
            
            l_total = hp['r_w'] * l_recon + hp['perc_w'] * l_perceptual + hp['gan_w'] * l_gan
            l_total.backward()
            
            loss_dict = {
                'l_total': l_total.detach(),
                'l_recon': l_recon.detach(),
                'l_percp': l_perceptual.detach(),
                'l_gan': l_gan.detach(),
            }
            return res_b, out_b, loss_dict
        elif mode == 'dis':
            self.dis.train()
            self.gen.eval()
            with torch.no_grad():
                co_b = self.gen.enc_content(fake_b)
                s_a = self.gen.enc_class_model(real_a)
                res_b = self.gen.decode(co_b, s_a)  # translation
                out_b = res_b + fake_b
                
            dis_b_fake = self.dis(out_b)
            dis_b_real = self.dis(real_b)
            l_dis = (self.gan_criterion(dis_b_fake, False) + self.gan_criterion(dis_b_real, True)) / 2
            l_dis.backward()
            return l_dis.detach()
        else:
            assert 0, 'Mode dis or gen only'
    
    def test(self, fake_b, real_a):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        co_b = self.gen.enc_content(fake_b)
        s_a = self.gen.enc_class_model(real_a)
        
        r = self.gen.decode(co_b, s_a)  # translation
        out_b = r + fake_b
        # print('Residual range: [%.6f, %.6f]', r.max(), r.min())
        # print('Final out range: [%.6f, %.6f]', out.max(), out.min())
        
        return r, out_b

    def translate_k_shot(self, co_data, cl_data, k):
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            c_xa_current = self.gen_test.enc_content(xa)
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
