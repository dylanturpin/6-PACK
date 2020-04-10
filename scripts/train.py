import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import wandb
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import hydra
from dataset.dataset_nocs import Dataset
from libs.network import KeyNet
from libs.loss import Loss

cate_list = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

@hydra.main(config_path='../conf/train/config.yaml')
def main(config):
    wandb.init(project="6pack", config=config, resume=True)

    if wandb.run.resumed:
        print(f'resuming! at step: {wandb.run.step}')
        config.resume_ckpt = os.path.join(config.outf, 'model_latest.pth')
        config.resume_ckpt_opt_state = os.path.join(config.outf, 'optim_state_latest.pth')

    model = KeyNet(num_points = config.num_points, num_key = config.num_kp, num_cates = config.num_cates)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if config.resume_ckpt != '':
        model.load_state_dict(torch.load(config.resume_ckpt))
        optimizer_state_dict = torch.load(config.resume_ckpt_opt_state)
        optimizer.load_state_dict(optimizer_state_dict)

    # dataset init

    #dataset = Dataset('train', config.dataset_root, True, config.num_points, opt.num_cates, 5000, opt.category)
    dataset = hydra.utils.instantiate(config.dataset, set_name='train', num_points=config.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.workers)
    #test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, opt.num_cates, 1000, opt.category)
    test_dataset = hydra.utils.instantiate(config.dataset, set_name='val', num_points=config.num_points)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config.workers)

    criterion = Loss(config.num_kp, config.num_cates, config.loss_term_weights)

    start_count = 0
    start_epoch = 0
    if wandb.run.resumed:
        start_count = wandb.run.step
        start_epoch = wandb.run.step // len(dataloader)

    train_count = start_count
    for epoch in range(start_epoch, config.n_epochs):
        model.train()
        train_dis_avg = 0.0
        train_losses_dict_avg = {}

        optimizer.zero_grad()

        for i, data in enumerate(dataloader, 0):
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = data
            anchor = torch.Tensor([[[0.,0.,0.]]]).to(img_fr.device)
            #print(f'img_fr.shape: {img_fr.shape}, choose_fr.shape: {choose_fr.shape}, cloud_fr.shape: {cloud_fr.shape}, r_fr.shape: {r_fr.shape}, t_fr.shape: {t_fr.shape}, mesh.shape: {mesh.shape}, anchor.shape: {anchor.shape}, scale.shape: {scale.shape}')
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = Variable(img_fr).cuda(), \
                                                                                                                         Variable(choose_fr).cuda(), \
                                                                                                                         Variable(cloud_fr).cuda(), \
                                                                                                                         Variable(r_fr).cuda(), \
                                                                                                                         Variable(t_fr).cuda(), \
                                                                                                                         Variable(img_to).cuda(), \
                                                                                                                         Variable(choose_to).cuda(), \
                                                                                                                         Variable(cloud_to).cuda(), \
                                                                                                                         Variable(r_to).cuda(), \
                                                                                                                         Variable(t_to).cuda(), \
                                                                                                                         Variable(mesh).cuda(), \
                                                                                                                         Variable(anchor).cuda(), \
                                                                                                                         Variable(scale).cuda(), \
                                                                                                                         Variable(cate).cuda()

            Kp_fr, anc_fr, att_fr = model(img_fr, choose_fr, cloud_fr, anchor, scale, cate, t_fr)
            Kp_to, anc_to, att_to = model(img_to, choose_to, cloud_to, anchor, scale, cate, t_to)

            Kp_fr *= config.scale_loss_inputs_by
            Kp_to *= config.scale_loss_inputs_by
            anc_fr *= config.scale_loss_inputs_by
            anc_to *= config.scale_loss_inputs_by
            t_fr *= config.scale_loss_inputs_by
            t_to *= config.scale_loss_inputs_by
            mesh *= config.scale_loss_inputs_by
            scale *= config.scale_loss_inputs_by

            loss, _, losses_dict = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate)
            for k, v in losses_dict.items():
                if k not in train_losses_dict_avg:
                    train_losses_dict_avg[k] = 0.0
                train_losses_dict_avg[k] += v.cpu().detach()
            loss.backward()

            train_dis_avg += loss.item()
            train_count += 1

            if train_count != 0 and train_count % config.log_every_n_samples == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(train_count, float(train_dis_avg) / config.log_every_n_samples)
                log_dict = {}
                for k, v in train_losses_dict_avg.items():
                    log_dict['train_'+k] = v / config.log_every_n_samples
                wandb.log(log_dict, step=train_count)
                train_dis_avg = 0.0
                train_losses_dict_avg = {}

            if train_count != 0 and train_count % config.checkpoint_every_n_samples == 0:
                fname = os.path.join(config.outf, 'model_at_step_{0}.pth'.format(train_count))
                torch.save(model.state_dict(), fname)
                wandb.save(fname)
                fname = os.path.join(config.outf, 'model_latest.pth')
                torch.save(model.state_dict(), fname)
                fname = os.path.join(config.outf, 'optim_state_latest.pth')
                torch.save(optimizer.state_dict(), fname)

        optimizer.zero_grad()
        model.eval()
        score = []
        val_losses_dict_avg = {}
        for j, data in enumerate(testdataloader, 0):
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = data
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = Variable(img_fr).cuda(), \
                                                                                                                         Variable(choose_fr).cuda(), \
                                                                                                                         Variable(cloud_fr).cuda(), \
                                                                                                                         Variable(r_fr).cuda(), \
                                                                                                                         Variable(t_fr).cuda(), \
                                                                                                                         Variable(img_to).cuda(), \
                                                                                                                         Variable(choose_to).cuda(), \
                                                                                                                         Variable(cloud_to).cuda(), \
                                                                                                                         Variable(r_to).cuda(), \
                                                                                                                         Variable(t_to).cuda(), \
                                                                                                                         Variable(mesh).cuda(), \
                                                                                                                         Variable(anchor).cuda(), \
                                                                                                                         Variable(scale).cuda(), \
                                                                                                                         Variable(cate).cuda()

            Kp_fr, anc_fr, att_fr = model(img_fr, choose_fr, cloud_fr, anchor, scale, cate, t_fr)
            Kp_to, anc_to, att_to = model(img_to, choose_to, cloud_to, anchor, scale, cate, t_to)
            Kp_fr *= config.scale_loss_inputs_by
            Kp_to *= config.scale_loss_inputs_by
            anc_fr *= config.scale_loss_inputs_by
            anc_to *= config.scale_loss_inputs_by
            t_fr *= config.scale_loss_inputs_by
            t_to *= config.scale_loss_inputs_by
            mesh *= config.scale_loss_inputs_by
            scale *= config.scale_loss_inputs_by

            _, item_score, losses_dict = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate)
            for k, v in losses_dict.items():
                if k not in val_losses_dict_avg:
                    val_losses_dict_avg[k] = 0.0
                val_losses_dict_avg[k] += v.cpu().detach()


            print(item_score)
            score.append(item_score)

        test_dis = np.mean(np.array(score))
        log_dict = {}
        for k, v in val_losses_dict_avg.items():
            log_dict['val_'+k] = v / config.log_every_n_samples
        wandb.log(log_dict, step=train_count)
        val_losses_dict_avg = {}


if __name__ == "__main__":
    main()
