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

    model = KeyNet(num_points = config.num_points, num_key = config.num_kp, num_cates = config.num_cates)
    model.cuda()

    if config.resume != '':
        model.load_state_dict(torch.load('{0}/{1}'.format(config.outf, config.resume)))

    # dataset init

    #dataset = Dataset('train', config.dataset_root, True, config.num_points, opt.num_cates, 5000, opt.category)
    dataset = hydra.utils.instantiate(config.dataset, set_name='train', num_points=config.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.workers)
    #test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, opt.num_cates, 1000, opt.category)
    test_dataset = hydra.utils.instantiate(config.dataset, set_name='val', num_points=config.num_points)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config.workers)

    criterion = Loss(config.num_kp, config.num_cates)

    best_test = np.Inf
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    train_count = 0
    for epoch in range(0, config.n_epochs):
        model.train()
        train_dis_avg = 0.0

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

            loss, _ = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate)
            loss.backward()

            train_dis_avg += loss.item()
            train_count += 1

            if train_count != 0 and train_count % config.log_every_n_samples == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(train_count, float(train_dis_avg) / config.log_every_n_samples)
                wandb.log({'train_loss': float(train_dis_avg) / config.log_every_n_samples}, step=train_count)
                train_dis_avg = 0.0

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

            _, item_score = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate)

            print(item_score)
            score.append(item_score)

        test_dis = np.mean(np.array(score))
        wandb.log({'val_loss': test_dis}, step=train_count)

if __name__ == "__main__":
    main()
