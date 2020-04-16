import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src'))

# take leading -- off of --local_rank=
for i, arg in enumerate(sys.argv):
    if arg[0:2] == '--':
        sys.argv[i] = arg[2:]

import pdb
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
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from dataset.dataset_nocs import Dataset
from libs.network import KeyNet
from libs.loss import Loss
from utils.utils import initialize_distributed, strip_ddp_state_dict

cate_list = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

@hydra.main(config_path='../conf/train/config.yaml')
def main(config):

    # parse ddp related args
    config.rank = 0
    if config.use_ddp:
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.rank = int(os.environ['RANK'])
        config.local_rank = int(os.environ['LOCAL_RANK'])
    print(f'world_size: {config.world_size}, rank: {config.rank}, local_rank: {config.local_rank}')

    if config.world_size > 1:
        initialize_distributed(config)
        print(f'initialized distributed rank: {config.rank}')


    if config.rank == 0:
        print('initializing wandb...')
        wandb.init(project="6pack", resume=True, name=config.slurm.job_name[0])
        print(f'initialized wandb, resume={wandb.run.resumed}')

    #if config.rank == 0 and wandb.run.resumed:
    #if wandb.run.resumed:
    chkpt_path = os.path.join(config.outf, 'model_latest.pth')
    if os.path.exists(chkpt_path):
        config.resume_ckpt = chkpt_path
        config.resume_ckpt_opt_state = os.path.join(config.outf, 'optim_state_latest.pth')


    model = KeyNet(num_points = config.num_points, num_key = config.num_kp, num_cates = config.num_cates)
    criterion = Loss(config.num_kp, config.num_cates, config.loss_term_weights)

    if config.resume_ckpt != '':
        print(f'rank: {config.rank}, resuming model from {config.resume_ckpt}')
        model.load_state_dict(torch.load(config.resume_ckpt, map_location=lambda storage, loc: storage))

    model.cuda(torch.cuda.current_device())

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    if config.resume_ckpt_opt_state != '':
        print(f'rank: {config.rank}, resuming optim state from {config.resume_ckpt_opt_state}')
        optimizer_state_dict = torch.load(config.resume_ckpt_opt_state, map_location=lambda storage, loc: storage)
        loaded_groups = set(optimizer_state_dict['param_groups'][0]['params'])
        groups = set(optimizer.state_dict()['param_groups'][0]['params'])
        optimizer.load_state_dict(optimizer_state_dict)

    if config.use_ddp:
        model = DDP(model, find_unused_parameters=True, device_ids=[config.rank])
        #criterion = DDP(criterion, device_ids=[config.local_rank], output_device=[config.local_rank])


    # dataset init
    #dataset = Dataset('train', config.dataset_root, True, config.num_points, opt.num_cates, 5000, opt.category)
    dataset = hydra.utils.instantiate(config.dataset, set_name='train', num_points=config.num_points)
    #test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, opt.num_cates, 1000, opt.category)
    test_dataset = hydra.utils.instantiate(config.dataset, set_name='val', num_points=config.num_points)

    # distributed sampler if using ddp
    train_sampler = None
    test_sampler = None
    if config.use_ddp:
        train_sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=config.workers, sampler=train_sampler)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.workers, sampler=test_sampler)

    start_count = 0
    start_epoch = 0
    if config.resume_ckpt_opt_state != '':
        optimizer_state_dict = torch.load(config.resume_ckpt_opt_state, map_location=lambda storage, loc: storage)
        optim_state_step = optimizer_state_dict['state'][list(optimizer_state_dict['state'].keys())[0]]['step']
        start_count = optim_state_step * config.batch_size
        start_epoch = optim_state_step * config.batch_size // len(dataloader)
        print(f'rank: {config.rank}, resuming from step {start_count} and epoch {start_epoch}')


    train_count = start_count
    train_dis_avg = 0.0
    train_losses_dict_avg = {}
    for epoch in range(start_epoch, config.n_epochs):
        if config.use_ddp:
            train_sampler.set_epoch(epoch)
        model.train()

        optimizer.zero_grad()

        for i, data in enumerate(dataloader, 0):
            # print some params to verify ddp is working
            #s = ''
            #for j in range(20):
                #p = list(model.parameters())[j]
                #s += f' {torch.flatten(p)[50].item():.10f}'
            #s += f' {p.device}'
            #print(s)

            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, faces, anchor, scale, cate = data
            anchor = torch.Tensor([[[0.,0.,0.]]]).to(img_fr.device)
            #print(f'img_fr.shape: {img_fr.shape}, choose_fr.shape: {choose_fr.shape}, cloud_fr.shape: {cloud_fr.shape}, r_fr.shape: {r_fr.shape}, t_fr.shape: {t_fr.shape}, mesh.shape: {mesh.shape}, anchor.shape: {anchor.shape}, scale.shape: {scale.shape}')
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, faces, anchor, scale, cate = Variable(img_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(choose_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(cloud_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(r_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(t_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(img_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(choose_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(cloud_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(r_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(t_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(mesh).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(faces).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(anchor).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(scale).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(cate).cuda(torch.cuda.current_device())

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

            loss, _, losses_dict = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, faces, scale, cate)
            for k, v in losses_dict.items():
                if k not in train_losses_dict_avg:
                    train_losses_dict_avg[k] = 0.0
                train_losses_dict_avg[k] += v.cpu().detach()
            loss.backward()

            train_dis_avg += loss.item()
            train_count += 1

            if train_count != 0 and train_count % config.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            if train_count != 0 and train_count % config.log_every_n_samples == 0:
                print(train_count, float(train_dis_avg) / config.log_every_n_samples)
                log_dict = {}
                for k, v in train_losses_dict_avg.items():
                    log_dict['train_'+k] = v / config.log_every_n_samples

                if config.local_rank == 0:
                    wandb.log(log_dict, step=train_count*config.world_size)
                train_dis_avg = 0.0
                train_losses_dict_avg = {}

            if config.local_rank == 0 and  train_count != 0 and train_count % config.checkpoint_every_n_samples == 0:
                #fname = os.path.join(config.outf, 'model_at_step_{0}.pth'.format(train_count * config.world_size))
                state_dict = model.state_dict()
                if config.use_ddp:
                    state_dict = strip_ddp_state_dict(state_dict)
                #torch.save(state_dict, fname)
                # save to tmp location then mv, incase we are pre-empted while writing
                fname_tmp = os.path.join(config.outf, 'model_latest.pth.tmp')
                fname = os.path.join(config.outf, 'model_latest.pth')
                torch.save(state_dict, fname_tmp)
                os.system(f'mv {fname_tmp} {fname}')
                fname_wandb = os.path.join(config.outf, 'model_latest_wandb.pth')
                torch.save(state_dict, fname_wandb)
                wandb.save(fname_wandb)
                fname_tmp = os.path.join(config.outf, 'optim_state_latest.pth.tmp')
                fname = os.path.join(config.outf, 'optim_state_latest.pth')
                torch.save(optimizer.state_dict(), fname_tmp)
                os.system(f'mv {fname_tmp} {fname}')

        optimizer.zero_grad()
        model.eval()
        score = []
        val_losses_dict_avg = {}
        for j, data in enumerate(testdataloader, 0):
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, faces, anchor, scale, cate = data
            anchor = torch.Tensor([[[0.,0.,0.]]]).to(img_fr.device)
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, faces, anchor, scale, cate = Variable(img_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(choose_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(cloud_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(r_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(t_fr).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(img_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(choose_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(cloud_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(r_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(t_to).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(mesh).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(faces).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(anchor).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(scale).cuda(torch.cuda.current_device()), \
                                                                                                                         Variable(cate).cuda(torch.cuda.current_device())

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

            _, item_score, losses_dict = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, faces, scale, cate)
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
        if config.local_rank == 0:
            wandb.log(log_dict, step=train_count*config.world_size)
        val_losses_dict_avg = {}


if __name__ == "__main__":
    main()
