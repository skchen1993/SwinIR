#sync test
import os.path
import math
import argparse
import random
import numpy as np

import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from utils import utils_logger
from utils import utils_image as util
#----for set5 validation----
import train_validation_set5 as set5test
#---------------------------
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import matplotlib.pyplot as plt

'''
# --------------------------------------------
# training code for SwinIR
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main():
    print("----project SwinIR------")
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default="./options/swinir/train_swinir_sr_classical.json", help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    #----for set5 validation----
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
      #在此設定testing set路徑
    parser.add_argument('--folder_lq', type=str, default="/home/skchen/dataset/IPT_data/X2/benchmark/Set5/LR_bicubic/X2_swinIR/", help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default="/home/skchen/dataset/IPT_data/X2/benchmark/Set5/HR/", help='input ground-truth test image folder')
    #----improved model saving path----
    parser.add_argument('--model_save_dir', type=str, default="/home/skchen/ML_practice/KAIR/superresolution/swinir_sr_classical_patch48_x2/models/improved/", help='if model get performance improved, save model to this path')
    #----chart saving path-------------
    parser.add_argument('--chart_save_dir',type=str, default="/home/skchen/ML_practice/KAIR/set5test_results/chart/", help='path for chart saving')

    args = parser.parse_args()

    #parser.parse_args().opt = /.../train_swinir_sr_classical.json
    #option.py 下的 parse() function => 把json紀錄的資訊讀近來到opt
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    # SwinIR 中不會用到 E model
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E

    # 若有過去訓練過程中存的model以及 optimizer設定，會load進來繼續train
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        #用他的test part會有 shape unmatch 問題，故直接在後面採用其先前released 的 test code
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)


        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)


    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    # network and parameter information can be viewed
    #if opt['rank'] == 0:
        #logger.info(model.info_network())
        #logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    print("Start the main training")
    #"checkpoint_test": 5000           // for testing
    #"checkpoint_save": 5000           // for saving model
    #"checkpoint_print": 200           // for print
    #"checkpoint_chart": 5000           // for plotting chart
    # milestones": [250000, 400000, 450000, 475000, 500000] => half the lr

    # list for chart plotting
    iter = []
    lr_y=[]
    train_l1_y=[]
    set5valid_x = []
    set5valid_y = []

    psnr_y_record = 0

    for epoch in range(500000):  # keep running, is current_step that matter
        for i, train_data in enumerate(train_loader):

            current_step += 1

            if opt['rank'] == 0 and current_step % 100 == 0:
                print("Current step: ", current_step)
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            logs = model.current_log()  # such as loss
            iter.append(current_step)
            lr_y.append(model.current_learning_rate())
            train_l1_y.append(logs['G_loss'])
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)



            # -------------------------------
            # 5) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
            #if current_step % 50 == 0 :
                print(" ----set5 validation---")
                print("current_step: ", current_step)
                psnr_y  = set5test.validate_set5(args, model)
                set5valid_x.append(current_step)
                set5valid_y.append(psnr_y)
                if (psnr_y > psnr_y_record):
                    model.save_better_model(psnr_y, args, current_step )
                psnr_y_record = psnr_y

            # -------------------------------
            # 6) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
            #if current_step % 50 == 0:
                logger.info('Saving the model.')
                print("saving model")
                model.save(current_step)

        # chart plot
        if current_step % 5000 == 0 and opt['rank'] == 0:
            dir_chart = args.chart_save_dir
            info = "--------------chart plot=>   epoch: " + str(epoch) + ", current_step: " + str(current_step) + "-----------"
            logger.info(info)
            plt.figure(figsize=(10, 7))
            plt.plot(iter, lr_y, color='orange', label='lr_scheduler')
            plt.xlabel('current_step')
            plt.ylabel('lr')
            plt.legend()
            dir1 = os.path.join(dir_chart, "SwinIR_lr.png")
            plt.savefig(dir1)

            plt.figure(figsize=(10, 7))
            plt.plot(iter, train_l1_y, color='orange', label='l1_loss')
            plt.xlabel('current_step')
            plt.ylabel('l1_loss')
            plt.legend()
            dir2 = os.path.join(dir_chart, "SwinIR_l1loss.png")
            plt.savefig(dir2)

            plt.figure(figsize=(10, 7))
            plt.plot(set5valid_x, set5valid_y, color='orange', label='psnr_y')
            plt.xlabel('current_step')
            plt.ylabel('psnr')
            plt.legend()
            dir3 = os.path.join(dir_chart, "SwinIR_psnr.png")
            plt.savefig(dir3)


if __name__ == '__main__':
    main()
