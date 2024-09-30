# BEST MODEL Train script
import argparse
import gc
import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from dataloaders.dataset import *
from monai.losses import DiceCELoss
from networks.critic import Discriminator
from networks.net_factory import net_factory
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import ramps, losses, test_patch
from utils.losses import loss_diff1, loss_mask, loss_diff2, disc_loss, gen_loss

from dataloaders.lft_lisa import LISA

from utils.dice_brain import EDiceLoss


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LF_MRI', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='LoFiHippSeg_LISA', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=30000, help='maximum iteration to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--alpha', type=float, default=0.3, help='weight to balance generator masked loss')
parser.add_argument('--mu', type=float, default=0.01, help='weight to balance generator adversarial loss')
parser.add_argument('--t_m', type=float, default=0.2, help='mask threashold')
parser.add_argument('--ce_w', type=float, default=0.2, help='weight to balance ce loss')
parser.add_argument('--dl_w', type=float, default=1.0, help='weight to balance dice loss')

args = parser.parse_args()

snapshot_path = args.root_path + "/model/{}_{}_ce_{}_dl_{}_mu_{}_tm_{}_alpha_{}_bs_{}/{}".format(args.dataset_name, args.exp, args.ce_w, args.dl_w, args.mu, args.t_m, args.alpha, args.batch_size, args.model)
checkpoint_path = args.root_path + "/model/pretrained_checkpoint/{}".format(args.model)


num_classes = 3

patch_size = (128, 128, 128)
args.root_path = '/data/data_lisa/train'
args.save_path_1 = snapshot_path + '/validation_predictions_1'
args.save_path_2 = snapshot_path + '/validation_predictions_2'
train_data_path = args.root_path

labeled_bs = args.batch_size
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if not os.path.exists(args.save_path_1):
        os.makedirs(args.save_path_1)

    if not os.path.exists(args.save_path_2):
        os.makedirs(args.save_path_2)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model_1 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes-1, mode="train")
    model_1 = model_1.cuda()

    model_2 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes - 1, mode="train")
    model_2 = model_2.cuda()

    critic_1 = Discriminator()
    critic_1 = critic_1.cuda()

    save_mode_path = os.path.join(checkpoint_path, 'best_model_1.pth')
    model_1.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))

    save_mode_path2 = os.path.join(checkpoint_path, 'best_model_2.pth')
    model_2.load_state_dict(torch.load(save_mode_path2), strict=False)
    print("init weight from {}".format(save_mode_path2))

    db_train = LISA(base_dir=train_data_path, split='train', patch_size=patch_size)
    db_val = LISA(base_dir=train_data_path, split='val', patch_size=patch_size)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, shuffle=True, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn, batch_size=args.batch_size)
    valloader = DataLoader(db_val, batch_size=1, num_workers=12, pin_memory=True)

    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    dis_optimizer_1 = torch.optim.AdamW(critic_1.parameters(), lr=1e-4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice_1 = 0
    best_dice_2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    CE = torch.nn.BCELoss()
    iterator = tqdm(range(max_epoch), ncols=70)
    criterion = EDiceLoss().cuda()

    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=max_epoch)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=max_epoch)

    c_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer_1, T_max=max_epoch)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            torch.cuda.empty_cache()

            volume_batch, volume_batch_t, label_batch = sampled_batch['image'].float(), sampled_batch['image_t'].float(), sampled_batch['label'].float()
            volume_batch, volume_batch_t, label_batch = volume_batch.cuda(), volume_batch_t.cuda(), label_batch.cuda()

            model_1.train()
            model_2.train()

            prediction_batch_1 = torch.sigmoid(model_1(volume_batch))
            loss_sup_1 = criterion(prediction_batch_1, label_batch, ce_w=args.ce_w, dl_w=args.dl_w)

            prediction_batch_2 = torch.sigmoid(model_2(volume_batch_t))
            loss_sup_2 = criterion(prediction_batch_2, label_batch, ce_w=args.ce_w, dl_w=args.dl_w)

            iter_num = iter_num + 1

            critic_segs_1 = torch.sigmoid(critic_1(prediction_batch_2))
            masked_loss_1 = loss_mask(prediction_batch_1, prediction_batch_2, critic_segs_1, args.t_m)

            target_real_1 = torch.ones_like(label_batch)
            target_real_1.cuda()
            target_fake_1 = torch.zeros_like(label_batch)
            target_fake_1.cuda()

            g_critic_segs_1_1 = torch.sigmoid(critic_1(prediction_batch_1))
            g_critic_segs = torch.sigmoid(critic_1(label_batch.float()))
            target_real_g_1 = torch.ones_like(label_batch)
            target_real_g_1.cuda()
            loss_adversarial_gen_1 = gen_loss(g_critic_segs_1_1)
            loss_adversarial_1 = disc_loss(g_critic_segs_1_1, g_critic_segs, target_fake_1, target_real_1)

            loss_1 = loss_sup_1 + args.alpha * masked_loss_1 + args.mu * loss_adversarial_gen_1

            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()
            logging.info(
                'M1 iteration %d : loss : %03f, loss_sup: %03f, loss_mask: %03f, loss_adv: %03f, best_dice_1: %03f' % (
                    iter_num, loss_1, loss_sup_1, masked_loss_1,  loss_adversarial_1, best_dice_1))

            writer.add_scalar('loss1/loss_seg_dice', loss_sup_1, iter_num)

            critic_segs_2 = torch.sigmoid(critic_1(prediction_batch_1))
            masked_loss_2 = loss_mask(prediction_batch_2, prediction_batch_1, critic_segs_2, args.t_m)

            g_critic_segs_2_1 = torch.sigmoid(critic_1(prediction_batch_2))
            loss_adversarial_gen_2 = gen_loss(g_critic_segs_2_1)
            loss_adversarial_2 = disc_loss(g_critic_segs_2_1, g_critic_segs, target_fake_1, target_real_1)

            loss_2 = loss_sup_2 + args.alpha * masked_loss_2 + args.mu * loss_adversarial_gen_2

            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()
            logging.info(
                'M2 iteration %d : loss : %03f, loss_sup: %03f, loss_mask: %03f,  loss_adv: %03f, best_dice_2: %03f' % (
                    iter_num, loss_2, loss_sup_2, masked_loss_2, loss_adversarial_2, best_dice_2))

            writer.add_scalar('loss2/loss_seg_dice', loss_sup_2, iter_num)

            del loss_1, loss_2, loss_sup_2, loss_sup_1, masked_loss_1, g_critic_segs_1_1,g_critic_segs_2_1, loss_adversarial_gen_2, loss_adversarial_gen_1, target_real_1, target_fake_1
            torch.cuda.empty_cache()

            # Train Discriminator 1
            loss_adversarial_1 = loss_adversarial_1.clone().detach().requires_grad_(True)
            loss_adversarial_2 = loss_adversarial_2.clone().detach().requires_grad_(True)

            dis_optimizer_1.zero_grad()

            critic_loss_1 = loss_adversarial_1

            writer.add_scalar('loss/loss_critic1', critic_loss_1, iter_num)
            critic_loss_1.backward()
            dis_optimizer_1.step()

            if scheduler_1 is not None:
                scheduler_1.step()
            if scheduler_2 is not None:
                scheduler_2.step()

            if c_scheduler_1 is not None:
                c_scheduler_1.step()

            if iter_num >= 150 and iter_num % 150 == 0:
                model_1.eval()

                dice_sample_1 = test_patch.var_all_case_cotrain(model_1, valloader, patch_size=patch_size, save_result=True, test_save_path=args.save_path_1)

                if dice_sample_1 > best_dice_1:
                    best_dice_1 = dice_sample_1
                    save_best_path_m1 = os.path.join(snapshot_path, 'best_model_1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_best_path_m1)

                    save_best_pathc1 = os.path.join(snapshot_path, 'best_critic_1.pth'.format(args.model))
                    torch.save(critic_1.state_dict(), save_best_pathc1)
                    logging.info("save best model to {}".format(save_best_path_m1))

                writer.add_scalar('Var_dice1/Dice', dice_sample_1, iter_num)
                writer.add_scalar('Var_dice1/Best_dice', best_dice_1, iter_num)
                logging.info('M1 Best Dice :  %03f Current Dice  %03f' % (best_dice_1, dice_sample_1))
                model_2.eval()

                dice_sample_2 = test_patch.var_all_case_cotrain_t(model_2, valloader, patch_size=patch_size,
                                                                      save_result=True, test_save_path=args.save_path_2)

                if dice_sample_2 > best_dice_2:
                    best_dice_2 = dice_sample_2
                    save_best_path_m2 = os.path.join(snapshot_path, 'best_model_2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_best_path_m2)

                    save_best_pathc2 = os.path.join(snapshot_path, 'best_critic_2.pth'.format(args.model))
                    torch.save(critic_1.state_dict(), save_best_pathc2)
                    logging.info("save best model to {}".format(save_best_path_m2))

                writer.add_scalar('Var_dice2/Dice', dice_sample_2, iter_num)
                writer.add_scalar('Var_dice2/Best_dice', best_dice_2, iter_num)
                logging.info('M2 Best Dice :  %03f Current Dice  %03f' % (best_dice_2, dice_sample_2))
                torch.cuda.empty_cache()
            if iter_num >= max_iterations:
                save_mode_path_1 = os.path.join(snapshot_path, 'm1_iter_' + str(iter_num) + '.pth')
                torch.save(model_1.state_dict(), save_mode_path_1)
                logging.info("save model 1 to {}".format(save_mode_path_1))

                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()