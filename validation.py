import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch import test_all_case
from dataloaders.dataset import *
from dataloaders.lft_lisa import LISA
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='LF_MRI', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='LoFiHippSeg_LISA', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=30000, help='maximum iteration to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--alpha', type=float, default=0.2, help='weight to balance generator masked loss')
parser.add_argument('--mu', type=float, default=0.01, help='weight to balance generator adversarial loss')
parser.add_argument('--t_m', type=float, default=0.1, help='mask threashold')
parser.add_argument('--ce_w', type=float, default=0.2, help='weight to balance ce loss')
parser.add_argument('--dl_w', type=float, default=1.0, help='weight to balance dice loss')

parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--detail', type=int,  default=0, help='print metrics for every samples?')

args = parser.parse_args()

snapshot_path = args.root_path + "/model/{}_{}_ce_{}_dl_{}_mu_{}_tm_{}_alpha_{}_bs_{}/{}".format(args.dataset_name, args.exp, args.ce_w, args.dl_w, args.mu, args.t_m, args.alpha, args.batch_size, args.model)

test_save_path = args.root_path + "/model/{}_{}_ce_{}_dl_{}_mu_{}_tm_{}_alpha_{}_bs_{}/{}/vnet_predictions_ensemble/".format(args.dataset_name, args.exp,  args.ce_w, args.dl_w, args.mu, args.t_m, args.alpha, args.batch_size, args.model)

num_classes = 3

patch_size = (128, 128, 128)
args.root_path = '/data/data_lisa/validation'

train_data_path = args.root_path

db_test = LISA(base_dir=train_data_path, split='test', patch_size=patch_size)
testloader = DataLoader(db_test, batch_size=1, num_workers=12, pin_memory=True, shuffle=False)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)


def calculate_metric():
    net1 = net_factory(net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
    save_mode_path = os.path.join(snapshot_path, 'best_model_1.pth')
    net1.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net1.eval()

    net2 = net_factory(net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
    save_mode_path2 = os.path.join(snapshot_path, 'best_model_2.pth')
    net2.load_state_dict(torch.load(save_mode_path2), strict=False)
    print("init weight from {}".format(save_mode_path2))
    net2.eval()

    avg_metric = test_all_case(net1, net2, testloader, patch_size=patch_size, save_result=True, test_save_path=test_save_path, th=0.5)

    return avg_metric


if __name__ == '__main__':
    metric = calculate_metric()
    print(metric)
