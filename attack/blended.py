'''
this script is for blended attack

@article{Blended,
	title        = {Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning},
	author       = {Xinyun Chen and Chang Liu and Bo Li and Kimberly Lu and Dawn Song},
	journal      = {arXiv preprint arXiv:1712.05526},
	year         = {2017}
}
'''
import argparse
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from attack.badnet import BadNet, add_common_attack_args


#Blended基本是复用的badnet
#attack_train_blended_alpha和attack_test_blended_alpha表示攻击的程度，多明显的融进去
class Blended(BadNet):

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--attack_trigger_img_path", type=str, )
        parser.add_argument("--attack_train_blended_alpha", type=float, )
        parser.add_argument("--attack_test_blended_alpha", type=float, )
        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/blended/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser


if __name__ == '__main__':
    attack = Blended()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    #attack.stage2_training()
    #使用mnist数据集的时候，要在deepnetwork_backdoor_attack_original/BackdoorBench/utils/bd_img_transform/blended.py 里将init逻辑改成mnist的
    # 和 deepnetwork_backdoor_attack_original/BackdoorBench/attack/prototype.py 里将yaml改成mnist
    #attack.savedataset('blended','gtsrb')
    #attack.saveTrainset('blended','cifar',0.2,True)
    attack.saveTestset('blended','cifar',0.5,True)