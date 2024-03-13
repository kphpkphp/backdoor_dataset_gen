'''
this script is for badnet attack

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. attack or use the model to do finetune with 5% clean data
7. save the attack result for defense

@article{gu2017badnets,
  title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  journal={arXiv preprint arXiv:1708.06733},
  year={2017}
}
'''

import os
import sys
import yaml

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
import numpy as np
import torch
import logging

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform




def add_common_attack_args(parser):
    parser.add_argument('--attack', type=str, )
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    return parser
#from utils.bd_label_transform.backdoor_label_transform import *

class BadNet(NormalCase):

    def __init__(self):
        super(BadNet).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)

        parser.add_argument("--patch_mask_path", type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/badnet/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def add_bd_yaml_to_args(self, args):
        with open(args.bd_yaml_path, 'r') as f:
            mix_defaults = yaml.safe_load(f)
        mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = mix_defaults

    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        #干净数据集、数据变换方法之类的都是通过benign_prepare方法产生的
        #支持的数据集包括mnist、cifar10、gtsrb
        #所有数据集的变换都是随机裁剪，cifar10还有一个随机水平旋转
        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()
            
        #print(len(test_dataset_without_transform))

        #这里是返回生成受污染集的一个变换过程
        #bd是backdoor
        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
        ### get the backdoor transform on label
        #返回的是一个对象，看起来这个对象是存储目标标签用的
        bd_label_transform = bd_attack_label_trans_generate(args)
        
        #print(bd_label_transform())
        
        #确实是all to one 攻击        
        # if isinstance(bd_label_transform, AllToOne_attack):
        #     print('is all to one')

        ### 4. set the backdoor attack data and backdoor test data
        #这个方法就真的只生成一个index数组而已
        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if 'pratio' in args.__dict__ else None,
            p_num=args.p_num if 'p_num' in args.__dict__ else None,
        )
        
        #train_poison_index是一个numpy.narray数组，长度是50000，正好是训练集的长度
        #print(train_poison_index.shape[0])

        #保存index
        logging.debug(f"poison train idx is saved")
        torch.save(train_poison_index,
                   args.save_path + '/train_poison_index_list.pickle',
                   )

        ### generate train dataset for backdoor attack
        # 生成后门攻击数据集
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            #原始数据集
            deepcopy(train_dataset_without_transform),
            #攻击标签
            poison_indicator=train_poison_index,
            #攻击方法
            bd_image_pre_transform=train_bd_img_transform,
            #标签的变换方法
            bd_label_pre_transform=bd_label_transform,
            #保存位置
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )

        #进行变换
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )

        ### decide which img to poison in ASR Test
        #确定在ASR测试中，使用哪些图片
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        ### generate test dataset for ASR
        # ASR:攻击成功率
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )
        
        #print(len(bd_test_dataset))

        #找到 test_poison_index 数组中值为 1 的元素的索引，并将这些索引保存在一个一维数组
        #[0]表示从np.where返回的元组中提取索引为0的元素
        #np.where在只有一个条件，并且条件成立时，返回的是所有符合条件的元素的坐标，通过元组的方式返回
        #subset就是中毒图片的列表,这个方法的意思是，将内部index设置成全部中毒的标签，返回时，只返回那些中毒的数据而不返回没中毒的
        #test_poison_index在前面设置过，在test数据集时，就是全部的非目标标签，本来就是目标标签的部分是不进行染毒的，因此，cifar10数据集只会返回原始不是目标标签的那些数据
        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        #进行数据变换，三个参数分别是原始数据集、图像变换规则、标签变换规则
        #
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            #这几个变换方法就是随机裁剪之类，对于cifar会做一个RandomHorizontalFlip，不清楚为何
            test_img_transform,
            #标签变换
            test_label_transform,
        )
        
        #标签没有任何变换        
        # if train_label_transform == None:
        #     print('train_label_transform is none')
        
        # if test_label_transform == None:
        #     print('test_label_transform is none')
        #这里面好像是原始的样本
        #print(clean_test_dataset_targets)
        #长度10000，应该就是原始的标签类别
        #print(len(clean_test_dataset_targets))
        
        #长度50000，独热编码，这是中毒标签的记录
        #print((bd_train_dataset.poison_indicator))
        
        
        #完成攻击与数据增强，保存
        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform

        #保存攻击后，没有经过数据增强的数据集，准备保存到磁盘
        self.bd_train_dataset = bd_train_dataset
        self.bd_test_dataset = bd_test_dataset
        # blend-cifar时，这里是9000，说明肯定是去掉了那些本来标签就是原始标签的部分
        #print(len(self.bd_test_dataset.bd_data_container))
        
        
    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = self.stage1_results

        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        if "," in args.device:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        trainer = BackdoorModelTrainer(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        from torch.utils.data.dataloader import DataLoader
        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='attack',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
        )


    def saveTrainset(self,type,dataset,radio,ifrandom):
        from torchvision.transforms import ToPILImage
        
        if (ifrandom):
            tag = 'random_trigger_trainSet'
        else:
            tag = 'trainSet'
        
        train_save_folder = f'/root/deepnetwork_backdoor_attack_original/BackdoorBench/dataset_collection/{type}/{dataset}/{tag}/{radio}'
        os.makedirs(train_save_folder,exist_ok=True)
        label_file = open(os.path.join(train_save_folder,'labels.txt'),'w')   
        poisoned_list_file = open(os.path.join(train_save_folder,'poisoned_list.txt'),'w')
        
        #True,说明返回的将是img、original_target、original_index、poison_or_not、label
        #print(self.bd_train_dataset.getitem_all)
        
        #读取图片并保存
        #这样读取，出来的格式是：
        #图片文件 当前标签 编号 是否中毒 原始标签
        for img,original_target,original_index,poison_or_not,origin_label in self.bd_train_dataset:  
            #img = ToPILImage()(img)
            image_path = os.path.join(train_save_folder, f'image_{original_index}.png')
            img.save(image_path)    
            pic_label = original_target
            label_file.write(f'image_{original_index}.png:{pic_label}\n')
            
            if  poison_or_not ==1:
                poisoned_list_file.write(f'image_{original_index}.png:{origin_label}\n')
                
        label_file.close()
        poisoned_list_file.close()
        
    def saveTestset(self,type,dataset,radio,ifrandom):
        from torchvision.transforms import ToPILImage
        # 测试集
        # 经测试，testset不会返回那些原始标签就是目标标签的数据
        
        if (ifrandom):
            tag = 'random_trigger_testSet'
        else:
            tag = 'testSet'
        
        test_save_folder = f'/root/deepnetwork_backdoor_attack_original/BackdoorBench/dataset_collection/{type}/{dataset}/{tag}/{radio}'
        os.makedirs(test_save_folder,exist_ok=True)
        label_file = open(os.path.join(test_save_folder,'labels.txt'),'w')   
        poisoned_list_file = open(os.path.join(test_save_folder,'poisoned_list.txt'),'w')
        
        for img,original_target,original_index,poison_or_not,origin_label in self.bd_test_dataset:  
            image_path = os.path.join(test_save_folder, f'image_{original_index}.png')
            img.save(image_path)    
            pic_label = original_target
            label_file.write(f'image_{original_index}.png:{pic_label}\n')
            
            if  poison_or_not ==1:
                poisoned_list_file.write(f'image_{original_index}.png:{origin_label}\n')
                
        label_file.close()
        poisoned_list_file.close()    
    
    #保存
    def savedataset(self,type,dataset):
        from torchvision.transforms import ToPILImage
        train_save_folder = f'/root/deepnetwork_backdoor_attack_original/BackdoorBench/dataset_collection/{type}/{dataset}/trainSet'
        os.makedirs(train_save_folder,exist_ok=True)
        label_file = open(os.path.join(train_save_folder,'labels.txt'),'w')   
        poisoned_list_file = open(os.path.join(train_save_folder,'poisoned_list.txt'),'w')
        
        #True,说明返回的将是img、original_target、original_index、poison_or_not、label
        #print(self.bd_train_dataset.getitem_all)
        
        #读取图片并保存
        #这样读取，出来的格式是：
        #图片文件 当前标签 编号 是否中毒 原始标签
        for img,original_target,original_index,poison_or_not,origin_label in self.bd_train_dataset:  
            #img = ToPILImage()(img)
            image_path = os.path.join(train_save_folder, f'image_{original_index}.png')
            img.save(image_path)    
            pic_label = original_target
            label_file.write(f'image_{original_index}.png:{pic_label}\n')
            
            if  poison_or_not ==1:
                poisoned_list_file.write(f'image_{original_index}.png:{origin_label}\n')
                
        label_file.close()
        poisoned_list_file.close()
        self.saveTestset(self,type,dataset)
        
        
        test_save_folder = f'/root/deepnetwork_backdoor_attack_original/BackdoorBench/dataset_collection/{type}/{dataset}/testSet'
        os.makedirs(test_save_folder,exist_ok=True)
        label_file = open(os.path.join(test_save_folder,'labels.txt'),'w')   
        poisoned_list_file = open(os.path.join(test_save_folder,'poisoned_list.txt'),'w')
        
        for img,original_target,original_index,poison_or_not,origin_label in self.bd_test_dataset:  
            image_path = os.path.join(test_save_folder, f'image_{original_index}.png')
            img.save(image_path)    
            pic_label = original_target
            label_file.write(f'image_{original_index}.png:{pic_label}\n')
            
            if  poison_or_not ==1:
                poisoned_list_file.write(f'image_{original_index}.png:{origin_label}\n')
                
        label_file.close()
        poisoned_list_file.close()   
        
        
        

if __name__ == '__main__':
    attack = BadNet()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    logging.debug("Be careful that we need to give the bd yaml higher priority. So, we put the add bd yaml first.")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    #attack.stage2_training()
    attack.savedataset('badnet')
