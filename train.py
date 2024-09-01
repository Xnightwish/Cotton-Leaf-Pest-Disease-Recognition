import os
import math
import argparse
import time
import inspect

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np


# from datasets.my_dataset import MyDataSet
from datasets.pest_dataset import PestDataset
# 对比性实验-第一批*3
# from models.model_comparative.resnet import resnet50 as create_model
# from models.model_comparative.efficientnetv2 import efficientnetv2_s as create_model
# from models.model_comparative.convnext import convnext_tiny as create_model
# 对比性实验-第二批*3
# from models.model_comparative.mobilevitv2 import mobile_vit_xx_small as create_model
# from models.model_comparative.swin_transformer import swin_tiny_patch4_window7_224 as create_model
# from models.model_comparative.vit_model import vit_base_patch16_224_in21k as create_model
# 有效性实验*3
# from models.model_effective.vit_1_01_patch import vit_base_patch16_224_in21k as create_model
from models.model_effective.vit_1_02_v12e1 import vit_base_patch16_224_in21k as create_model
# 最优性实验-第一批*4
# from models.model_optimal.vit_2_01_v12e3 import vit_base_patch16_224_in21k as create_model
# from models.model_optimal.vit_2_02_v12e6 import vit_base_patch16_224_in21k as create_model
# from models.model_optimal.vit_2_03_v12e9 import vit_base_patch16_224_in21k as create_model
# from models.model_optimal.vit_2_04_v12e12 import vit_base_patch16_224_in21k as create_model
# 最优性实验-第二批*4
# from models.model_optimal.vit_3_01_patch1 import vit_base_patch16_224_in21k as create_model
# from models.model_optimal.vit_3_02_patch2 import vit_base_patch16_224_in21k as create_model
# from models.model_optimal.vit_3_03_patch4 import vit_base_patch16_224_in21k as create_model
# from models.model_optimal.vit_3_04_patch8 import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])
    data_dir=args.data_path
    train_dataset = PestDataset(txt_file=os.path.join(data_dir, 'train.txt'), root_dir=os.path.join(data_dir, 'train'), transform=data_transform["train"])
    

    # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])
    val_dataset = PestDataset(txt_file=os.path.join(data_dir, 'val.txt'), root_dir=os.path.join(data_dir, 'val'), transform=data_transform["val"])

    batch_size = args.batch_size
    max_num_workers = args.num_workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, max_num_workers])  # number of workers
    # nw = min([os.cpu_count(), max_num_workers])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    # model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    model = create_model(num_classes=args.num_classes).to(device)
    # 将模型写入tensorboard
    # init_img = torch.zeros((1, 3, 224, 224), device=device)
    # tb_writer.add_graph(model, init_img)


    """
    实验0:baseline,无变化
    实验1:增强patch,去除patch
    实验2:交替EMA,无变化
    实验3:权重*9+展平+EMA*3,只添加9权重
    实验4:权重*11+展平+EMA*1,只添加1权重
    实验5:.
    """
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        for k in list(weights_dict.keys()):
            if "head" in k or "classifier" in k: # vit,swin-vit,convnext,convnext,mobilevit
                del weights_dict[k]
            if "resnet" in inspect.getfile(create_model) and "fc" in k: # resnet
                del weights_dict[k]
            if "patch" in inspect.getfile(create_model) and "patch_embed.proj" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))



    
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.
    total_time=0
    for epoch in range(args.epochs):
        start_time = time.time()

        # train
        train_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        val_acc, val_precision, val_recall, val_f1, true_label_list, pred_label_list = evaluate(model=model,
                                                            data_loader=val_loader,
                                                            device=device,
                                                            epoch=epoch)

        tags = ["train_loss",
                "val_acc", "val_precision", "val_recall", "val_f1", 
                "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[1], val_acc, epoch)
        tb_writer.add_scalar(tags[2], val_precision, epoch)
        tb_writer.add_scalar(tags[3], val_recall, epoch)
        tb_writer.add_scalar(tags[4], val_f1, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc
            with open('./weights/y_test.txt', 'w') as file:
                for arr in true_label_list:
                    for num in arr:
                        file.write(str(num) + '\n')
            with open('./weights/y_pred.txt', 'w') as file:
                for arr in pred_label_list:
                    for num in arr:
                        file.write(str(num) + '\n')
        
        end_time=time.time()
        epoch_time=end_time-start_time
        total_time+=epoch_time
    
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = "{:02}:{:02}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print(f"Total training time: {formatted_duration} (hh:mm:ss.ss)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=13)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=17)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")
    # parser.add_argument('--data-path', type=str,
    #                     default=r"/root/autodl-tmp/p_Vision_transformer/datasets/flower_photos")
    # parser.add_argument('--data-path', type=str,
    #                     default=r"D:\X_PythonProdect\p_Vision_transformer\datasets\flower_photos")
    parser.add_argument('--data-path', type=str,
                        default=r"./datasets/cotton_insect")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default=r'models/model_comparative/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
