import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision.datasets  import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from define_dataset import read_dataset
import random
import torchvision.transforms.functional as tf
from PIL import Image
from New_net import Net
import os
from choose_datafloder import kflod
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def my_transform(image):
    if random.random() > 0.5:
        image = tf.hflip(image)
    # image = transforms.ToPILImage()(image)
    if random.random() > 0.5:
        angle = random.choice([-90, 90])
        #image = image.rotate(angle)
        image = rot_img(image, theta=angle)

    if random.random() > 0.5:
        print('dealing with the jpeg')
    # #choose some batch,and 100 pictures in batch to make the jpeg
    #     for number in range(100):
    #         sub_image = image[number, :, :, :]
    #         sub_image = transforms.ToPILImage()(sub_image)
    #         quality = random.randint(50, 95)
    #         sub_image.save('/home/laippe/PycharmProjects/practice/tupian.jpg', format='JPEG', quality=quality)
    #         sub_image = Image.open('/home/laippe/PycharmProjects/practice/tupian.jpg')
    #         sub_image = transforms.ToTensor()(sub_image)
    #         image[number, :, :, :] = sub_image
    #choose some batch,and 100 pictures in batch to make the jpeg
        sub_image = image.squeeze()
        sub_image = transforms.ToPILImage()(sub_image)
        quality = random.randint(50, 95)
        sub_image.save('/home/haodong1/lailie/practice/tupian.jpg', format='JPEG', quality=quality)
        sub_image = Image.open('/home/haodong1/lailie/practice/tupian.jpg')
        sub_image = transforms.ToTensor()(sub_image)
        image = sub_image.unsqueeze(0)
    return image

#rot function
def rot_img(x, theta, dtype=torch.FloatTensor):
    theta = torch.tensor(theta).float()
    get_rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])
    B, C, H, W = x.shape
    rot_mat = get_rot_mat[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, size=[B, C, W, H]).type(dtype)
    x = F.grid_sample(x, grid)
    return x





batch_size = 1
learning_rate = 0.0001
folder_path = '/home/haodong1/lailie/kfolder/5_floder'
for num in range(1, 6):
    print('now starting the %d train ' % num)
    #prepare the dataset
    root = '/home/haodong1/lailie/ktrain1'
    out_path = os.path.join(root, 'train%d' % num)
    kflod(num, folder_path, out_path)
    train_dir = os.path.join(out_path, 'train')
    test_dir = os.path.join(out_path, 'test')
    transformss = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ]
    )

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transformss, target_transform=None, is_valid_file=None)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transforms.ToTensor(), target_transform=None, is_valid_file=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True)
    #define the net
    net = Net()
    model_dict = net.state_dict()
    pre_net = torch.load('/home/haodong1/lailie/model/net1.pth')
    pretrained_dict = pre_net.state_dict()
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    net.load_state_dict(model_dict)

    #define device as the first visible cuda device if we have CUDA available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #convert parameters and buffers of all modules to CUDA Tensor
    net.to(device)


    #build the optimer and loss
    # criterion = torch.nn.NLLLoss2d(weight=torch.tensor([0.6, 0.4]))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)



    #train
    num_epoch = 100
    train_loss = []
    test_loss = []
    for epoch in range(num_epoch):
        sub_train_loss = []
        sub_test_loss = []
        net.train()
        print("starting %d epoch train" % epoch)
        for i, (image, target) in enumerate(train_loader, 0):
            #enlarge the image
            print('now train the %d picture'%i)
            image = my_transform(image)
            #convert tensor to Variable
            image = image.type(torch.FloatTensor)
            image = Variable(image).cuda()
            # image = Variable(image)
            target = target.type(torch.LongTensor)
            target = Variable(target).cuda()
            # target = Variable(target)
            #forward backward optimize
            optimizer.zero_grad()
            outputs = net(image, stage=2)
            loss_train = criterion(outputs, target)
            loss_train.backward()
            optimizer.step()
            print('The loss in train data: ', loss_train.item())
            if (i+1) % 10 == 0:
                print('epoch [%d/%d], step [%d/%d], loss: %.4f'
                      % (epoch+1, num_epoch, i+1, len(train_loader), loss_train.item()))
                sub_train_loss.append(loss_train.item())
        train_loss.append(np.mean(sub_train_loss))
        net.eval()
        print("starting %d epoch eval" % epoch)
        for images, target in test_loader:
            images = images.type(torch.FloatTensor)
            images = Variable(images).cuda()
            # images = Variable(images)
            output = net(images, stage=2)
            target = target.type(torch.LongTensor)
            target = Variable(target).cuda()
            # target = Variable(target)
            loss_test = criterion(output, target)
            print('The loss in test data: ', loss_test.item())
            sub_test_loss.append(loss_test.item())
        test_loss.append(np.mean(sub_test_loss))
        print('loss in epoch %d is train:%d test %d' % (epoch, np.mean(sub_train_loss), np.mean(sub_test_loss)))
    #save the model and loss
    state = {'net': net.state_dict(), 'train_loss': train_loss, 'test_loss': test_loss}
    torch.save(state, '/home/haodong1/lailie/model1/state%d.pth' % num)
    torch.save(net, '/home/haodong1/lailie/model1/net%d.pth' % num)
    print("save the %d train sucessfully!!" % num)
