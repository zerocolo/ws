import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data as data_
from coco_dataset import COCODataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import os
from dataset_image import MyTestData
from PIL import Image
import numpy as np
from pre_train_1 import resnext101

use_cam = True
def train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=50):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        obj_running_loss = 0.0
        k = len(dataload)

        for i, batch in enumerate(dataload):
            image, label = batch
            label = label.type(torch.LongTensor)
            labels = torch.zeros((1, 81)).scatter_(1, label, 1)

            inputs = image.to(device).float()
            labels = labels.to(device)

            optimizer_ft.zero_grad()

            obj_outputs, _ = model_ft(inputs)

            obj_loss = criterion(obj_outputs, labels)

            obj_loss.backward()

            optimizer_ft.step()

            obj_running_loss += obj_loss.data[0]

            if i % 50 == 0:
                print('{}/{}, obj Loss: {:.4f}'.format(
                     i, k, obj_loss.item()))

            del inputs, labels, obj_outputs,  obj_loss

        obj_epoch_loss = obj_running_loss / k

        print('obj Loss: {:.4f}'.format(obj_epoch_loss))

        torch.save(model_ft.state_dict(), "./coco.pth")

        test(Testloader, model_ft, out_put)

    return model_ft

def test(loader, feature, output_dir1):

    for ib, (data, ori_img, name, size) in enumerate(loader):

        inputs = Variable(data).cuda().float()

        _, mak = feature(inputs)

        mak = F.interpolate(mak, size=(size[1], size[0]), mode='bilinear')
        mak += 0.05
        mask = mak.data[0, 0].cpu().numpy()

        mask = (mask * 255)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(output_dir1, name[0] + '.png'), 'png')

    print('all save!')

def load_pth_res(model_ft):

    resnet101 = models.resnet101(pretrained=True)
    fc_features = resnet101.fc.in_features
    resnet101.fc = nn.Linear(fc_features, 81)

    pretrained_dict = resnet101.state_dict()
    model_dict = model_ft.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(len(pretrained_dict.keys()))
    model_dict.update(pretrained_dict)
    model_ft.load_state_dict(model_dict)

    return model_ft


if __name__ == '__main__':

    data_dir = './data/COCO/'
    data_dir1 = './DUTS/DUTS-TR/'
    out_put = './test_val/'
    piror = 'DUTS-TR-Image'

    training_params = {"batch_size": 1,
                       "shuffle": False,
                       "drop_last": True}

    training_set = COCODataset(data_dir, "2014", "train", 256)

    dataload = DataLoader(training_set, **training_params)

    Testloader = DataLoader(MyTestData(data_dir1, piror, transform=True),batch_size=1, num_workers=4, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = resnext101()
    model_ft.train()

    model_ft = load_pth_res(model_ft)
    #model_ft.load_state_dict(torch.load('./coco.pth'))
    model_ft = model_ft.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00001)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=30)
    torch.save(model_ft.state_dict(), "./coco.pth")
    print('model save')