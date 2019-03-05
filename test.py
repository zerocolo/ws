from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F
import os
from torch.utils import data as data_
from pre_train_1 import resnext101
from PIL import Image
from dataset_image import MyTestData
from crf import run_crf
from dataset_image import MyTestData


def main():
    output_dir1 = './test_val1/'
    data_dir1 = './MSRA-B/'
    pior = 'MSRA-B'

    Testloader = data_.DataLoader(MyTestData(data_dir1, pior), batch_size=1, num_workers=4, shuffle=False)

    feature = resnext101()
    feature.load_state_dict(torch.load('./coco.pth'))
    feature.cuda()

    test(Testloader, feature, output_dir1)

def test(loader, feature, output_dir1):

    with torch.no_grad():
        feature.eval()
        it = 0
        for ib, (data, ori_img, name, size) in enumerate(loader):
            print(it)
            '''if it > 100:
                break'''

            inputs = Variable(data).cuda()

            _, msk = feature(inputs)

            mak = F.interpolate(msk, size=256, mode='bilinear')
            mak = F.sigmoid(mak)

            mask = mak.data[0].cpu().numpy()
            ori_img = inputs.data[0].cpu().numpy()

            mask = run_crf(ori_img, mask)

            mask = mask.data[0].cpu().numpy()
            mask = (mask * 255)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(output_dir1, name[0] + '.png'), 'png')
            it += 1


if __name__ == '__main__':
    main()
