#coding=utf-8
# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FGSM tutorial on mnist using advbox tool.
FGSM method is non-targeted attack while FGSMT is targeted attack.
"""

import logging
#logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
#logger=logging.getLogger(__name__)



import sys
sys.path.append("..")
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data.dataloader as Data


from advbox.adversary import Adversary
from advbox.attacks.gradient_method import FGSM, BIM
from advbox.models.pytorchmnist import PytorchModel
from tutorials.mnist_model_pytorch import Net

def mahalanobis_dist(x,y):
    cov = np.load('../cov.data.npy')
    # 奇异矩阵的情况需要加上一个小的对角矩阵以便计算逆矩阵
    cov_abs = np.abs(cov)
    cov_abs[cov_abs == 0] = 10000
    min = np.min(cov_abs)
    append_matrix = np.eye(28*28) * min * 0.00001
    print('aa',append_matrix[0][0])
    covI = np.linalg.pinv(cov)

    A=x.reshape(28*28)
    B=y.reshape(28*28)

    return np.sqrt((A-B).dot(covI).dot((A-B).T))

def eu_dist(x,y):
    A=x.reshape(28*28)
    B=y.reshape(28*28)
    return np.linalg.norm(A - B)

def main():
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 5
    pretrained_model="./mnist-pytorch/net.pth"


    loss_func = torch.nn.CrossEntropyLoss()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist-pytorch/data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=False)

    # Define what device we are using
    logging.info("CUDA Available: {}".format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the network
    model = Net().to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    # advbox demo
    m = PytorchModel(
        model, loss_func,(0, 1),
        channel_axis=1)
    attack = BIM(m)

    attack_config = {"epsilons": 0.05, "epsilon_steps": 1, "epsilons_max": 0.05, "norm_ord": 1, "steps": 10}


    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data

        #inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels=inputs.numpy(),labels.numpy()

        #inputs.requires_grad = True
        #print(inputs.shape)

        total_count += 1
        adversary = Adversary(inputs, labels[0])

        # FGSM non-targeted attack
        adversary = attack(adversary, **attack_config)
        print(adversary.original.shape)
        pertubation = adversary.perturbation()  # (1,1,28,28)
        m_dist = mahalanobis_dist(adversary.original, adversary.adversarial_example)
        e_dist = eu_dist(adversary.original, adversary.adversarial_example)
        print('###')
        print(m_dist)
        print(e_dist)
        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (labels, adversary.adversarial_label, total_count))

        else:
            print('attack failed, original_label=%d, count=%d' %
                  (labels, total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    print("fgsm attack done")


if __name__ == '__main__':
    main()
