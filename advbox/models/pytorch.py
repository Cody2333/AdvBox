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
Tensorflow model
"""
from __future__ import absolute_import

import numpy as np
import os

from .base import Model

import logging
logger=logging.getLogger(__name__)


import torchvision
from torch.autograd import Variable
import torch.nn as nn


#直接加载pb文件
class PytorchModel(Model):


    def __init__(self,
                 model,
                 loss,
                 bounds,
                 channel_axis=3,
                 preprocess=None):

        import torch


        if preprocess is None:
            preprocess = (0, 1)

        super(PytorchModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)


        self._model = model

        #暂时不支持自定义loss
        self._loss=loss

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(self._device)

        logger.info("Finish PytorchModel init")

    #返回值为标量
    def predict(self, data):
        """
        Calculate the prediction of the data.
        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).
        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """

        import torch

        scaled_data = self._process_input(data)

        scaled_data = torch.from_numpy(scaled_data).to(self._device)


        # Run prediction
        predict = self._model(scaled_data)
        predict = np.squeeze(predict, axis=0)

        predict=predict.detach()

        predict=predict.cpu().numpy().copy()

        #logging.info(predict)

        return predict

    #返回值为tensor
    def predict_tensor(self, data):
        """
        Calculate the prediction of the data.
        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).
        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """

        import torch

        scaled_data = self._process_input(data).to(self._device)

        #scaled_data = torch.from_numpy(scaled_data)


        # Run prediction
        predict = self._model(scaled_data)
        #predict = np.squeeze(predict, axis=0)

        #predict=predict.detach()

        #predict=predict.numpy()

        #logging.info(predict)

        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """

        return self._nb_classes

    def gradient(self, data, label):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.
        Args:
            data(numpy.ndarray): input data with shape (size, height, width,
            channels).
            label(int): Label used to calculate the gradient.
        Return:
            numpy.ndarray: gradient of the cross-entropy loss w.r.t the image
                with the shape (height, width, channel).
        """

        import torch
        cov = np.load('../cov.cifar.npy')
        covI = np.linalg.inv(cov)
        u = np.load('../u.cifar.npy')
        covI = torch.from_numpy(covI).float().to(self._device)
        u = torch.from_numpy(u).float().to(self._device)
        scaled_data = self._process_input(data)

        #logging.info(scaled_data)

        scaled_data = torch.from_numpy(scaled_data).to(self._device)
        scaled_data.requires_grad = True

        label = np.array([label])
        label = torch.from_numpy(label).to(self._device)
        #label = torch.Tensor(label).to(self._device)

        output=self.predict_tensor(scaled_data).to(self._device)
        loss=-self._loss(output, label)

        # 计算马氏距离
        tt=scaled_data.reshape(32*32*3).float()
        mat = (tt-u).reshape(32*32*3, 1)
        loss2=torch.sqrt(mat.t().float().mm(covI.float()).mm(mat.float()).float()).reshape(1)[0]
        # loss = loss + loss2 * 0.0001
        # ce = nn.CrossEntropyLoss()
        # loss=-ce(output, label)

        #计算梯度
        # Zero all existing gradients
        self._model.zero_grad()
        loss.backward()
        grad1 = scaled_data.grad.cpu().numpy().copy()
        grad1 = torch.from_numpy(grad1)
        sum1 = torch.sum(torch.abs(scaled_data.grad))
        self._model.zero_grad()
        loss2.backward()
        grad2 = scaled_data.grad.cpu().numpy().copy()
        grad2 = torch.from_numpy(grad2)
        sum2 = torch.sum(torch.abs(scaled_data.grad))
        # print(sum1,sum2)
        # print(grad1, grad2)
        grad = grad1 * 1 + sum1 /sum2 * grad2 * 100 # 负数为拉近距离，正数为拉远距离
        print(torch.mean(grad2))
        #技巧 梯度也是tensor 需要转换成np
        grad = grad.cpu().numpy().copy()
        return grad.reshape(scaled_data.shape)

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
