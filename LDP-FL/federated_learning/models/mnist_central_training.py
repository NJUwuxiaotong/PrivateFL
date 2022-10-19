import json
import numpy as np
import torch
import torchvision
from torch import nn

from constant import consts as const
from data_process.data_load import DatasetMnist
from federated_learning.models.cnn2layer import CNN2Layer
from federated_learning.models.mnist_2nn_model import MNIST2NN

from federated_learning.models.resnet import \
    _resnet, BasicBlock, Bottleneck


class MNISTCentralGen(object):
    def __init__(self, model_name, neuro_no, epoch_no=10):
        self.model_name = model_name
        self.neuron_no = neuro_no
        self.training_examples = None
        self.training_labels = None
        self.training_example_no = None
        self.label_unique_no = None
        self.unique_labels = None
        self.training_row_pixel = None
        self.training_column_pixel = None

        self.test_examples = None
        self.test_labels = None
        self.test_example_no = None

        self.normalize = torchvision.transforms.Normalize(mean=[0.5], std=[0.5])

        self.mnist_training_model = None
        self.epoch_no = epoch_no
        self.opt = None
        self.loss_fn = None

    def prepare_data(self):
        # data
        mnist_data = DatasetMnist()
        mnist_data.read_data()

        self.training_examples = \
            torch.from_numpy(mnist_data.training_examples).type(torch.float32)
        self.training_row_pixel = mnist_data.training_row_pixel
        self.training_column_pixel = mnist_data.training_column_pixel
        self.training_example_no = self.training_examples.shape[0]

        self.training_labels = \
            torch.from_numpy(mnist_data.training_labels).type(torch.int64)
        self.training_labels = self.training_labels.reshape(-1, 1)
        self.unique_labels = self.training_labels.unique()
        self.label_unique_no = self.unique_labels.size()[0]

        self.test_examples = \
            torch.from_numpy(mnist_data.test_examples).type(torch.float32)
        self.test_labels = \
            torch.from_numpy(mnist_data.test_labels).type(torch.int64)
        self.test_example_no = self.test_examples.shape[0]

        # normalize images
        # self.training_examples = self.normalize(self.training_examples)
        # self.test_examples = self.normalize(self.test_examples)

    def initial_model(self, conv_kernel_size=None, conv_stride=None,
                      conv_padding=None, conv_channels=None,
                      pooling_kernel_size=2, pooling_stride=2,
                      fc_neuron_no=512):
        if self.model_name == const.MNIST_MLP_MODEL:
            self.mnist_training_model = MNIST2NN(
                self.training_row_pixel,
                self.training_column_pixel,
                self.label_unique_no,
                self.neuron_no
            )
            self.mnist_training_model.initial_layers()
        elif self.model_name == const.MNIST_CNN_MODEL:
            self.mnist_training_model = CNN2Layer(
                self.training_row_pixel,
                self.training_column_pixel,
                self.label_unique_no,
                conv_kernel_size,
                conv_stride,
                conv_padding,
                conv_channels,
                pooling_kernel_size,
                pooling_stride,
                fc_neuron_no)
            self.mnist_training_model.initial_layers()
        elif self.model_name == const.ResNet18_MODEL:
            self.mnist_training_model = \
                _resnet(BasicBlock, [2, 2, 2, 2], None, False)
        elif self.model_name == const.ResNet34_MODEL:
            self.mnist_training_model = _resnet(
                BasicBlock, [3, 4, 6, 3], None, False)
        elif self.model_name == const.ResNet50_MODEL:
            self.mnist_training_model = _resnet(
                Bottleneck, [3, 4, 6, 3], None, False)
        elif self.model_name == const.ResNet101_MODEL:
            self.mnist_training_model = _resnet(
                Bottleneck, [3, 4, 23, 3], None, False)
        elif self.model_name == const.ResNet152_MODEL:
            self.mnist_training_model = _resnet(
                Bottleneck, [3, 8, 36, 3], None, False)
        else:
            print("Warning: The specified model [%s] does not exist!" %
                  self.model_name)
            exit(1)

        self.opt = None
        self.loss_fn = nn.CrossEntropyLoss()

    def model_parameter_no(self):
        paras = list(self.mnist_training_model.parameters())
        print("------------- Model Structure -------------")
        for num, para in enumerate(paras):
            para_size = para.size()
            print("%s: %s" % (num, para_size))
        print("------------------- END -------------------")

    def training_model(self, batch_size = 50, output_file_dir=None):
        opt = torch.optim.SGD(self.mnist_training_model.parameters(), lr=0.005)

        batch_no = int(self.training_example_no / batch_size)
        accuracy_set = list()

        for epoch in range(self.epoch_no):
            start_pos = 0
            new_example_order = np.arange(self.training_example_no)
            np.random.shuffle(new_example_order)
            for i in range(batch_no):
                training_examples_order = \
                    new_example_order[start_pos: start_pos + batch_size]
                examples_feature = \
                    self.training_examples[training_examples_order]
                examples_labels = \
                    self.training_labels[training_examples_order].reshape(1, -1)

                if self.model_name in \
                        [const.MNIST_CNN_MODEL, const.ResNet18_MODEL]:
                    examples_feature = examples_feature.reshape(
                        batch_size, 1, self.training_row_pixel,
                        self.training_column_pixel)

                pred_labels = self.mnist_training_model(examples_feature)
                loss = self.loss_fn(pred_labels, examples_labels[0])
                opt.zero_grad()
                loss.backward()
                opt.step()
                start_pos = start_pos + batch_size

            with torch.no_grad():
                acc = self.compute_accuracy()
                accuracy_set.append(acc)
                print("Epoch %s: Accuracy %.2f%%" % (epoch, acc * 100))

        # with open(output_file_dir, "r") as output_f:
        #     json.dump(accuracy_set, output_f)

    def compute_accuracy(self):
        accuracy = 0.0
        if self.model_name in [const.MNIST_CNN_MODEL, const.ResNet18_MODEL]:
            result = self.mnist_training_model(
                self.test_examples.reshape(
                    self.test_example_no, 1, self.training_row_pixel,
                    self.training_column_pixel))\
                .reshape(self.test_example_no, -1)
        elif self.model_name == const.MNIST_MLP_MODEL:
            result = self.mnist_training_model(self.test_examples)\
                .reshape(self.test_example_no, -1)
        else:
            exit(1)

        for i in range(self.test_example_no):
            pred_result = torch.argmax(result[i])
            if pred_result == self.test_labels[i]:
                accuracy = accuracy + 1
        return accuracy / self.test_example_no

        # optimizer = torch.optim.Adam(mo.parameters(), lr=0.0001,
        #                            weight_decay=0.001)
        # weight_decay is the coefficient of the normalization. Based on the above
        # definition, all the parameters of the model are be normalized.
        # Theoretically, it is not necessary to normalize the bias. Instead, it limits
        # the capability of the model.
        # if we hope that the bias is not be normalized,
        # optimizer = torch.optim.Adam([
        #    {"params": mo.parameters(), "weight_decay": 0.0001},
        #    {"params": mo.layer.bias}], lr=0.001)
