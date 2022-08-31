import collections
import numpy as np
import torch

from constant import consts as const
from data_process.data_mnist_read import DatasetMnist

from federated_learning.models.mlp import MLP
from federated_learning.models.mnist_cnn_model import MNISTCNN
from federated_learning.models.mnist_resnet_model \
    import _resnet, BasicBlock, Bottleneck

from federated_learning.frameworks.fedavg_client import FedAvgClient
from pub_lib.pub_libs import analyze_dist_of_single_att


class FedAvgServer(object):
    def __init__(self, client_no, client_train_ratio, dataset, model_type,
                 is_iid=True, round_no=500, epoch_no=10, lr=0.001):
        # client info
        self.client_no = client_no
        self.client_train_ratio = client_train_ratio
        self.is_iid = is_iid
        self.clients = list()

        # training examples
        self.dataset = dataset
        self.training_examples = None
        self.training_example_no = None
        self.training_row_pixel = None
        self.training_column_pixel = None
        self.training_labels = None
        self.label_unique_no = None
        self.unique_labels = None

        # test examples
        self.test_examples = None
        self.test_labels = None
        self.test_example_no = None

        # model info
        self.model_type = model_type
        self.global_model = None
        self.epoch_no = epoch_no
        self.lr = lr
        self.loss_fn = None
        self.client_data_dispatch = None
        self.round_no = round_no
        self.current_client_model_params = None

    def prepare_data(self):
        if self.dataset.upper() == const.DATASET_MNIST:
            data = DatasetMnist()
            data.read_data()

        self.training_examples = \
            torch.from_numpy(data.training_examples).type(torch.float32)
        self.training_row_pixel = data.training_row_pixel
        self.training_column_pixel = data.training_column_pixel
        self.training_example_no = self.training_examples.shape[0]

        self.training_labels = \
            torch.from_numpy(data.training_labels).type(torch.int64)
        self.unique_labels = self.training_labels.unique()
        self.label_unique_no = self.unique_labels.size()[0]
        self.training_labels = self.training_labels.reshape(-1, 1)

        self.test_examples = \
            torch.from_numpy(data.test_examples).type(torch.float32)
        self.test_labels = \
            torch.from_numpy(data.test_labels).type(torch.int64)
        self.test_example_no = self.test_examples.shape[0]

    def data_dispatcher(self):
        """
        Function: execute the data assignment for the clients.
        self.client_data_dispatch - type: array, shape: client_no * example_no
        """
        if self.is_iid:
            # training examples shuffle
            self.client_data_dispatch = np.arange(self.training_example_no)
            np.random.shuffle(self.client_data_dispatch)
            self.client_data_dispatch = \
                self.client_data_dispatch.reshape(self.client_no, -1)
        else:
            """
            Dispatch method 1:
            self.client_data_dispatch = \
                self.training_labels.reshape(1, -1).numpy().argsort()[0]
            """
            # 10 is ok.
            example_block_no = 1
            client_order = np.arange(self.client_no * example_block_no)
            np.random.shuffle(client_order)
            client_order = client_order.reshape(-1, example_block_no)

            self.client_data_dispatch = \
                self.training_labels.reshape(1, -1).numpy().argsort()[0]
            self.client_data_dispatch = \
                self.client_data_dispatch.reshape(
                    self.client_no * example_block_no, -1)
            self.client_data_dispatch = self.client_data_dispatch[client_order]
            self.client_data_dispatch = \
                self.client_data_dispatch.reshape(self.client_no, -1)

        print("---------- Data Distribution -------------")
        print("Dist of Total Examples is %s" %
              analyze_dist_of_single_att(self.training_labels))
        for i in range(self.client_no):
            label_dist = analyze_dist_of_single_att(
                self.training_labels[self.client_data_dispatch[i]])
            print("Client %s - Dist of Examples: %s" % (i, label_dist))
        print("--------------- End ----------------------")

    def construct_model(self, conv_kernel_size=None, conv_stride=None,
                        conv_padding=None, conv_channels=None,
                        pooling_kernel_size=2, pooling_stride=2,
                        fc_neuron_no=512):
        if self.model_type == const.MNIST_MLP_MODEL:
            num_neurons = [200, 200]
            self.global_model = MLP(self.training_row_pixel,
                                    self.training_column_pixel,
                                    1,
                                    self.label_unique_no,
                                    num_neurons)
            self.global_model.construct_model()
        elif self.model_type == const.MNIST_CNN_MODEL:
            self.global_model = MNISTCNN(
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
            self.global_model.initial_layers()
        elif self.model_type == const.ResNet18_MODEL:
            self.global_model = _resnet(BasicBlock, [2, 2, 2, 2], None, False)
        elif self.model_type == const.ResNet34_MODEL:
            self.global_model = _resnet(BasicBlock, [3, 4, 6, 3], None, False)
        elif self.model_type == const.ResNet50_MODEL:
            self.global_model = _resnet(Bottleneck, [3, 4, 6, 3], None, False)
        elif self.model_type == const.ResNet101_MODEL:
            self.global_model = _resnet(Bottleneck, [3, 4, 23, 3], None, False)
        elif self.model_type == const.ResNet152_MODEL:
            self.global_model = _resnet(Bottleneck, [3, 8, 36, 3], None, False)
        else:
            print("Error: There is no model named %s" % self.model_type)
            exit(1)

    def init_client_models(self):
        for i in range(self.client_no):
            train_client = FedAvgClient(
                self.model_type,
                self.training_row_pixel,
                self.training_column_pixel,
                self.label_unique_no,
                self.epoch_no, self.lr)
            self.clients.append(train_client)

    def train_model(self):
        client_train_no = int(self.client_no * self.client_train_ratio)

        for i in range(self.round_no):
            # randomly select part of clients
            chosen_clients = np.random.choice(self.client_no, client_train_no,
                                              replace=False)
            client_model_params = list()
            training_num = 0
            model_paras = self.global_model.state_dict()

            for j in chosen_clients:
                self.clients[j].construct_model(self.global_model)
                # self.clients[j].initial_model(
                #     self.mnist_model, {"conv_kernel_size": })
                training_num = training_num + len(self.client_data_dispatch[j])
                client_model_params.append(
                    self.clients[j].training_model(
                        self.training_examples[self.client_data_dispatch[j]],
                        self.training_labels[self.client_data_dispatch[j]],
                        len(self.client_data_dispatch[j]),
                        self.test_examples,
                        self.test_labels))

            # update the global model parameters
            weight_keys = list(client_model_params[0].keys())
            fed_state_dict = collections.OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for k in range(client_train_no):
                    # client_paras = \
                    #     self.clients[i].local_mnist_2nn_model.state_dict()
                    client_data_ratio = \
                        len(self.client_data_dispatch[k]) / training_num
                    key_sum += client_data_ratio * client_model_params[k][key]
                fed_state_dict[key] = key_sum

            self.global_model.load_state_dict(fed_state_dict)
            with torch.no_grad():
                acc = self.compute_accuracy()
                print("Round %s: Accuracy %.2f%%" % (i, acc * 100))

    def aggregate_global_model(self):
        weight_keys = list(self.current_client_model_params[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(self.client_no):
                # client_paras = \
                #     self.clients[i].local_mnist_2nn_model.state_dict()
                client_data_ratio = \
                    len(self.client_data_dispatch[i]) / self.training_example_no
                key_sum += \
                    client_data_ratio * self.current_client_model_params[i][key]
            fed_state_dict[key] = key_sum
        return fed_state_dict

    def model_parameter_no(self):
        paras = list(self.global_model.parameters())
        print("------------- Model Structure -------------")
        for num, para in enumerate(paras):
            para_size = para.size()
            print("%s: %s" % (num, para_size))
        print("------------------- END -------------------")

    def compute_accuracy(self):
        accuracy = 0.0
        if self.model_type in [const.MNIST_CNN_MODEL, const.ResNet18_MODEL]:
            result = self.global_model(
                self.test_examples.reshape(
                    self.test_example_no, 1, self.training_row_pixel,
                    self.training_column_pixel))\
                .reshape(self.test_example_no, -1)
        elif self.model_type == const.MNIST_MLP_MODEL:
            result = self.global_model(self.test_examples)\
                .reshape(self.test_example_no, -1)
        else:
            exit(1)

        for i in range(self.test_example_no):
            pred_result = torch.argmax(result[i])
            if pred_result == self.test_labels[i]:
                accuracy = accuracy + 1
        return accuracy / self.test_example_no