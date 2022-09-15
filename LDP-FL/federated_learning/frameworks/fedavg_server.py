import collections
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
import torchvision

from loss import Classification

from attack.reconstruction_algorithms import FedAvgReconstructor, GradientReconstructor
from attack import metrics
from attack.modules import MetaMonkey
from constant import consts
from data_process.data_mnist_read import DatasetMnist
from federated_learning.models.mlp import MLP
from federated_learning.models.mnist_cnn_model import MNISTCNN
from federated_learning.models.mnist_resnet_model \
    import _resnet, BasicBlock, Bottleneck
from federated_learning.frameworks.fedavg_client import FedAvgClient
from pub_lib.pub_libs import analyze_dist_of_single_att


class FedAvgServer(object):
    def __init__(self, sys_args, sys_setup):
        # get arguments
        self.sys_args = sys_args
        self.sys_setup = sys_setup

        # client info
        self.client_no = sys_args.client_no
        self.client_ratio = sys_args.client_ratio
        self.is_iid = sys_args.is_iid
        self.clients = list()

        # dataset and the corresponding dm and ds ([[[*]]]) from consts
        self.dataset = sys_args.dataset
        self.img_shape = None
        self.dm = torch.as_tensor(
            getattr(consts,
                    f"{self.dataset.upper()}_MEAN"), **sys_setup)[:, None, None]
        self.ds = torch.as_tensor(
            getattr(consts,
                    f"{self.dataset.upper()}_STD"), **sys_setup)[:, None, None]

        # training examples
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
        self.model_type = sys_args.model_name
        self.epoch_no = sys_args.epoch_no
        self.round_no = sys_args.round_no
        self.lr = sys_args.lr

        self.global_model = None
        self.loss_fn = None
        self.client_data_dispatch = None
        self.current_client_model_params = None

        # attack information
        self.attack_no = sys_args.attack_no
        self.attack_rounds = list()
        self.attack_targets = list()

    def prepare_before_training(self):
        # prepare for the model training
        self.prepare_data()
        self.data_dispatcher()
        self.construct_model()
        self.init_client_models()
        self.global_model.to(**self.sys_setup)
        self.global_model.eval()

        # prepare for the attack
        self.select_attack_rounds()

    def prepare_data(self):
        """
        training examples:
            torch: no * num_access * row_pixel * column_pixel
        training labels:
            torch: no * labels
        """
        if self.dataset.upper() == consts.DATASET_MNIST:
            data = DatasetMnist()
            data.read_data()
            num_access = 1

        self.training_examples = \
            torch.from_numpy(data.training_examples).type(torch.float32)
        self.training_row_pixel = data.training_row_pixel
        self.training_column_pixel = data.training_column_pixel
        self.training_example_no = self.training_examples.shape[0]
        self.training_examples = self.training_examples.reshape(
            self.training_example_no, num_access, self.training_row_pixel,
            self.training_column_pixel)

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

        # the first parameter is the number of image channels
        # img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])
        self.img_shape = (num_access, self.training_row_pixel,
                          self.training_column_pixel)

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
            # print("Client %s - Dist of Examples: %s" % (i, label_dist))
        print("--------------- End ----------------------")

    def construct_model(self, conv_kernel_size=None, conv_stride=None,
                        conv_padding=None, conv_channels=None,
                        pooling_kernel_size=2, pooling_stride=2,
                        fc_neuron_no=512):
        if self.model_type == consts.MNIST_MLP_MODEL:
            num_neurons = [200, 200]
            self.global_model = MLP(self.training_row_pixel,
                                    self.training_column_pixel,
                                    1,
                                    self.label_unique_no,
                                    num_neurons)
            self.global_model.construct_model()
            self.global_model.eval()
        elif self.model_type == consts.MNIST_CNN_MODEL:
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
        elif self.model_type == consts.ResNet18_MODEL:
            self.global_model = _resnet(BasicBlock, [2, 2, 2, 2], None, False)
        elif self.model_type == consts.ResNet34_MODEL:
            self.global_model = _resnet(BasicBlock, [3, 4, 6, 3], None, False)
        elif self.model_type == consts.ResNet50_MODEL:
            self.global_model = _resnet(Bottleneck, [3, 4, 6, 3], None, False)
        elif self.model_type == consts.ResNet101_MODEL:
            self.global_model = _resnet(Bottleneck, [3, 4, 23, 3], None, False)
        elif self.model_type == consts.ResNet152_MODEL:
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
                self.label_unique_no)
            self.clients.append(train_client)

    def train_model(self):
        print("Launch inverting attack:")
        ground_truth, labels, target_example_index = \
            self.init_target_example(0)
        recon_result = self.invert_gradient_attack(ground_truth, labels)
        self.save_reconstruction_example(
            ground_truth, recon_result, labels, target_example_index)

    def train_model1(self):
        client_train_no = int(self.client_no * self.client_ratio)
        for i in range(self.round_no):
            # randomly select part of clients
            chosen_clients = np.random.choice(self.client_no, client_train_no,
                                              replace=False)
            client_model_params = list()
            training_num = 0
            model_paras = self.global_model.state_dict()

            for chosen_client in chosen_clients:
                self.clients[chosen_client].construct_model(self.global_model)
                training_num = \
                    training_num + len(self.client_data_dispatch[chosen_client])

                # train the local model
                local_model = self.clients[chosen_client].training_model(
                        self.training_examples[
                            self.client_data_dispatch[chosen_client]],
                        self.training_labels[
                            self.client_data_dispatch[chosen_client]],
                        len(self.client_data_dispatch[chosen_client]),
                        self.epoch_no, self.lr)
                client_model_params.append(local_model)

            # launch inverting gradient attack
            if i in self.attack_rounds:
                target_client_index = self.select_attack_targets()
                target_client_id = chosen_clients[target_client_index]
                print("Launch inverting attack:")
                ground_truth, labels, target_example_index = \
                    self.init_target_example(target_client_id)
                recon_result = self.invert_gradient_attack(ground_truth, labels)
                self.save_reconstruction_example(
                    ground_truth, recon_result, labels, target_example_index)

            # update the global model parameters
            weight_keys = list(client_model_params[0].keys())
            fed_state_dict = collections.OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for k in range(client_train_no):
                    client_data_ratio = \
                        len(self.client_data_dispatch[k]) / training_num
                    key_sum += client_data_ratio * client_model_params[k][key]
                fed_state_dict[key] = key_sum

            self.global_model.load_state_dict(fed_state_dict)

            with torch.no_grad():
                if (i+1) % 10 == 0:
                    acc = self.compute_accuracy()
                    print("Round %s: Accuracy %.2f%%" % (i+1, acc * 100))

    def aggregate_global_model(self):
        weight_keys = list(self.current_client_model_params[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(self.client_no):
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
        if self.model_type in [consts.MNIST_CNN_MODEL, consts.ResNet18_MODEL]:
            result = self.global_model(
                self.test_examples.reshape(
                    self.test_example_no, 1, self.training_row_pixel,
                    self.training_column_pixel))\
                .reshape(self.test_example_no, -1)
        elif self.model_type == consts.MNIST_MLP_MODEL:
            result = self.global_model(self.test_examples)\
                .reshape(self.test_example_no, -1)
        else:
            exit(1)

        for i in range(self.test_example_no):
            pred_result = torch.argmax(result[i])
            if pred_result == self.test_labels[i]:
                accuracy = accuracy + 1
        return accuracy / self.test_example_no

    def select_attack_rounds(self):
        """
        select attack round and targets.
        """
        # select attack round
        if self.attack_no is None:
            self.attack_no = 1
            self.attack_rounds = [0]
        else:
            while len(self.attack_rounds) < self.attack_no:
                attack_round_id = np.random.randint(self.round_no)
                if attack_round_id not in self.attack_rounds:
                    self.attack_rounds.append(attack_round_id)
            self.attack_rounds.sort()

    def select_attack_targets(self):
        # select attack target in each attack round
        target_id = np.random.randint(int(self.client_no * self.client_ratio))
        return target_id

    def init_target_example(self, target_client_index):
        if self.sys_args.demo_target:  # demo image
            # Specify PIL filter for lower pillow versions
            ground_truth = torch.as_tensor(
                np.array(Image.open(consts.ATTACK_EXAMPLE_DEMO_DIR).resize(
                    (self.training_row_pixel, self.training_column_pixel),
                    Image.BICUBIC)) / 255, **self.sys_setup
            )
            ground_truth = ground_truth.permute(2, 0, 1).sub(self.dm) \
                .div(self.ds).unsqueeze(0).contiguous()
            if not self.sys_args.label_flip:
                labels = torch.as_tensor((1,), device=self.sys_setup["device"])
            else:
                labels = torch.as_tensor((5,), device=self.sys_setup["device"])
        else:
            target_example_id = np.random.randint(
                len(self.client_data_dispatch[target_client_index]))

            # ground_truth, labels = validloader.dataset[target_id]
            target_example_index = \
                self.client_data_dispatch[target_client_index][target_example_id]
            ground_truth = self.training_examples[target_example_index]
            labels = self.training_labels[target_example_index]

            if self.sys_args.label_flip:
                labels = (labels + 1) % self.label_unique_no

            ground_truth, labels = (
                ground_truth.unsqueeze(0).to(**self.sys_setup),
                torch.as_tensor((labels,), device=self.sys_setup["device"]),
            )
            print("The target example of client [ID: %s] is %s" %
                  (target_client_index, target_example_index))
        return ground_truth, labels, target_example_index

    def compute_updated_parameters(self, updated_parameters):
        patched_model = MetaMonkey(self.global_model)
        patched_model_origin = deepcopy(patched_model)

        # patched_model.parameters.items()
        patched_model.parameters = collections.OrderedDict(
            (name, param - param_origin)
            for ((name, param), (name_origin, param_origin))
            in zip(updated_parameters.items(),
                   patched_model_origin.parameters.items()))
        return list(patched_model.parameters.values())

    def invert_gradient_attack(self, target_ground_truth, target_labels):
        local_gradient_steps = self.sys_args.accumulation
        # local_lr = 1e-4
        local_lr = 1e-4
        target_client = FedAvgClient(self.model_type,
                                     self.training_row_pixel,
                                     self.training_column_pixel,
                                     self.label_unique_no)
        target_client.construct_model(self.global_model)

        """
        updated_parameters = \
            target_client.training_model(target_ground_truth, target_labels, 1,
                                         local_gradient_steps, local_lr)
        updated_parameters = self.compute_updated_parameters(updated_parameters)
        updated_parameters = [p.detach() for p in updated_parameters]
        """

        updated_parameters = self.test_compute_gradient(
            target_ground_truth, target_labels)

        config = dict(
            signed=self.sys_args.signed,
            boxed=self.sys_args.boxed,
            cost_fn=self.sys_args.cost_fn,
            indices=self.sys_args.indices,
            weights=self.sys_args.weights,
            lr=0.1,
            optim=self.sys_args.optimizer,
            restarts=self.sys_args.restarts,
            max_iterations=2_4000,
            total_variation=self.sys_args.tv,
            init=self.sys_args.init,
            filter="none",
            lr_decay=True,
            scoring_choice=self.sys_args.scoring_choice,
        )

        """
        rec_machine = FedAvgReconstructor(
            self.global_model, (self.dm, self.ds),
            local_gradient_steps, local_lr, config,
            num_images=self.sys_args.num_images, use_updates=True
        )
        """

        rec_machine = GradientReconstructor(
            self.global_model, (self.dm, self.ds), config, num_images=1)

        output, stats = rec_machine.reconstruct(
            updated_parameters, target_labels, img_shape=self.img_shape,
            dryrun=self.sys_args.dryrun)

        # Compute stats
        test_mse = (output - target_ground_truth).pow(2).mean().item()
        feat_mse = \
            (self.global_model(output) - self.global_model(target_ground_truth))\
                .pow(2).mean().item()
        test_psnr = \
            metrics.psnr(output, target_ground_truth, factor=1 / self.ds)

        print("Test Mse: %s, Feat Mse: %s, Test Psnr: %s" %
              (test_mse, feat_mse, test_psnr))
        return output

    def save_reconstruction_example(self, ground_truth, output, labels,
                                    example_id):
        # Save the resulting image
        if self.sys_args.save_image:
            os.makedirs(self.sys_args.image_path, exist_ok=True)
            output_denormalized = torch.clamp(output * self.ds + self.dm, 0, 1)
            rec_filename = (
                f'Re_{labels}_{self.sys_args.model_name}_{self.sys_args.cost_fn}-{example_id}.png'
            )
            torchvision.utils.save_image(output_denormalized, os.path.join(
                self.sys_args.image_path, rec_filename))

            gt_denormalized = torch.clamp(ground_truth * self.ds + self.dm, 0, 1)
            gt_filename = f"Int_{labels}_ground_truth-{example_id}.png"
            torchvision.utils.save_image(gt_denormalized, os.path.join(
                self.sys_args.image_path, gt_filename))

    def test_compute_gradient(self, ground_truth, labels):
        self.global_model.zero_grad()
        loss_fn = Classification()
        self.global_model.zero_grad()
        target_loss, _, _ = loss_fn(self.global_model(ground_truth), labels)

        # compute the gradients based on loss, parameters
        input_gradient = torch.autograd.grad(
            target_loss, self.global_model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        return input_gradient
