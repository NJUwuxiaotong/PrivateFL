import collections
import copy
import json
import numpy as np
import os
import torch
import torchvision

from copy import deepcopy
from PIL import Image
from torch import nn

from attack.reconstruction_algorithms \
    import FedAvgReconstructor, GradientReconstructor
from attack import metrics
from attack.modules import MetaMonkey
from constant import consts
from federated_learning.models.mlp2layer import MLP2Layer
from federated_learning.models.cnn2layer import CNN2Layer
from federated_learning.models.cnn4layer import CNN4Layer
from federated_learning.models.resnet \
    import _resnet, BasicBlock, Bottleneck


class FedAvgServer(object):
    def __init__(self, sys_args, sys_defs, sys_setup, valid_loader, valid_info,
                 class_no):
        # get arguments
        self.sys_args = sys_args
        self.sys_setup = sys_setup

        # client info
        self.client_no = sys_args.client_no
        self.client_ratio = sys_args.client_ratio

        # dataset and the corresponding dm and ds ([[[*]]]) from consts
        self.dataset = sys_args.dataset
        self.dm = torch.as_tensor(
            getattr(consts,
                    f"{self.dataset.upper()}_MEAN"), **sys_setup)[:, None, None]
        self.ds = torch.as_tensor(
            getattr(consts,
                    f"{self.dataset.upper()}_STD"), **sys_setup)[:, None, None]

        # example info
        self.example_shape = valid_info.example_shape
        self.example_channel = self.example_shape[0]
        self.example_row_pixel = self.example_shape[1]
        self.example_column_pixel = self.example_shape[2]
        self.class_no = class_no

        # test examples
        self.valid_loader = valid_loader
        self.valid_info = valid_info
        self.test_examples = None
        self.test_labels = None
        self.test_example_no = valid_info.example_no
        self.get_test_examples()

        # model info
        self.model_type = sys_args.model_name
        self.epoch_no = sys_args.epoch_no
        self.round_no = sys_args.round_no
        self.lr = sys_args.lr
        self.batch_size = sys_args.batch_size

        self.global_model = None
        self.model_shape = None
        self.center_radius_stats = None
        self.loss_fn = None
        self.current_client_model_params = None

        # attack information
        self.attack_no = sys_args.attack_no
        self.attack_rounds = list()
        self.attack_targets = list()

        self.softmax = nn.Softmax(dim=1)

    def prepare_before_training(self):
        # prepare for the model training
        self.construct_model()
        self.global_model.to(**self.sys_setup)
        self.get_model_shape()
        self.get_center_radius_of_model()
        # prepare for the attack
        self.select_attack_rounds()

    def get_test_examples(self):
        examples = list()
        labels = list()
        for example, label in self.valid_loader:
            examples.extend(example.tolist())
            labels.extend(label.tolist())

        self.test_examples = \
            torch.tensor(examples, device=self.sys_setup["device"])
        self.test_labels = torch.tensor(labels, device=self.sys_setup["device"])

    def construct_model(self):
        if self.dataset.upper() == consts.DATASET_MNIST:
            if self.model_type == consts.MNIST_MLP_MODEL:
                num_neurons = [200, 200]
                self.global_model = MLP2Layer(self.example_shape,
                                              self.class_no,
                                              num_neurons)
                self.global_model.construct_model()
            elif self.model_type == consts.MNIST_CNN_MODEL:
                model_params = \
                    {"conv1": {"in_channel": 1,
                               "out_channels": 32,
                               "kernel_size": 5,
                               "stride": 1,
                               "padding": 2},
                     "pool1": {"kernel_size": 2,
                               "stride": 2},
                     "conv2": {"in_channel": 32,
                               "out_channels": 64,
                               "kernel_size": 5,
                               "stride": 1,
                               "padding": 2},
                     "pool2": {"kernel_size": 2,
                               "stride": 2},
                     "fc": {"in_neuron": 7*7*64,
                            "out_neuron": 512}}
                self.global_model = CNN2Layer(
                    self.example_shape,
                    self.class_no, **model_params)
                self.global_model.initial_layers()
            elif self.model_type == consts.ResNet18_MODEL:
                self.global_model = _resnet(BasicBlock, [2, 2, 2, 2],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet34_MODEL:
                self.global_model = _resnet(BasicBlock, [3, 4, 6, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet50_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 4, 6, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet101_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 4, 23, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet152_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 8, 36, 3],
                                            self.example_channel, None, False)
        elif self.dataset.upper() == consts.DATASET_CIFAR10:
            if self.model_type == consts.CIFAR10_CNN_MODEL:
                model_params = {"conv1": {"in_channel": 3,
                               "out_channels": 32,
                               "kernel_size": 3,
                               "stride": 1,
                               "padding": 1},
                     "pool1": {"kernel_size": 2,
                               "stride": 2},
                     "conv2": {"in_channel": 32,
                               "out_channels": 64,
                               "kernel_size": 3,
                               "stride": 1,
                               "padding": 1},
                     "pool2": {"kernel_size": 2,
                               "stride": 2},
                     "conv3": {"in_channel": 64,
                               "out_channels": 128,
                               "kernel_size": 3,
                               "stride": 1,
                               "padding": 1},
                     "pool3": {"kernel_size": 2,
                               "stride": 2},
                     "conv4": {"in_channel": 128,
                               "out_channels": 256,
                               "kernel_size": 3,
                               "stride": 1,
                               "padding": 1},
                     #"pool4": {"kernel_size": 2,
                     #          "stride": 2},
                     "fc1": {"in_neuron": 4*4*128,
                            "out_neuron": 4*4*128},
                     "fc2": {"in_neuron": 4*4*128,
                             "out_neuron": 128*4}}
                self.global_model = CNN4Layer(
                    self.example_shape,
                    self.class_no, **model_params)
                self.global_model.initial_layers()
            elif self.model_type == consts.ResNet18_MODEL:
                self.global_model = _resnet(BasicBlock, [2, 2, 2, 2],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet34_MODEL:
                self.global_model = _resnet(BasicBlock, [3, 4, 6, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet50_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 4, 6, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet101_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 4, 23, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet152_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 8, 36, 3],
                                            self.example_channel, None, False)
        elif self.dataset == consts.DATASET_IMAGENET:
            if self.model_type == consts.ResNet18_MODEL:
                self.global_model = _resnet(BasicBlock, [2, 2, 2, 2],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet34_MODEL:
                self.global_model = _resnet(BasicBlock, [3, 4, 6, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet50_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 4, 6, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet101_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 4, 23, 3],
                                            self.example_channel, None, False)
            elif self.model_type == consts.ResNet152_MODEL:
                self.global_model = _resnet(Bottleneck, [3, 8, 36, 3],
                                            self.example_channel, None, False)
        else:
            print("Error: There is no model named %s" % self.model_type)
            exit(1)
        self.present_network_structure()

    def get_client_order(self):
        # randomly generate the order of clients
        each_client_no = int(self.round_no * self.client_ratio)
        clients_order = [i for i in range(self.client_no)] * each_client_no
        clients_order = np.array(clients_order)
        np.random.shuffle(clients_order)
        clients_order = clients_order.reshape(self.round_no, -1)
        return clients_order

    def train_model(self, fl_clients, is_attack=False):
        clients_order = self.get_client_order()
        client_train_no = int(self.client_no * self.client_ratio)
        experiment_results = list()

        for client_order in range(self.round_no):
            client_model_params = list()
            training_example_no_set = list()
            chosen_clients = clients_order[client_order]

            try:
                for chosen_client_index in chosen_clients:
                    local_model_param, example_no = \
                        fl_clients[chosen_client_index].train_model(
                            self.global_model, self.epoch_no, self.lr,
                            clip_norm=self.sys_args.clip_norm,
                            center_radius=self.center_radius_stats)
                    client_model_params.append(local_model_param)
                    training_example_no_set.append(example_no)
                        # print("Info: Round %s - Client %s finish training." %
                        #       (client_order, chosen_client_index))
            except:
                exp_details = {"perturb": self.sys_args.perturb_mechanism,
                               "privacy budget": self.sys_args.privacy_budget,
                               "broken probability": self.sys_args.broken_probability,
                               "noise dist": self.sys_args.noise_dist,
                               "epoch": self.sys_args.epoch_no,
                               "batch size": self.sys_args.batch_size,
                               "lr": self.sys_args.lr,
                               "dataset": self.sys_args.dataset,
                               "model name": self.sys_args.model_name,
                               "clip norm": self.sys_args.clip_norm,
                               "is iid": self.sys_args.is_iid,
                               "is balanced": self.sys_args.is_balanced,
                               "results": experiment_results}
                with open(consts.EXP_RESULT_DIR, "a") as f:
                    json.dump(exp_details, f, indent=4)
                exit(1)


            # launch inverting gradient attack
            if client_order in self.attack_rounds and is_attack:
                print("Launch inverting attack:")
                target_client_id = self.select_attack_targets()
                ground_truth, labels = \
                    self.init_target_example(fl_clients[target_client_id])
                print("Attack %s: client %s" % (client_order + 1,
                                                target_client_id))
                recon_result = self.invert_gradient_attack(
                    fl_clients[target_client_id], ground_truth, labels)
                self.save_reconstruction_example(
                    ground_truth, recon_result, labels)

            # update the global model parameters
            weight_keys = list(client_model_params[0].keys())
            fed_state_dict = collections.OrderedDict()
            training_example_no = sum(training_example_no_set)
            for key in weight_keys:
                key_sum = 0
                for k in range(client_train_no):
                    client_data_ratio = \
                        training_example_no_set[k] / training_example_no
                    key_sum += client_data_ratio * client_model_params[k][key]
                fed_state_dict[key] = key_sum
            self.global_model.load_state_dict(fed_state_dict)
            # self.get_center_radius_of_model()

            with torch.no_grad():
                if (client_order+1) % 5 == 0:
                    acc = self.compute_accuracy()
                    experiment_results.append(acc.tolist())
                    print("Round %s: Accuracy %.2f%%" %
                          (client_order+1, acc * 100))
            # except:
            ##    print("Warn: zero is 0")
            #    continue

        exp_details = {"perturb": self.sys_args.perturb_mechanism,
                       "privacy budget": self.sys_args.privacy_budget,
                       "broken probability": self.sys_args.broken_probability,
                       "noise dist": self.sys_args.noise_dist,
                       "epoch": self.sys_args.epoch_no,
                       "batch size": self.sys_args.batch_size,
                       "lr": self.sys_args.lr,
                       "dataset": self.sys_args.dataset,
                       "model name": self.sys_args.model_name,
                       "clip norm": self.sys_args.clip_norm,
                       "is iid": self.sys_args.is_iid,
                       "is balanced": self.sys_args.is_balanced,
                       "results": experiment_results}
        with open(consts.EXP_RESULT_DIR, "a") as f:
            json.dump(exp_details, f, indent=4)

    def get_model_shape(self):
        if self.global_model is None:
            print("Error: The local model is Null!")
            exit(1)
        else:
            self.model_shape = dict()
            origin_model = MetaMonkey(self.global_model)
            for name, param in origin_model.parameters.items():
                self.model_shape[name] = param.shape

    def get_center_radius_of_model(self):
        self.center_radius_stats = dict()
        weights = copy.deepcopy(self.global_model.state_dict())
        for name, params in self.model_shape.items():
            self.center_radius_stats[name] = list()
            self.center_radius_stats[name] = \
                self.get_center_radius_of_vector(weights[name])
        #print("Info: Success to Update the center and the radius of the "
        #      "weights in the model.")

    def get_center_radius_of_vector(self, value_vector):
        """
        :param value_vector: tensor array
        :return:
        """
        max_value = value_vector.max()
        min_value = value_vector.min()
        radius_v = (max_value - min_value) / 2.0
        center_v = min_value + radius_v
        return (center_v, radius_v)

    def present_network_structure(self):
        paras = list(self.global_model.parameters())
        print("------------- Model Structure -------------")
        for num, para in enumerate(paras):
            para_size = para.size()
            print("%s: %s" % (num, para_size))
        print("------------------- END -------------------")

    def compute_accuracy(self):
        pred_r = torch.argmax(self.global_model(self.test_examples), dim=-1)
        return sum(pred_r == self.test_labels)/self.test_example_no

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

    def init_target_example(self, target_client):
        if self.sys_args.demo_target:  # demo image
            # Specify PIL filter for lower pillow versions
            ground_truth = torch.as_tensor(
                np.array(Image.open(consts.ATTACK_EXAMPLE_DEMO_DIR).resize(
                    (self.example_row_pixel, self.example_column_pixel),
                    Image.BICUBIC)) / 255, **self.sys_setup
            )
            ground_truth = ground_truth.permute(2, 0, 1).sub(self.dm) \
                .div(self.ds).unsqueeze(0).contiguous()
            if not self.sys_args.label_flip:
                label = torch.as_tensor((1,), device=self.sys_setup["device"])
            else:
                label = torch.as_tensor((5,), device=self.sys_setup["device"])
        else:
            target_example_id = np.random.randint(target_client.example_no)
            ground_truth, label = \
                target_client.get_example_by_index(target_example_id)
            if self.sys_args.label_flip:
                label = (label + 1) % self.class_no
            ground_truth, label = (
                ground_truth.unsqueeze(0).to(**self.sys_setup),
                torch.as_tensor((label,), device=self.sys_setup["device"]),
            )
        return ground_truth, label

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

    def invert_gradient_attack(self, target_client, target_ground_truth,
                               target_labels):
        local_gradient_steps = self.sys_args.accumulation
        local_lr = 1e-4
        updated_parameters = target_client.compute_gradient_by_opt(
            self.global_model, target_ground_truth, target_labels)

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
            updated_parameters, target_labels, img_shape=self.example_shape,
            dryrun=self.sys_args.dryrun)

        # Compute stats
        test_mse = (output - target_ground_truth).pow(2).mean().item()
        feat_mse = (self.global_model(output) - self.global_model(
            target_ground_truth)).pow(2).mean().item()
        test_psnr = \
            metrics.psnr(output, target_ground_truth, factor=1 / self.ds)

        print("Test Mse: %s, Feat Mse: %s, Test Psnr: %s" %
              (test_mse, feat_mse, test_psnr))
        return output

    def save_reconstruction_example(self, ground_truth, output, labels):
        # Save the resulting image
        if self.sys_args.save_image:
            os.makedirs(self.sys_args.image_path, exist_ok=True)
            output_denormalized = torch.clamp(output * self.ds + self.dm, 0, 1)
            rec_filename = (
                f'Re_{labels}_{self.sys_args.model_name}'
                f'_{self.sys_args.cost_fn}.png'
            )
            torchvision.utils.save_image(output_denormalized, os.path.join(
                self.sys_args.image_path, rec_filename))

            gt_denormalized = torch.clamp(
                ground_truth * self.ds + self.dm, 0, 1)
            gt_filename = f"Int_{labels}_ground_truth.png"
            torchvision.utils.save_image(gt_denormalized, os.path.join(
                self.sys_args.image_path, gt_filename))
