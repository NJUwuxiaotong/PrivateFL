import collections
import copy
import math

import numpy as np
from scipy.stats import bernoulli

import torch

from constant import consts
from copy import deepcopy
from loss import Classification
from attack.modules import MetaMonkey

from pub_lib.pub_libs import bound, random_value_with_probs


class FedAvgClient(object):
    def __init__(self, sys_setup, model_type, data_loader, data_info, example_shape,
                 class_no, loss_fn, privacy_budget, training_no,
                 perturb_mechanism):
        self.sys_setup = sys_setup
        # epoch_no=10, lr=0.001
        self.model_type = model_type
        self.data_loader = data_loader
        self.data_info = data_info
        self.example_shape = example_shape
        self.label_unique_no = class_no
        self.example_no = data_info.example_no

        self.channel_no = example_shape[0]
        self.training_row_pixel = example_shape[1]
        self.training_column_pixel = example_shape[2]

        self.local_model = None
        self.model_shape = None
        self.loss_fn = loss_fn
        self.epoch_total_loss = 0.0

        self.perturb_mechanism = perturb_mechanism
        self.privacy_budget = privacy_budget
        self.training_no = training_no
        self.single_privacy_cost = (privacy_budget + 0.0) / self.training_no

    def get_example_by_index(self, example_id):
        return self.data_loader.dataset[example_id]

    def training_model(self, global_model, epoch_no=10, lr=0.001,
                       weight_decay=5e-4):
        self.local_model = deepcopy(global_model)
        self.local_model.to(**self.sys_setup)
        self.get_model_shape()

        if self.perturb_mechanism == consts.NO_PERTURB:
            return self.training_model_without_noise(
                epoch_no, lr, weight_decay)
        elif self.perturb_mechanism == consts.G_LAPLACE_PERTURB:
            return self.training_model_with_g_Lap_perturb(
                epoch_no, lr, weight_decay)
        elif self.perturb_mechanism == consts.FED_SEL:
            return self.train_model_with_fedsel(global_model, 10, 10, 10)
        elif self.perturb_mechanism == consts.G_GAUSSIAN_PERTURB:
            return None

    def training_model_without_noise(self, epoch_no=10,
                                     lr=0.001, weight_decay=5e-4):
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        for epoch in range(epoch_no):
            for step, (examples, labels) in enumerate(self.data_loader):
                examples = examples.to(self.sys_setup["device"])
                labels = labels.to(self.sys_setup["device"])
                pred_labels = self.local_model(examples)
                opt.zero_grad()
                loss = self.loss_fn(pred_labels, labels)
                loss.backward()
                opt.step()
        return self.local_model.state_dict(), self.data_info.example_no

    def training_model_with_g_Lap_perturb(self, epoch_no=10, lr=0.001,
                                          weight_decay=5e-4):
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        privacy_cost = \
            self.single_privacy_cost / epoch_no / len(self.data_loader)
        for epoch in range(epoch_no):
            for step, (examples, labels) in enumerate(self.data_loader):
                examples = examples.to(self.sys_setup["device"])
                labels = labels.to(self.sys_setup["device"])
                pred_labels = self.local_model(examples)
                opt.zero_grad()
                loss = self.loss_fn(pred_labels, labels)
                loss.backward()
                opt.step()

                noise_params = self.add_dynamic_value(
                    self.local_model, consts.LAPLACE_DIST, 1.0/privacy_cost)
                self.local_model.load_state_dict(noise_params)

        return self.local_model.state_dict(), self.data_info.example_no

    def train_model_with_fedsel(self, gradients, threshold, gradient_range,
                               total_dimen_no, epoch_no=10, lr=0.001,
                               weight_decay=5e-4):
        privacy_cost1 = 1.0 / 2 * self.single_privacy_cost
        privacy_cost2 = self.single_privacy_cost - privacy_cost1

        dimen_magnitude = list()
        for name, params in self.model_shape.items():
            if len(params) == 1:
                dimen_magnitude.append(params[0])
            elif len(params) == 2:
                dimen_magnitude.extend([params[1]]*params[0])
            else:
                print("Error: Dimensions > 2!")
                exit(1)

        dimen_no = len(dimen_magnitude)
        dimen_status_vector = np.array(dimen_magnitude)
        dimen_status_vector = dimen_status_vector.argsort

        dimen_probs = list()
        for i in range(len(dimen_magnitude)):
            dimen_index = dimen_status_vector.index(i)
            prob = math.exp(privacy_cost1 * (dimen_index + 1) /(dimen_no - 1) )
            dimen_probs.append(prob)

        prob_sum = sum(dimen_probs)
        dimen_probs = np.array(dimen_probs)
        dimen_probs /= prob_sum
        dimen_probs = dimen_probs.tolist()

        chosen_dimens_index = random_value_with_probs(dimen_probs)
        return chosen_dimens_index

    def train_model_with_ldp_fl(self, center_radius_of_weights):
        """
        :param range1: the centers of the range of every layer
        :param range2: the radius of the range of every layer
        :return:
        """
        # train the model on the local data
        origin_model = MetaMonkey(self.local_model)
        model_params = copy.deepcopy(origin_model.state_dict())

        for name, params in self.model_shape.items():
            if len(params) == 1:
                for i in range(params[0]):
                    model_params[name][i] = \
                        self.bernoulli_noise(
                            model_params[name][i], self.single_privacy_cost,
                            center_radius_of_weights[name][i][0],
                            center_radius_of_weights[name][i][1])
            elif len(params) == 2:
                for i in range(params[0]):
                    for j in range(params[1]):
                        model_params[name][i][j] = \
                            self.bernoulli_noise(
                                model_params[name][i][j],
                                self.single_privacy_cost,
                                center_radius_of_weights[name][i][j][0],
                                center_radius_of_weights[name][i][j][1])
            else:
                print("Error: !")
                exit(1)
            self.local_model.load_state_dict(model_params)

    def bernoulli_noise(self, weight, privacy_budget, center_v,
                                 radius_v):
        prob = ((weight - center_v)*(math.exp(privacy_budget) - 1) +
                radius_v *(math.exp(privacy_budget) + 1)) / \
               (2*radius_v*(math.exp(privacy_budget) + 1))
        random_v = bernoulli.rvs(prob)
        if random_v == 1:
            return center_v + \
                   radius_v * (math.exp(privacy_budget) + 1) / \
                   (math.exp(privacy_budget) - 1)
        else:
            return center_v - \
                   radius_v * (math.exp(privacy_budget) + 1) / \
                   (math.exp(privacy_budget) - 1)

    def train_model_with_dssgd(self, gradients, threshold, gradient_range,
                               total_gradient_no, epoch_no=10, lr=0.001,
                               weight_decay=5e-4):
        pre_privacy_cost = self.single_privacy_cost * 8.0 / 9
        perturb_privacy_cost = self.single_privacy_cost * 1.0 / 9
        noise1 = np.random.laplace(
            2.0 * total_gradient_no * 1 / pre_privacy_cost)
        upload_gradients = list()

        gradient_no = 0
        while gradient_no <= total_gradient_no:
            chosen_layer_name, chosen_pos = self.selected_gradient_pos()
            if len(chosen_pos) == 1:
                chosen_gradient = gradients[chosen_layer_name][chosen_pos[0]]
            else:
                chosen_gradient = \
                    gradients[chosen_layer_name][chosen_pos[0]][chosen_pos[1]]
            noise2 = np.random.laplace(
                2 * 2 * total_gradient_no * 1 / pre_privacy_cost)
            if np.abs(bound(chosen_gradient, gradient_range)) + noise2 \
                    >= threshold + noise1:
                noise = np.random.laplace(
                    2 * total_gradient_no * 1 / perturb_privacy_cost)
                upload_gradients.append(
                    (chosen_layer_name, chosen_pos,
                     bound(chosen_gradient + noise, gradient_range) ))
                gradient_no = gradient_no + 1
        return upload_gradients

    def train_model_Local_update(self, gradients, threshold, gradient_range,
                               total_gradient_no, top_k, epoch_no=10, lr=0.001,
                               weight_decay=5e-4):
        # model gradient
        for name, params in gradients.items():
            if len(params) == 1:
                values = gradients[name][params]
                value_top_k = values.sort().values[top_k]
                gradients[name][ values < value_top_k ] = 0
            elif len(params) == 2:
                for i in range(len(params[0])):
                    values = gradients[name][i]
                    value_top_k = values.sort().values[top_k]
                    gradients[name][values < value_top_k] = 0
            else:
                print("Error: !")
                exit(1)

    def model_l2_norm(self, model_params):
        l2_norm = 0.0
        for name, params in model_params.items():
            l2_norm += torch.norm(model_params[name], p=2)
        return l2_norm

    def selected_gradient_pos(self):
        chosen_layer = np.random.randint(0, len(self.model_shape))
        layer_names = list(self.model_shape.keys())
        chosen_layer_name = layer_names[chosen_layer]

        layer_shape = self.model_shape[chosen_layer_name]
        if len(layer_shape) == 1:
            chosen_pos = (np.random.randint(0, layer_shape[0]))
        elif len(layer_shape) == 2:
            pos = np.random.randint(0, layer_shape[0] * layer_shape[1])
            first_layer = int(pos / layer_shape[1])
            second_layer = pos % layer_shape[1]
            chosen_pos = (first_layer, second_layer)
        else:
            print("Error: The shape of the model is not in [1, 2]!")
            exit(1)
        return chosen_layer_name, chosen_pos

    def get_model_shape(self):
        if self.local_model is None:
            print("Error: The local model is Null!")
            exit(1)
        else:
            self.model_shape = dict()
            origin_model = MetaMonkey(self.local_model)
            for name, param in origin_model.parameters.items():
                self.model_shape[name] = param.shape

    def add_constant_to_value(self, local_model, value):
        origin_model = MetaMonkey(local_model)
        with torch.no_grad():
            updated_params = collections.OrderedDict(
                (name, param + value)
                for (name, param) in origin_model.parameters.items())
        return updated_params

    def add_dynamic_value(self, local_model, noise_dist, lap_sigma=None,
                          gauss_sigma=None):
        origin_model = MetaMonkey(local_model)
        updated_parames = list()
        with torch.no_grad():
            for name, param in origin_model.parameters.items():
                param_shape = param.shape
                new_tensor = copy.deepcopy(param)

                if len(param_shape) == 1:
                    noises = self.generate_noise(
                        noise_dist, lap_sigma, param_shape[0])
                    noises = torch.tensor(
                        noises, device=self.sys_setup["device"])
                elif len(param_shape) == 2:
                    noises = self.generate_noise(
                        noise_dist, lap_sigma, param_shape[0] * param_shape[1])
                    noises = torch.tensor(
                        noises, device=self.sys_setup["device"])\
                        .reshape((param_shape[0], param_shape[1]))
                else:
                    print("exceed")
                    exit(1)

                new_tensor = new_tensor + noises
                updated_parames.append((name, new_tensor))
        return collections.OrderedDict(updated_parames)

    def generate_noise(self, noise_dist, lap_sigma, noise_no):
        if noise_dist == consts.LAPLACE_DIST:
            return np.random.laplace(lap_sigma, 1, noise_no)
        elif noise_dist == consts.GAUSSIAN_DIST:
            return np.random.normal(0, lap_sigma, noise_no)
        else:
            print("No distribution %s" % noise_dist)
            exit(1)

    def compute_gradient_by_autograd(self, global_model, ground_truth, labels):
        local_model = deepcopy(global_model)
        loss_fn = Classification()
        local_model.zero_grad()
        target_loss, _, _ = loss_fn(local_model(ground_truth), labels)

        # compute the gradients based on loss, parameters
        input_gradient = torch.autograd.grad(
            target_loss, local_model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        return input_gradient

    def compute_gradient_by_opt(self, global_model, ground_truth, labels):
        local_model = deepcopy(global_model)
        local_lr = 1e-4
        updated_parameters = self.stochastic_gradient_descent(
            local_model, ground_truth, labels, 1, lr=local_lr)
        updated_gradients = self.compute_updated_parameters(
            global_model, updated_parameters, local_lr)
        updated_gradients = [p.detach() for p in updated_gradients]
        return updated_gradients

    def stochastic_gradient_descent(self, local_model, training_examples,
                                    training_labels,
                                    epoch_no=10, lr=0.001):
        opt = torch.optim.SGD(local_model.parameters(), lr)
        for epoch in range(epoch_no):
            pred_label = local_model(training_examples)
            loss = self.loss_fn(pred_label, training_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return local_model.state_dict()

    def compute_updated_parameters(self, global_model, updated_parameters,
                                   local_lr):
        patched_model = MetaMonkey(global_model)
        patched_model_origin = deepcopy(patched_model)
        patched_model.parameters = collections.OrderedDict(
            (name, (param - param_origin)/local_lr)
            for ((name, param), (name_origin, param_origin))
            in zip(patched_model_origin.parameters.items(),
                   updated_parameters.items()))
        return list(patched_model.parameters.values())
