import collections
import copy
import math
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

import torch

from constant import consts
from copy import deepcopy
from loss import Classification
from attack.modules import MetaMonkey

from pub_lib.pub_libs import bound, random_value_with_probs, gradient_l2_norm, \
    model_l2_norm


class FedAvgClient(object):
    def __init__(self, sys_setup, model_type, data_loader, data_info,
                 example_shape, class_no, loss_fn, privacy_budget, training_no,
                 perturb_mechanism, noise_dist, broken_prob):
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
        self.layer_weight_no = None
        self.model_shape = None
        self.model_shape_name = None
        self.loss_fn = loss_fn
        self.epoch_total_loss = 0.0

        self.perturb_mechanism = perturb_mechanism
        self.privacy_budget = privacy_budget
        self.broken_prob = broken_prob
        self.training_no = training_no
        self.single_privacy_cost = (privacy_budget + 0.0) / self.training_no
        self.noise_dist = noise_dist

    def get_example_by_index(self, example_id):
        return self.data_loader.dataset[example_id]

    def train_model(self, global_model, epoch_no=10, lr=0.001,
                    clip_norm=None, center_radius=None):
        self.local_model = deepcopy(global_model)
        self.local_model.to(**self.sys_setup)
        self.get_model_shape()
        self.get_layer_weight_no()

        if self.perturb_mechanism in consts.ALGs_GradSGD_OPT:
            return self.train_model_with_gradient_sgd(
                epoch_no, lr, clip_norm, None, self.privacy_budget,
                self.broken_prob, self.perturb_mechanism)
        elif self.perturb_mechanism in consts.ALGs_GradBatchOPT:
            return self.train_model_with_gradient_mini_batch_gd(
                epoch_no, lr, clip_norm, None, self.privacy_budget,
                self.broken_prob, self.perturb_mechanism)
        elif self.perturb_mechanism in consts.ALGs_Weight_OPT:
            return self.train_model_with_weight(
                epoch_no, lr, None, self.privacy_budget, self.broken_prob,
                self.perturb_mechanism, center_radius)
        elif self.perturb_mechanism in consts.ALGs_Sample_OPT:
            return self.train_model_with_sample(
                global_model, epoch_no, lr, clip_norm, None,
                self.privacy_budget, self.broken_prob, self.perturb_mechanism)
        else:
            print("Error: Perturbation mechanism %s does not exist!"
                  % self.perturb_mechanism)
            exit(1)

    def train_model_with_gradient_sgd(self, epoch_no, lr, norm_bound, sigma,
                                      epsilon, delta, perturb_mec):
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        batch_no = len(self.data_loader)

        for epoch in range(epoch_no):
            chosen_batch_index = np.random.randint(0, batch_no)
            for step, (examples, labels) in enumerate(self.data_loader):
                if step == chosen_batch_index:
                    examples = examples.to(self.sys_setup["device"])
                    labels = labels.to(self.sys_setup["device"])
                    example_no = examples.shape[0]

                    gradients = []
                    for i in range(example_no):
                        pred_label = self.local_model(examples[i])
                        opt.zero_grad()
                        loss = self.loss_fn(pred_label[0], labels[i])
                        input_gradient = torch.autograd.grad(
                            loss, self.local_model.parameters())
                        input_gradient = \
                            [grad.detach() for grad in input_gradient]
                        gradients.append(input_gradient)

            updated_gradient = list()
            for layer_gradient in gradients[0]:
                updated_gradient.append(torch.zeros_like(layer_gradient))

            gradients_l2_norm = list()
            gradients_layers_l2_norm = list()
            for gradient in gradients:
                l2_norm, layer_l2_norm = gradient_l2_norm(gradient)
                gradients_l2_norm.append(l2_norm)
                gradients_layers_l2_norm.append(layer_l2_norm)

            # generate the noise
            max_dimen = 0
            for layer_name, layer_shape in self.model_shape.items():
                layer_dimen = 1
                for i in range(len(layer_shape)):
                    layer_dimen *= layer_shape[i]

                if max_dimen < layer_dimen:
                    max_dimen = layer_dimen

            #self.show_hist_of_gradients(gradients[0])

            if sigma is None:
                if perturb_mec == consts.ALG_rGaussAGrad16:
                    # Influence: (epsilon, delta) --> sigma
                    # Direct sigma: (1, 2, 3, ...)
                    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
                    sigma *= norm_bound
                elif perturb_mec == consts.ALG_eGaussAGrad19:
                    s = math.log(math.sqrt(2.0/math.pi)/delta)
                    sigma = (math.sqrt(s)+math.sqrt(s+epsilon))/(math.sqrt(2)*epsilon)
                elif perturb_mec == consts.ALG_eGaussAGrad22:
                    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
                elif perturb_mec == consts.ALG_rGaussPGrad22:
                    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon / math.sqrt(0.1)
                else:
                    # consts.ALG_NoGradSGD
                    sigma = 1.0

            noise = self.generate_noise(consts.GAUSSIAN_DIST, sigma, max_dimen)
            noise99 = self.value99(noise)
            gradient99 = self.gradients99(gradients[0])

            noise /= ( 5 * noise99/gradient99)
            noise = torch.tensor(noise, device=self.sys_setup["device"])

            noise_bound = gradient99
            greater_0025_index = noise > noise_bound
            less_0025_index = noise < -1 * noise_bound
            noise[greater_0025_index] = noise_bound
            noise[less_0025_index] = -1 * noise_bound

            for i in range(example_no):
                for j in range(len(updated_gradient)):
                    if perturb_mec == consts.ALG_NoGradSGD:
                        updated_gradient[j] += gradients[i][j]
                    else:
                        # norm bound: 1, 2, 3, 4, 5, 6, 7, ...
                        updated_gradient[j] = \
                            updated_gradient[j] + \
                            gradients[i][j] * \
                            min(1, norm_bound / gradients_l2_norm[i])
                        vector_shape = updated_gradient[j].shape
                        # import pdb; pdb.set_trace()
                        updated_gradient[j] += noise[:vector_shape[-1]]

            for i in range(len(updated_gradient)):
                updated_gradient[i] /= example_no

            # self.show_hist_of_gradients(updated_gradient)
            local_model_params = self.local_model.state_dict()
            with torch.no_grad():
                for i in range(len(self.model_shape_name)):
                    local_model_params[self.model_shape_name[i]] -= \
                        lr * updated_gradient[i]
                self.local_model.load_state_dict(local_model_params)
        return self.local_model.state_dict(), self.example_no

    def value99(self, noise):
        g_noise = len(noise[noise>=0])
        n_noise = len(noise[noise<0])

        if g_noise > n_noise:
            noise = noise[noise>=0].tolist()
        else:
            noise = noise[noise<0] * -1.0
            noise = noise.tolist()

        noise.sort()
        value = noise[int(0.99*len(noise))]
        return value

    def gradients99(self, gradient):
        values = list()
        for g in gradient:
            g = g.reshape((1, -1))[0].tolist()
            values.extend(g)

        values = np.array(values)
        values = values[values >= 0].tolist()
        values.sort()
        value = values[int(0.99*len(values))]
        return value

    def params99(self, model_params):
        model_params = copy.deepcopy(model_params)
        values = list()
        for name, params in model_params.items():
            param = params.reshape((1, -1))[0].tolist()
            values.extend(param)

        values = np.array(values)

        g_values = len(values[values >= 0])
        n_values = len(values[values<0])

        if g_values > n_values:
            values = values[values>=0].tolist()
        else:
            values = values[values<0] * -1.0
            values = values.tolist()

        values.sort()
        value99 = values[int(0.99 * len(values))]
        return value99

    def train_model_with_gradient_mini_batch_gd(self, epoch_no, lr, norm_bound,
                                                sigma, epsilon, delta,
                                                perturb_mec):
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        for epoch in range(epoch_no):
            gradients = []
            for step, (examples, labels) in enumerate(self.data_loader):
                examples = examples.to(self.sys_setup["device"])
                labels = labels.to(self.sys_setup["device"])
                pred_labels = self.local_model(examples)
                opt.zero_grad()
                loss = self.loss_fn(pred_labels, labels)
                input_gradient = torch.autograd.grad(
                    loss, self.local_model.parameters())
                input_gradient = \
                    [grad.detach() for grad in input_gradient]
                gradients.append(input_gradient)

            updated_gradient = list()
            for layer_gradient in gradients[0]:
                updated_gradient.append(torch.zeros_like(layer_gradient))

            gradients_l2_norm = list()
            gradients_layers_l2_norm = list()
            for gradient in gradients:
                l2_norm, layer_l2_norm = gradient_l2_norm(gradient)
                gradients_l2_norm.append(l2_norm)
                gradients_layers_l2_norm.append(layer_l2_norm)

            # generate the noise
            max_dimen = 0
            for layer_name, layer_shape in self.model_shape.items():
                if max_dimen < layer_shape[-1]:
                    max_dimen = layer_shape[-1]

            if sigma is None:
                if perturb_mec == consts.ALG_rGaussAGrad18:
                    # Influence: (epsilon, delta) --> sigma
                    # Direct sigma: (1, 2, 3, ...)
                    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
                else:
                    sigma = 1.0

            noise = self.generate_noise(consts.GAUSSIAN_DIST, sigma, max_dimen)
            noise = torch.tensor(noise, device=self.sys_setup["device"])

            noise99 = self.value99(noise)
            gradient99 = self.gradients99(gradients[0])
            noise /= ( 5 * noise99/gradient99)

            # import pdb; pdb.set_trace()
            # noise = torch.tensor(noise, device=self.sys_setup["device"])

            noise_bound = gradient99
            greater_0025_index = noise > noise_bound
            less_0025_index = noise < -1 * noise_bound
            noise[greater_0025_index] = noise_bound
            noise[less_0025_index] = -1 * noise_bound

            for i in range(len(self.data_loader)):
                for j in range(len(updated_gradient)):
                    if perturb_mec == consts.ALG_No_GradBatch:
                        updated_gradient[j] += gradients[i][j]
                    else:
                        updated_gradient[j] = \
                            updated_gradient[j] + gradients[i][j] * \
                            min(1, gradients_layers_l2_norm[i][j] * 0.75 /
                                gradients_l2_norm[i])
                        vector_shape = updated_gradient[j].shape
                        # import pdb; pdb.set_trace()
                        updated_gradient[j] += noise[:vector_shape[-1]]

            for i in range(len(updated_gradient)):
                updated_gradient[i] /= len(self.data_loader)

            local_model_params = self.local_model.state_dict()
            with torch.no_grad():
                for i in range(len(self.model_shape_name)):
                    local_model_params[self.model_shape_name[i]] -= \
                        lr * updated_gradient[i]
                self.local_model.load_state_dict(local_model_params)
        return self.local_model.state_dict(), self.data_info.example_no

    def train_model_with_weight(self, epoch_no, lr, sigma, epsilon, delta,
                                perturb_mec, center_radius_stats=None):
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

                if perturb_mec == consts.ALG_bGaussAWeig21:
                    if sigma is None:
                        sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

                    updated_params = self.add_dynamic_noise_to_model(
                        self.local_model, consts.GAUSSIAN_DIST, sigma)

                    with torch.no_grad():
                        self.local_model.load_state_dict(updated_params)

        if perturb_mec == consts.ALG_rGaussAWeig19:
            if self.noise_dist == consts.LAPLACE_DIST:
                sigma = 1 / epsilon
                updated_params = self.add_dynamic_noise_to_model(
                    self.local_model, consts.LAPLACE_DIST, sigma)
            elif self.noise_dist == consts.GAUSSIAN_DIST:
                sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
                updated_params = self.add_dynamic_noise_to_model(
                    self.local_model, consts.GAUSSIAN_DIST, sigma)
            else:
                print("Error: No distribution %s" % self.noise_dist)
                exit(1)
            with torch.no_grad():
                self.local_model.load_state_dict(updated_params)

        if perturb_mec == consts.ALG_rRResAWeig21:
            if center_radius_stats is None:
                print("Error: Perturb Mechanism %s needs central and radius "
                      "info!" % perturb_mec)
                exit(1)

            updated_params = self.add_bernoulli_noise_to_model(
                self.local_model, center_radius_stats, epsilon)
            with torch.no_grad():
                self.local_model.load_state_dict(updated_params)

        return self.local_model.state_dict(), self.data_info.example_no

    def train_model_with_sample(self, global_model, epoch_no, lr, norm_bound,
                                sigma, epsilon, delta, perturb_mec):

        global_model_params = copy.deepcopy(global_model.state_dict())
        init_model_params = copy.deepcopy(self.local_model.state_dict())
        updated_model_params, example_no = \
            self.train_model_with_gradient_sgd(epoch_no, lr, None, sigma,
                                               epsilon, delta,
                                               consts.ALG_NoGradSGD)

        updated_gradients = self.compute_updated_parameters(
            init_model_params, updated_model_params, lr)

        if perturb_mec == consts.ALG_rLapPGrad15:
            noise_gradients = self.train_model_with_rLapPGrad15(
                updated_gradients, -0.0001, 1, 0.1)
        elif perturb_mec == consts.ALG_rExpPWeig20:
            noise_gradients = self.train_model_with_rExpPWeig20(
                updated_gradients, self.privacy_budget, 0.5)
        else: # consts.ALG_rGaussPGrad22:
            noise_gradients = self.train_model_with_rGaussPGrad22(
                updated_gradients, 0.2, norm_bound, sigma, epsilon, delta,
                self.perturb_mechanism)

        with torch.no_grad():
            layer_index = 0
            for name, params in global_model_params.items():
                global_model_params[name] -= lr * noise_gradients[layer_index]
                layer_index += 1
            return global_model_params, example_no

    def train_model_with_rLapPGrad15(self, gradients, threshold, gradient_range,
                                     total_gradient_ratio):
        pre_privacy_cost = self.privacy_budget * 8.0 / 9
        perturb_privacy_cost = self.privacy_budget * 1.0 / 9
        
        total_param_no = 0
        for name, params in self.model_shape.items():
            layer_param_no = 1
            for dimen in params:
                layer_param_no *= dimen
            total_param_no += layer_param_no
        
        total_gradient_no = int(total_param_no * total_gradient_ratio)

        noise1 = np.random.laplace(2.0 * 1 / pre_privacy_cost) / 1000
            # 2.0 * total_gradient_no * 1 / pre_privacy_cost)
        upload_gradients = list()

        # change the gradients to one dimension
        noise_gradient = list()
        for name, layer_gradient in gradients:
            g = torch.zeros_like(layer_gradient)
            g = g.reshape((1, -1))[0]
            noise_gradient.append(g)

        tmp_gradient = list()
        for name, layer_gradient in gradients:
            tmp = copy.deepcopy(layer_gradient)
            tmp = tmp.reshape((1, -1))[0]
            tmp_gradient.append(tmp)

        gradient_no = 0
        chosen_gradient_pos_list = list()
        while gradient_no <= total_gradient_no:
            chosen_layer_index, chosen_pos = self.selected_gradient_pos()

            if (chosen_layer_index, chosen_pos) not in chosen_gradient_pos_list:
                chosen_gradient = tmp_gradient[chosen_layer_index][chosen_pos]
                noise2 = np.random.laplace(
                    2 * 2 * 1 / pre_privacy_cost) / 1000
                    # 2 * 2 * total_gradient_no * 1 / pre_privacy_cost)
                # import pdb; pdb.set_trace()
                if bound(chosen_gradient, gradient_range).abs_() + noise2 \
                        >= threshold + noise1:
                    noise = np.random.laplace(
                        2 / perturb_privacy_cost) / 1000
                        # 2 * total_gradient_no * 1 / perturb_privacy_cost)

                    noise_gradient[chosen_layer_index][chosen_pos] = \
                        bound(chosen_gradient + noise, gradient_range)
                    chosen_gradient_pos_list.append((chosen_layer_index, chosen_pos))
                    gradient_no = gradient_no + 1

        for i in range(len(noise_gradient)):
            noise_gradient[i] = noise_gradient[i].reshape(gradients[i][1].shape)
        return noise_gradient

    def train_model_with_rExpPWeig20(self, gradients, privacy_budget,
                                     total_dimen_ratio):

        privacy_cost1 = 1.0 / 2 * privacy_budget
        privacy_cost2 = privacy_budget - privacy_cost1

        dimens_dict = dict()          # e.g., {name: dim, name: dim, ...}
        for name, params in self.model_shape.items():
            dimen = 1
            for i in range(len(params)):
                dimen *= params[i]
            dimens_dict[name] = dimen

        dimen_list = list(dimens_dict.values())
        dimen_status_vector = np.array(dimen_list)
        dimen_status_vector = dimen_status_vector.argsort() # index small -> big
        dimen_status_vector = dimen_status_vector.tolist()

        dimen_probs = list()
        for i in range(len(dimen_list)):
            dimen_index = dimen_status_vector.index(i)
            prob = math.exp(privacy_cost1 *
                            (dimen_index + 1) /(dimen_list[i] - 1))
            dimen_probs.append(prob)

        prob_sum = sum(dimen_probs)
        dimen_probs = np.array(dimen_probs)
        dimen_probs /= prob_sum
        dimen_probs = dimen_probs.tolist()

        chosen_dimens_index = random_value_with_probs(
            dimen_probs, int(len(dimen_list)*total_dimen_ratio))

        noise_gradient = list()
        for name, layer_gradient in gradients:
            noise_gradient.append(torch.zeros_like(layer_gradient))

        sigma = math.sqrt(2 * math.log(1.25 / self.broken_prob)) / self.privacy_budget
        for dimen_index in chosen_dimens_index:
            noise = self.generate_noise(consts.GAUSSIAN_DIST,
                                        sigma,
                                        dimen_list[dimen_index])

            # import pdb; pdb.set_trace()
            noise_bound = 0.02
            greater_0025_index = noise > noise_bound
            less_0025_index = noise < -1 * noise_bound
            noise[greater_0025_index] = noise_bound
            noise[less_0025_index] = -1 * noise_bound

            noise_tensor = \
                torch.tensor(noise, device=self.sys_setup["device"]).reshape(
                    gradients[dimen_index][1].shape)
            noise_gradient[dimen_index] = noise_gradient[dimen_index] \
                                          + gradients[dimen_index][1] \
                                          + noise_tensor
        return noise_gradient

    def train_model_with_rGaussPGrad22(self, gradients, top_k_ratio, norm_bound,
                                       sigma, epsilon, delta, perturb_mec):

        gradients_list = [params for name, params in gradients]
        l2_norm, layer_l2_norm = gradient_l2_norm(gradients_list)

        dimens_dict = dict()          # e.g., {name: dim, name: dim, ...}
        for name, params in self.model_shape.items():
            dimen = 1
            for i in range(len(params)):
                dimen *= params[i]
            dimens_dict[name] = dimen

        if sigma is None:
            if perturb_mec == consts.ALG_rGaussPGrad22:
                # Influence: (epsilon, delta) --> sigma
                # Direct sigma: (1, 2, 3, ...)
                sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
                sigma  = sigma * norm_bound / math.sqrt(100)
            else:
                sigma = 1.0

        noisy_gradient = list()
        # for layer_gradient in gradients:
        #     noisy_gradient.append(torch.zeros_like(layer_gradient))
        for name, params in gradients:
            noise = self.generate_noise(consts.GAUSSIAN_DIST, sigma,
                                        dimens_dict[name])
            noise = torch.tensor(noise, device=self.sys_setup["device"])

            noise_bound = 0.2
            greater_0025_index = noise > noise_bound
            less_0025_index = noise < -1 * noise_bound
            noise[greater_0025_index] = noise_bound
            noise[less_0025_index] = -1 * noise_bound

            raw_values = copy.deepcopy(params)
            raw_values = raw_values.reshape(1, -1)[0]
            noise_values = copy.deepcopy(raw_values)

            param_len = len(raw_values)
            top_k = int(param_len * top_k_ratio)

            values = raw_values.tolist()
            values.sort()
            values.reverse()
            value_top_k = values[top_k]

            raw_values[raw_values < value_top_k] = 0.0
            noise[raw_values< value_top_k] = 0.0

            #import pdb; pdb.set_trace()
            noise_values = noise_values * min(1, norm_bound / l2_norm) +  noise
            noise_values = noise_values.reshape(params.shape)

            noisy_gradient.append(noise_values)
        return noisy_gradient

    def add_dynamic_noise_to_model(self, local_model, noise_dist, sigma):
        origin_model = copy.deepcopy(local_model.state_dict())

        # import pdb; pdb.set_trace()
        param99 = self.params99(origin_model)
        updated_parames = list()
        with torch.no_grad():
            for name, param in origin_model.items():
                noises = self.generate_noise(noise_dist, sigma,
                                             self.layer_weight_no[name])
                noises = torch.tensor(noises, device=self.sys_setup['device'])

                noise99 = self.value99(noises)
                noises /= (5* noise99 / param99)

                noise_bound = param99
                greater_0025_index = noises > noise_bound
                less_0025_index = noises < -1 * noise_bound
                noises[greater_0025_index] = noise_bound
                noises[less_0025_index] = -1 * noise_bound

                noises = noises.reshape(self.model_shape[name])
                updated_parames.append((name, param + noises))
        return collections.OrderedDict(updated_parames)

    def add_bernoulli_noise_to_model(self, local_model,
                                     center_radius_of_weights, epsilon):
        #origin_model = MetaMonkey(local_model)
        model_params = copy.deepcopy(local_model.state_dict())

        updated_parames = list()
        for name, params in model_params.items():
            layer_params = params.reshape((1, -1))[0]
            noise_layer_params = torch.zeros_like(layer_params)
            for i in range(len(noise_layer_params)):
                noise_layer_params[i] = self.bernoulli_noise(
                            layer_params[i], epsilon,
                            center_radius_of_weights[name][0],
                            center_radius_of_weights[name][1])

            # import pdb; pdb.set_trace()

            noise_layer_params = noise_layer_params.reshape(params.shape)
            updated_parames.append((name, noise_layer_params))

        return collections.OrderedDict(updated_parames)

    def bernoulli_noise(self, weight, privacy_budget, center_v,
                                 radius_v):
        prob = ((weight - center_v)*(math.exp(privacy_budget) - 1) +
                radius_v *(math.exp(privacy_budget) + 1)) / \
               (2*radius_v*(math.exp(privacy_budget) + 1))

        try:
            random_v = bernoulli.rvs(prob.tolist())
        except:
            random_v = 0
        if random_v == 1:
            return center_v + \
                   radius_v * (math.exp(privacy_budget) + 1) / \
                   (math.exp(privacy_budget) - 1)
        else:
            return center_v - \
                   radius_v * (math.exp(privacy_budget) + 1) / \
                   (math.exp(privacy_budget) - 1)

    def compute_updated_parameters(self, initial_parameters,
                                   updated_parameters, lr):
        # patched_model = MetaMonkey(initial_parameters)
        # patched_model_origin = deepcopy(patched_model)
        # patched_model.parameters = collections.OrderedDict(
        #    (name, (param - param_origin)/local_lr)
        #     for ((name, param), (name_origin, param_origin))
        #     in zip(updated_parameters.items(), initial_parameters.items()))
        updated_values = [(name, (param_origin - param) / lr)
                          for ((name, param), (name_origin, param_origin))
                          in zip(updated_parameters.items(),
                                 initial_parameters.items())]
        return updated_values

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

                noise_params = self.add_dynamic_noise_to_model(
                    self.local_model, consts.LAPLACE_DIST, 1.0/privacy_cost)
                self.local_model.load_state_dict(noise_params)

        return self.local_model.state_dict(), self.data_info.example_no

    def selected_gradient_pos(self):
        chosen_layer = np.random.randint(0, len(self.model_shape))
        layer_names = list(self.model_shape.keys())
        chosen_layer_name = layer_names[chosen_layer]

        layer_shape = self.model_shape[chosen_layer_name]

        layer_dimen = 1
        for i in range(len(layer_shape)):
            layer_dimen *= layer_shape[i]

        pos = np.random.randint(0, layer_dimen)
        #chosen_pos = list()
        #for i in range(len(layer_shape)):
        #    chosen_pos.append(pos % layer_shape[-1*(i+1)])
        #    pos = int(pos / layer_shape[-1*(i+1)])
        #
        #chosen_pos.reverse()
        return chosen_layer, pos

    def get_model_shape(self):
        if self.local_model is None:
            print("Error: The local model is Null!")
            exit(1)
        else:
            self.model_shape = dict()
            self.model_shape_name = list()
            for name, param in self.local_model.state_dict().items():
                self.model_shape[name] = param.shape
                self.model_shape_name.append(name)

    def add_constant_to_value(self, local_model, value):
        origin_model = MetaMonkey(local_model)
        with torch.no_grad():
            updated_params = collections.OrderedDict(
                (name, param + value)
                for (name, param) in origin_model.parameters.items())
        return updated_params

    def generate_noise(self, noise_dist, lap_sigma, noise_no):
        if noise_dist == consts.LAPLACE_DIST:
            return np.random.laplace(0, lap_sigma, noise_no)
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

    def show_hist_of_gradients(self, gradients):
        results = list()
        for i in range(len(gradients)):
            layer_gradients = gradients[i]
            layer_gradients = layer_gradients.reshape((1, -1))
            results.extend(layer_gradients[0].tolist())

        plt.hist(results, bins=1000)
        plt.show()

    def get_layer_weight_no(self):
        self.layer_weight_no = dict()
        for name, param_shape in self.model_shape.items():
            param_no = 1
            for param_n in param_shape:
                param_no *= param_n
            self.layer_weight_no[name] = param_no
