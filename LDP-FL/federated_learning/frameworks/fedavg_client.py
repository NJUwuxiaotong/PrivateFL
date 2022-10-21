import collections
import torch

from copy import deepcopy
from loss import Classification
from attack.modules import MetaMonkey


class FedAvgClient(object):
    def __init__(self, model_type, data_loader, data_info, example_shape,
                 class_no, loss_fn):
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
        self.loss_fn = loss_fn
        self.epoch_total_loss = 0.0

    def get_example_by_index(self, example_id):
        return self.data_loader.dataset[example_id]

    def training_model(self, setup, global_model, epoch_no=10, lr=0.001):
        self.local_model = deepcopy(global_model)
        self.local_model.to(**setup)
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        for epoch in range(epoch_no):
            for step, (examples, labels) in enumerate(self.data_loader):
                examples = examples.to(setup["device"])
                labels = labels.to(setup["device"])
                pred_labels = self.local_model(examples)
                loss = self.loss_fn(pred_labels, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self.local_model.state_dict(), self.data_info.example_no

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
