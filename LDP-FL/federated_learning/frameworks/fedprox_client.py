import collections
import torch

from copy import deepcopy
from loss import Classification
from attack.modules import MetaMonkey


class FedProxClient(object):
    def __init__(self, model_type, data_loader, data_info, example_shape,
                 class_no, loss_fn):
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

    def training_model(self, setup, global_model, epoch_no=10, lr=0.001,
                       mu_t=0.5, weight_decay=5e-4):
        self.local_model = deepcopy(global_model)
        self.local_model.to(**setup)
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr,
                              weight_decay=weight_decay)
        for epoch in range(epoch_no):
            for step, (examples, labels) in enumerate(self.data_loader):
                examples = examples.to(setup["device"])
                labels = labels.to(setup["device"])
                pred_labels = self.local_model(examples)
                opt.zero_grad()

                proximal_term = 0.0
                for w, w_t in zip(self.local_model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = self.loss_fn(pred_labels, labels) + (mu_t / 2) * proximal_term

                loss.backward()
                opt.step()

        return self.local_model.state_dict(), self.data_info.example_no

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
