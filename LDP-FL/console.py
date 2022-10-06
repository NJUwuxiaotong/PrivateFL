import torch
import options
import utils

from constant import consts
from federated_learning import FedAvgServer, FedAvgClient
from optimization_strategy import training_strategy

from data_process.data_load \
    import construct_data_loaders, get_dataset, get_labels_from_loader
from data_process.data_dispatch import data_dispatcher

from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler

torch.backends.cudnn.benchmark = consts.BENCHMARK

# arguments
sys_args = options.options().parse_args()
sys_defs = training_strategy("conservative")
sys_defs.epochs = sys_args.epoch_no
if sys_args.deterministic:
    utils.set_deterministic()


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = utils.system_startup(sys_args)

    # load train and test dataset
    train_dataset, valid_dataset = get_dataset(sys_args.dataset)
    train_loader, train_info = \
        construct_data_loaders(train_dataset, batch_size=len(train_dataset),
                               shuffle=False)
    valid_loader, valid_info = \
        construct_data_loaders(valid_dataset, batch_size=sys_defs.batch_size,
                               shuffle=False)

    # construct examples' indexes that be allocated to clients
    # tensor: one dimension
    train_labels = get_labels_from_loader(train_loader)
    data_dispatch_index = data_dispatcher(sys_args.is_iid, sys_args.client_no,
                                          train_labels)

    # construct a server
    fl_server = FedAvgServer(sys_args, setup, valid_loader, valid_info,
                             valid_info.example_shape)

    # construct multiple clients
    fl_clients = list()
    for i in range(sys_args.client_no):
        client_data_subset = Subset(train_dataset, data_dispatch_index[0])
        client_train_loader, client_train_info = \
            construct_data_loaders(client_data_subset, sys_defs.batch_size)
        fl_client = FedAvgClient(sys_args, client_train_loader,
                                 client_train_info,
                                 client_train_info.example_shape,
                                 train_info.class_no)
        fl_clients.append(fl_client)

    # dispatch data to the clients
    fl_server.prepare_before_training()

    # train and attack model
    fl_server.train_model()
