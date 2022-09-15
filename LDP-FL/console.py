import torch
import options
import utils

from constant import consts
from federated_learning import FedAvgServer
from optimization_strategy import training_strategy


torch.backends.cudnn.benchmark = consts.BENCHMARK

# arguments
args = options.options().parse_args()
defs = training_strategy("conservative")
defs.epochs = args.epoch_no
if args.deterministic:
    utils.set_deterministic()


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = utils.system_startup(args)

    # get loss function for the given dataset
    # loss_fn = data_process.get_loss_fn(args.dataset)

    # construct a server and multiple clients
    fl_server = FedAvgServer(args, setup)

    # dispatch data to the clients
    fl_server.prepare_before_training()

    # train and attack model
    fl_server.train_model()
