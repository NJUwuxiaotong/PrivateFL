from constant import constants as const
from fl_framework.fl_models.fedavg_server import FedAvgServer


# const.MNIST_MLP_MODEL, const.MNIST_CNN_MODEL, const.ResNet18_MODEL
model_type = const.ResNet18_MODEL

# params:

fedavg = FedAvgServer(100, 0.1, model_type, is_iid=False, round_no=3000,
                      epoch_no=5, lr=0.001)
fedavg.prepare_data()
fedavg.data_dispatcher()
fedavg.initial_model(5, 1, 2, [1, 32, 64])
fedavg.init_client_models()
fedavg.train_model()
