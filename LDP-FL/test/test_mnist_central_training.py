from constant import consts as const
from federated_learning.models.mnist_central_training import MNISTCentralGen


output_file_dir = const.OUTPUT_ROOT_DIR + "MNIST_2NN_Neuron200_LR0001"

# test model:
#             const.MNIST_CNN_MODEL,
#             const.MNIST_MLP_MODEL,
#             const.ResNet18_MODEL
test_model_name = const.MNIST_MLP_MODEL

# training type: const.MINI_BATCH_GD, const.STOCHASTIC_GD
test_training_type = const.MINI_BATCH_GD

mnist_training = MNISTCentralGen(test_model_name, 200, 1500)
mnist_training.prepare_data()

if test_model_name == const.MNIST_MLP_MODEL:
    mnist_training.initial_model()
elif test_model_name == const.MNIST_CNN_MODEL:
    mnist_training.initial_model(5, 1, 2, [1, 32, 64])
elif test_model_name == const.ResNet18_MODEL:
    mnist_training.initial_model()
else:
    exit(1)

mnist_training.model_parameter_no()

if test_training_type == const.STOCHASTIC_GD:
    mnist_training.training_model(1, output_file_dir)
elif test_training_type == const.MINI_BATCH_GD:
    batch_size = 50
    mnist_training.training_model(batch_size, output_file_dir)
