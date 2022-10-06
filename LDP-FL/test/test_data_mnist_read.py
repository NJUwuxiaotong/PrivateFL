import torch

from constant import consts as const
from data_process.data_load import DatasetMnist, _build_mnist

mnist = DatasetMnist()
mnist.read_data()
# mnist.show_example(mnist.training_examples[0])

dataset_dir = const.DATASET_ROOT_DIR
trainset, validset = _build_mnist(dataset_dir)

print(len(trainset.dataset.classes))



