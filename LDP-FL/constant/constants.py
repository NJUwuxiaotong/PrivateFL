# related-data constants
DATASET_ROOT_DIR = "C:\\workspace\\workspace\\datasets\\"

DATASET_ADULT = "ADULT"
DATASET_BIRD = "BIRD"
DATASET_CIFR10 = "CIRF-10"
DATASET_MNIST = "MNIST"
DATASET_TCWWS = "TCWWS"
DATASET_NAMES = [DATASET_ADULT, DATASET_BIRD, DATASET_CIFR10, DATASET_MNIST,
                 DATASET_TCWWS]

DATASET_DIR = {
    DATASET_ADULT: DATASET_ROOT_DIR + "dataset_adult\\",
    DATASET_BIRD: DATASET_ROOT_DIR + "dataset_bird\\",
    DATASET_CIFR10: DATASET_ROOT_DIR + "dataset_cifar-10\\",
    DATASET_MNIST: DATASET_ROOT_DIR + "dataset_mnist\\",
    DATASET_TCWWS: DATASET_ROOT_DIR + "dataset_tcwws\\"}

MNIST_TRAIN_IMAGE_DIR = \
    DATASET_DIR[DATASET_MNIST] + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABEL_DIR = \
    DATASET_DIR[DATASET_MNIST] + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMAGE_DIR = DATASET_DIR[DATASET_MNIST] + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABEL_DIR = DATASET_DIR[DATASET_MNIST] + "t10k-labels-idx1-ubyte.gz"

# MNIST Model
MNIST_MLP_MODEL = "mnist mlp model"
MNIST_CNN_MODEL = "mnist cnn model"
ResNet18_MODEL = "resnet18 model"
ResNet34_MODEL = "resnet34 model"
ResNet50_MODEL = "resnet50 model"
ResNet101_MODEL = "resnet101 model"
ResNet152_MODEL = "resnet152 model"

# Training Batch
BATCH_GD = "batch gd"
STOCHASTIC_GD = "stochastic gd"
MINI_BATCH_GD = "mini batch gd"

# output file
OUTPUT_ROOT_DIR = \
    "C:\\workspace\workspace\\projects\\PrivateFL\\LDP-FL\\test_results\\"
