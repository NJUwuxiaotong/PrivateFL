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
MNIST_MLP_MODEL = "mlp"
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

# attack example demo path
ATTACK_EXAMPLE_DEMO_DIR = \
    "C:\\workspace\\workspace\\projects\\PrivateFL\\LDP-FL\\attack\\" \
    "attack_example_demo.jpg"

# output file
OUTPUT_ROOT_DIR = \
    "C:\\workspace\workspace\\projects\\PrivateFL\\LDP-FL\\test_results\\"

"""Setup constants, ymmv."""
PIN_MEMORY = True
NON_BLOCKING = False
BENCHMARK = True
MULTITHREAD_DATAPROCESSING = 4

# dataset mean and std
CIFAR10_MEAN = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
CIFAR10_STD = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
CIFAR100_MEAN = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
CIFAR100__STD = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
MNIST_MEAN = (0.13066373765468597,)
MNIST_STD = (0.30810782313346863,)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
