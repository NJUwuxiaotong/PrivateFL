# related-data constants
DATASET_ROOT_DIR = "C:\\workspace\\workspace\\datasets"

DATASET_ADULT = "ADULT"
DATASET_BIRD = "BIRD"
DATASET_CIFAR10 = "CIFAR10"
DATASET_MNIST = "MNIST"
DATASET_MNIST_GRAY = "MNIST-GRAY"
DATASET_CIFAR100 = "CIFAR100"
DATASET_IMAGENET = "IMAGENET"
DATASET_TCWWS = "TCWWS"
DATASET_NAMES = [DATASET_ADULT, DATASET_BIRD, DATASET_CIFAR10, DATASET_CIFAR100,
                 DATASET_MNIST, DATASET_MNIST_GRAY, DATASET_TCWWS,
                 DATASET_IMAGENET]

DATASET_DIR = {
    DATASET_ADULT: DATASET_ROOT_DIR + "dataset_adult\\",
    DATASET_BIRD: DATASET_ROOT_DIR + "dataset_bird\\",
    DATASET_CIFAR10: DATASET_ROOT_DIR + "dataset_cifar-10\\",
    DATASET_MNIST: DATASET_ROOT_DIR + "mnist\\",
    DATASET_TCWWS: DATASET_ROOT_DIR + "dataset_tcwws\\"}

MNIST_TRAIN_IMAGE_DIR = \
    DATASET_DIR[DATASET_MNIST] + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABEL_DIR = \
    DATASET_DIR[DATASET_MNIST] + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMAGE_DIR = DATASET_DIR[DATASET_MNIST] + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABEL_DIR = DATASET_DIR[DATASET_MNIST] + "t10k-labels-idx1-ubyte.gz"

# MNIST Model
MNIST_MLP_MODEL = "mnist mlp"
MNIST_CNN_MODEL = "mnist cnn"

# CIFAR-10
CIFAR10_CNN_MODEL = "cifar10 cnn"

# ResNet Model
ResNet18_MODEL = "resnet18"
ResNet34_MODEL = "resnet34"
ResNet50_MODEL = "resnet50"
ResNet101_MODEL = "resnet101"
ResNet152_MODEL = "resnet152"

# Training Batch
BATCH_GD = "batch gd"
STOCHASTIC_GD = "stochastic gd"
MINI_BATCH_GD = "mini batch gd"

# attack example demo path
ATTACK_EXAMPLE_DEMO_DIR = "attack/attack_example_demo.jpg"

# output file
OUTPUT_ROOT_DIR = "test_results/"

"""Setup constants, ymmv."""
PIN_MEMORY = True
NON_BLOCKING = False
BENCHMARK = True
MULTITHREAD_DATAPROCESSING = 4

# dataset mean and std
CIFAR10_MEAN = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
CIFAR10_STD = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
CIFAR100_MEAN = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
CIFAR100_STD = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
MNIST_MEAN = (0.13066373765468597,)
MNIST_STD = (0.30810782313346863,)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# perturbation mechanism
LAPLACE_DIST = "laplace"
GAUSSIAN_DIST = "gaussian"

NO_PERTURB = "no_perturb"
G_LAPLACE_PERTURB = "g_laplace"
G_GAUSSIAN_PERTURB = "g_gaussian"
FED_SEL = "fed_sel"

ALG_NoGradMiniBatch = "ALG_NoGaussMiniBatch"
ALG_rGaussAGrad16 = "ALG_rGaussAGrad16"
ALG_eGaussAGrad19 = "ALG_eGaussAGrad19"
ALG_eGaussAGrad22 = "ALG_eGaussAGrad22"

ALG_NoGradBatch = "ALG_NoGradBatch"
ALG_rGaussAGrad18 = "ALG_rGaussAGrad18"

ALG_bGaussAWeig21 = "ALG_bGaussAWeig21"
ALG_rGaussAWeig19 = "ALG_rGaussAWeig19"
ALG_rRResAWeig21 = "ALG_rRResAWeig21"

ALG_rLapPGrad15 = "ALG_rLapPGrad15"
ALG_rExpPWeig20 = "ALG_rExpPWeig20"
ALG_rGaussPGrad22 = "ALG_rGaussPGrad22"

ALGs_GradMiniBatchOPT = [ALG_rGaussAGrad16, ALG_eGaussAGrad19,
                         ALG_eGaussAGrad22, ALG_rGaussAGrad18,
                         ALG_NoGradMiniBatch]
ALGs_GradBatchOPT = [ALG_NoGradBatch, ALG_rGaussAGrad18]

ALGs_Weight_OPT = [ALG_bGaussAWeig21, ALG_rGaussAWeig19, ALG_rRResAWeig21]
ALGs_Sample_OPT = [ALG_rLapPGrad15, ALG_rExpPWeig20, ALG_rGaussPGrad22]
