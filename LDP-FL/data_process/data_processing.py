from loss import Classification, PSNR


def get_loss_fn(dataset):

    if dataset.upper() in ["CIFAR10", "CIFAR100", "MNIST", "MNIST_GRAY", "ImageNet"]:
        loss_fn = Classification()
    elif dataset in ["BSDS-SR", "BSDS-DN", "BSDS-RGB"]:
        loss_fn = PSNR()

    return loss_fn
