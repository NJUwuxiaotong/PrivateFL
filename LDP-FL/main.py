import torch
import torchvision

import numpy as np
from PIL import Image

from collections import defaultdict
import datetime
import time
import os

import options

import torch

from constant import consts
from optimization_strategy import training_strategy

import utils

from attack.reconstruction_algorithms import GradientReconstructor

import data_process

from federated_learning import FedAvgServer, FedAvgClient


torch.backends.cudnn.benchmark = consts.BENCHMARK

# arguments
args = options.options().parse_args()
defs = training_strategy("conservative")
defs.epochs = args.epochs
if args.deterministic:
    utils.set_deterministic()


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = utils.system_startup(args)

    # get loss function for the given dataset
    loss_fn = data_process.get_loss_fn(args.dataset)

    # get mean and std of the specified dataset from constants
    dm = torch.as_tensor(
        getattr(consts,
                f"{args.dataset.upper()}_MEAN"), **setup)[:, None, None]
    ds = torch.as_tensor(
        getattr(consts,
                f"{args.dataset.upper()}_STD"), **setup)[:, None, None]

    # construct the server and the clients
    fl_server = FedAvgServer(args.num_clients, args.client_ratio, args.dataset,
                             args.model, args.is_iid, args.num_round,
                             args.epochs, 0.001)
    fl_server.prepare_data()
    fl_server.data_dispatcher()
    fl_server.construct_model()
    fl_server.init_client_models()
    model = fl_server.global_model
    model.to(**setup)   # load the model to the given device
    model.eval()

    training_stats = defaultdict(list)

    num_examples_of_client = len(fl_server.client_data_dispatch[0])

    if args.client_id is None:
        client_id = np.random.randint(0, args.num_clients)
    elif 0 <= args.client_id < args.num_clients:
        raise ValueError("Chosen client id %s should be in [0, %s]. " %
                         (args.client_id, args.num_clients))

    if args.target_id == -1:  # demo image
        # Specify PIL filter for lower pillow versions
        ground_truth = torch.as_tensor(
            np.array(Image.open("auto.jpg").resize(
                (32, 32), Image.BICUBIC)) / 255, **setup
        )
        ground_truth = ground_truth.permute(2, 0, 1).sub(dm) \
            .div(ds).unsqueeze(0).contiguous()
        if not args.label_flip:
            labels = torch.as_tensor((1,), device=setup["device"])
        else:
            labels = torch.as_tensor((5,), device=setup["device"])
        target_id = -1
    else:
        if args.target_id is None:
            target_id = np.random.randint(len(validloader.dataset))
        else:
            target_id = args.target_id
        ground_truth, labels = validloader.dataset[target_id]

        if args.label_flip:
            labels = (labels + 1) % len(trainloader.dataset.classes)
        ground_truth, labels = (
            ground_truth.unsqueeze(0).to(**setup),
            torch.as_tensor((labels,), device=setup["device"]),
        )

    # the first parameter is the number of image channels
    # img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])
    img_shape = (1, ground_truth.shape[2], ground_truth.shape[3])



    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)

    # compute the gradients based on loss, parameters
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
    print(f"Full gradient norm is {full_norm:e}.")

    if args.dtype != "float":
        if args.dtype in ["double", "float64"]:
            setup["dtype"] = torch.double
        elif args.dtype in ["half", "float16"]:
            setup["dtype"] = torch.half
        else:
            raise ValueError(f"Unknown data type argument {args.dtype}.")
        print(f"Model and input parameter moved to {args.dtype}-precision.")
        dm = torch.as_tensor(consts.CIFAR10_MEAN, **setup)[:, None,
             None]
        ds = torch.as_tensor(consts.CIFAR10_STD, **setup)[:, None,
             None]
        ground_truth = ground_truth.to(**setup)
        input_gradient = [g.to(**setup) for g in input_gradient]
        model.to(**setup)
        model.eval()

    config = dict(
        signed=args.signed,
        boxed=args.boxed,
        cost_fn=args.cost_fn,
        indices="def",
        weights="equal",
        lr=0.1,
        optim=args.optimizer,
        restarts=args.restarts,
        max_iterations=24_000,
        total_variation=args.tv,
        init="randn",
        filter="none",
        lr_decay=True,
        scoring_choice="loss",
    )

    rec_machine = GradientReconstructor(
            model, (dm, ds), config, num_images=args.num_images)

    output, stats = rec_machine.reconstruct(
        input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)
