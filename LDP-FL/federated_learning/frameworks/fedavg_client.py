import numpy as np
import torch
from torch import nn

from constant import consts as const


class FedAvgClient(object):
    def __init__(self, model_type, training_row_pixel, training_column_pixel,
                 label_unique_no):
        # epoch_no=10, lr=0.001
        self.model_type = model_type
        self.label_unique_no = label_unique_no
        self.training_row_pixel = training_row_pixel
        self.training_column_pixel = training_column_pixel

        self.local_model = None
        self.loss_fn = nn.CrossEntropyLoss()

        self.epoch_total_loss = 0.0
        # self.epoch_no = None
        # self.lr = None

    def construct_model(self, model, **kwargs):
        """
        elif self.model_type == const.MNIST_CNN_MODEL:
            self.local_mnist_model = MNISTCNN(
                self.training_row_pixel,
                self.training_column_pixel,
                self.label_unique_no, )

        self.local_mnist_model.initial_layers()
        self.local_mnist_model.load_state_dict(model_params)
        """
        self.local_model = model

    def training_model(self, training_examples, training_labels,
                       training_example_no, epoch_no=10, lr=0.001,
                       batch_size=50):
        opt = torch.optim.SGD(
            self.local_model.parameters(), lr=lr)
        batch_no = int(training_example_no / batch_size)

        for epoch in range(epoch_no):
            start_pos = 0

            new_example_order = np.arange(training_example_no)
            np.random.shuffle(new_example_order)
            for i in range(batch_no):
                training_examples_order = \
                    new_example_order[start_pos: start_pos + batch_size]
                examples_feature = training_examples[training_examples_order]
                examples_labels = \
                    training_labels[training_examples_order].reshape(1, -1)

                if self.model_type in \
                        [const.MNIST_CNN_MODEL, const.ResNet18_MODEL]:
                    examples_feature = examples_feature.reshape(
                        batch_size, 1, self.training_row_pixel,
                        self.training_column_pixel)

                pred_labels = self.local_model(examples_feature)

                loss = self.loss_fn(pred_labels, examples_labels[0])

                opt.zero_grad()
                loss.backward()    # w.grad
                opt.step()
                start_pos = start_pos + batch_size

            # with torch.no_grad():
            #     acc = self.compute_accuracy(test_examples, test_labels)
            #     print("Epoch %s: Accuracy %.2f%%" % (epoch, acc * 100))
        return self.local_model.state_dict()

    def compute_accuracy(self, test_examples, test_labels):
        accuracy = 0.0
        test_example_no = len(test_examples)
        if self.model_type in [const.MNIST_CNN_MODEL, const.ResNet18_MODEL]:
            result = self.local_model(
                test_examples.reshape(
                    test_example_no, 1, self.training_row_pixel,
                    self.training_column_pixel))\
                .reshape(test_example_no, -1)
        elif self.model_type == const.MNIST_MLP_MODEL:
            result = self.local_model(test_examples)\
                .reshape(test_example_no, -1)
        else:
            exit(1)

        for i in range(test_example_no):
            pred_result = torch.argmax(result[i])
            if pred_result == test_labels[i]:
                accuracy = accuracy + 1
        return accuracy / test_example_no

    def stochastic_gradient_descent(self, training_examples, training_labels,
                                    training_example_no):
        opt = torch.optim.SGD(self.local_model.parameters(), self.lr)
        for epoch in range(self.epoch_no):
            for example_no in range(training_example_no):
                pred_label = \
                    self.local_model(training_examples[example_no])
                loss = \
                    self.loss_fn(pred_label, training_labels[example_no])
                with torch.no_grad():
                    self.epoch_total_loss = self.epoch_total_loss + loss
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self.local_model.state_dict()
