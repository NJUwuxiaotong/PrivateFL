import gzip
import numpy as np
import matplotlib.pyplot as plt

from data_process.data_read import DataInput
from constant import consts as const


class DatasetMnist(DataInput):
    def __init__(self):
        super().__init__(const.DATASET_MNIST)
        self.training_row_pixel = 0
        self.training_column_pixel = 0
        self.test_row_pixel = 0
        self.test_column_pixel = 0

    def read_data(self):
        # read the training examples
        self.training_examples_no, self.training_row_pixel, \
        self.training_column_pixel, self.training_examples = \
            self.read_images(const.MNIST_TRAIN_IMAGE_DIR)

        # read the labels
        self.training_examples_no, self.training_labels = \
            self.read_labels(const.MNIST_TRAIN_LABEL_DIR)
        self.num_classes = 0

        # read the test examples
        self.test_examples_no, self.test_row_pixel, self.test_column_pixel, \
        self.test_examples = \
            self.read_images(const.MNIST_TEST_IMAGE_DIR)

        # read the test labels
        self.test_examples_no, self.test_labels = \
            self.read_labels(const.MNIST_TEST_LABEL_DIR)

        # image_normalized
        self.centralized()

    @staticmethod
    def read_images(dataset_dir):
        # read the raw data
        with gzip.open(
                dataset_dir) as image_file:
            image_data = image_file.read()

        # dataset basic information
        image_no = int.from_bytes(image_data[4:8], byteorder="big")
        row_pixel_len = int.from_bytes(image_data[8:12], byteorder="big")
        column_pixel_len = int.from_bytes(image_data[12:16], byteorder="big")

        # read the images
        images = list()
        start_pos = 16
        image_pixel = row_pixel_len * column_pixel_len
        for i in range(image_no):
            # get a image
            image = list()
            for j in range(image_pixel):
                image.append(image_data[start_pos + j])
            image = np.array(image)
            images.append(image)
            start_pos = start_pos + image_pixel
        images = np.array(images)
        return image_no, row_pixel_len, column_pixel_len, images

    @staticmethod
    def read_labels(dataset_dir):
        # read the raw data
        with gzip.open(dataset_dir) as label_file:
            label_data = label_file.read()

        # dataset basic information
        label_no = int.from_bytes(label_data[4:8], byteorder="big")

        # read the labels
        labels = list()
        start_pos = 8
        for i in range(label_no):
            labels.append(label_data[start_pos + i])
        labels = np.array(labels)
        return label_no, labels

    def get_mean(self):
        train_image_mean = np.mean(self.training_examples, axis=0)
        return train_image_mean

    def get_std(self):
        train_image_std = np.std(self.training_examples, axis=0)
        return train_image_std

    def centralized(self):
        image_mean = self.get_mean()
        image_std = self.get_std()

        self.training_examples = self.training_examples - image_mean
        self.test_examples = self.test_examples - image_mean

    def show_example(self, image_data):
        img = np.array(image_data).reshape(
            self.training_row_pixel, self.training_column_pixel)
        plt.imshow(img)  # cmap=plt.cm.binary
        plt.show()
