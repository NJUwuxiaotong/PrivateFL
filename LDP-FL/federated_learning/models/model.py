from abc import abstractmethod
from torch import nn


class IntModel(nn.Module):
    def __init__(self, row_pixel, column_pixel, num_channels, num_classes):
        super().__init__()
        self.row_pixel = row_pixel
        self.column_pixel = column_pixel
        self.num_pixels = row_pixel * column_pixel
        self.num_channels = num_channels
        self.num_classes = num_classes

    @abstractmethod
    def construct_model(self):
        pass
