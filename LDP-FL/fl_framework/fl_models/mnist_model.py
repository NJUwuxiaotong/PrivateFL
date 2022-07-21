from abc import abstractmethod
from torch import nn


class MNISTModel(nn.Module):
    def __init__(self, row_pixel, column_pixel, label_no):
        super().__init__()
        self.row_pixel = row_pixel
        self.column_pixel = column_pixel
        self.pixels = row_pixel * column_pixel
        self.label_no = label_no

    @abstractmethod
    def initial_layers(self):
        pass
