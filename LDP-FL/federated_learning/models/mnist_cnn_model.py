from torch import nn

from federated_learning.models.model import IntModel


class MNISTCNN(IntModel):
    def __init__(self, row_pixel, column_pixel, num_channels, label_no,
                 conv_kernel_size, conv_stride, conv_padding, conv_channels,
                 pooling_kernel_size=2,
                 pooling_stride=2, fc_neuron_no=512):
        super().__init__(row_pixel, column_pixel, num_channels, label_no)

        # convolution layer
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.conv_channels = conv_channels

        # pooling layer
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride

        # fully connected layer
        self.fc_neuron_no = fc_neuron_no

    def initial_layers(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels[0],
                      out_channels=self.conv_channels[1],
                      kernel_size=self.conv_kernel_size,
                      stride=self.conv_stride,
                      padding=self.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size,
                         stride=self.pooling_stride))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels[1],
                      out_channels=self.conv_channels[2],
                      kernel_size=self.conv_kernel_size,
                      stride=self.conv_stride,
                      padding=self.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size,
                         stride=self.pooling_stride))

        self.dense = nn.Sequential(
            nn.Linear(7*7*self.conv_channels[2], self.fc_neuron_no),
            nn.ReLU(),
            nn.Linear(self.fc_neuron_no, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input_example):
        """
        input_example: one dimensional matrix
        """
        #input_example = input_example.reshape(self.row_pixel, self.column_pixel)
        #input_example = \
        #    input_example.view((1, 1) + input_example.shape)

        conv1_out = self.conv1(input_example)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)
        out = self.dense(res)
        return out
