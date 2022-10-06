from torch import nn


class CNN2Layer(object):
    def __init__(self, example_shape, class_no, **model_params):
        self.example_shape = example_shape
        self.channel_no = example_shape[0]
        self.row_pixel = example_shape[1]
        self.column_pixel = example_shape[2]
        self.class_no = class_no
        self.feature_no = self.access_no * self.row_pixel * self.column_pixel

        """
        conv_kernel_size, conv_stride, conv_padding, conv_channels,
        pooling_kernel_size = 2,
        pooling_stride = 2, fc_neuron_no = 512
        """

        # convolution layer
        self.conv_kernel_size = model_params["conv_kernel_size"]
        self.conv_stride = model_params["conv_stride"]
        self.conv_padding = model_params["conv_padding"]
        self.conv_channels = model_params["conv_channels"]

        # pooling layer
        self.pooling_kernel_size = model_params["pooling_kernel_size"]
        self.pooling_stride = model_params["pooling_stride"]

        # fully connected layer
        self.fc_neuron_no = model_params["fc_neuron_no"]

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
            nn.Linear(self.fc_neuron_no, self.class_no),
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
