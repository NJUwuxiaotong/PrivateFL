from torch import nn


class CNN4Layer(nn.Module):
    def __init__(self, example_shape, class_no, **model_params):
        super(CNN4Layer, self).__init__()
        self.example_shape = example_shape
        self.channel_no = example_shape[0]
        self.row_pixel = example_shape[1]
        self.column_pixel = example_shape[2]
        self.class_no = class_no
        self.feature_no = self.channel_no * self.row_pixel * self.column_pixel

        # convolution layer
        self.conv1_params = model_params["conv1"]
        self.conv2_params = model_params["conv2"]
        self.conv3_params = model_params["conv3"]
        self.conv4_params = model_params["conv4"]

        # pooling layer
        self.pool1_params = model_params["pool1"]
        self.pool2_params = model_params["pool2"]
        self.pool3_params = model_params["pool3"]
        # self.pool4_params = model_params["pool4"]

        # fully connected layer
        self.fc1_params = model_params["fc1"]
        self.fc2_params = model_params["fc2"]

        # model layers
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.flatt = None
        self.dense = None

    def initial_layers(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_params["in_channel"],
                      out_channels=self.conv1_params["out_channels"],
                      kernel_size=self.conv1_params["kernel_size"],
                      stride=self.conv1_params["stride"],
                      padding=self.conv1_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool1_params["kernel_size"],
                         stride=self.pool1_params["stride"]))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2_params["in_channel"],
                      out_channels=self.conv2_params["out_channels"],
                      kernel_size=self.conv2_params["kernel_size"],
                      stride=self.conv2_params["stride"],
                      padding=self.conv2_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool2_params["kernel_size"],
                         stride=self.pool2_params["stride"]))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv3_params["in_channel"],
                      out_channels=self.conv3_params["out_channels"],
                      kernel_size=self.conv3_params["kernel_size"],
                      stride=self.conv3_params["stride"],
                      padding=self.conv3_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool3_params["kernel_size"],
                         stride=self.pool3_params["stride"]))

        #self.conv4 = nn.Sequential(
        ##    nn.Conv2d(in_channels=self.conv4_params["in_channel"],
        #              out_channels=self.conv4_params["out_channels"],
        #              kernel_size=self.conv4_params["kernel_size"],
        #              stride=self.conv4_params["stride"],
        #              padding=self.conv4_params["padding"]),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=self.pool4_params["kernel_size"],
        #                 stride=self.pool4_params["stride"]))

        self.flatt = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(self.fc1_params["in_neuron"],
                      self.fc1_params["out_neuron"]),
            nn.Linear(self.fc2_params["in_neuron"],
                      self.fc2_params["out_neuron"]),
            nn.ReLU(),
            nn.Linear(self.fc2_params["out_neuron"], self.class_no),
            #nn.Softmax(dim=1)
        )

    def forward(self, input_example):
        input_example = input_example.reshape(
            (-1, self.channel_no, self.row_pixel, self.column_pixel))
        conv1_out = self.conv1(input_example)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        # conv4_out = self.conv4(conv3_out)
        res = self.flatt(conv3_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out
