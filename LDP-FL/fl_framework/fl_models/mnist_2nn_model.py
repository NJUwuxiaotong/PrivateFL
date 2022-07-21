from torch import nn

from fl_framework.fl_models.mnist_model import MNISTModel


class MNIST2NN(MNISTModel):
    def __init__(self, row_pixel, column_pixel, label_no, neuron_no):
        super().__init__(row_pixel, column_pixel, label_no)
        self.neuron_no = neuron_no

    def initial_layers(self):
        # hidden layers
        self.hidden_layer1 = nn.Linear(self.pixels, self.neuron_no)
        self.hidden_layer2 = nn.Linear(self.neuron_no, self.neuron_no)

        # output layer
        self.output_layer = nn.Linear(self.neuron_no, self.label_no)

        # activation function in hidden layers
        self.relu = nn.ReLU()

        # activation function in output layer
        # the input of self.softmax is a matrix
        # dim=0, columns in a matrix sums to 1.
        # dim=1,  rows in a matrix sums to 1.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        """
        return: a tensor such as [[a, b, c, ...]]
                only a row vector in the matrix
        """
        # the first hidden layer computation
        hidden_layer1_output = self.hidden_layer1(input)
        hidden_layer1_output = self.relu(hidden_layer1_output)

        # the second hidden layer computation
        hidden_layer2_output = self.hidden_layer2(hidden_layer1_output)
        hidden_layer2_output = self.relu(hidden_layer2_output)

        # the output layer computation
        output_layer_result = self.output_layer(hidden_layer2_output)
        output_layer_result = self.softmax(output_layer_result)
        return output_layer_result
