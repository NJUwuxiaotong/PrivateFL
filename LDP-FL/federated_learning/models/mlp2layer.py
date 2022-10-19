from torch import nn


class MLP2Layer(nn.Module):
    """
    MnistMLP is a mlp model for dataset MNIST.
    """
    def __init__(self, example_shape, class_no, neurons_of_hidden_layers):
        """
        num_neurons: a set of neurons in middle layers of mlp.
        """
        super(MLP2Layer, self).__init__()
        self.access_no = example_shape[0]
        self.row_pixel = example_shape[1]
        self.column_pixel = example_shape[2]
        self.class_no = class_no
        self.feature_no = self.access_no * self.row_pixel * self.column_pixel

        self.neurons_of_hidden_layers = neurons_of_hidden_layers
        self.hidden_layer1 = None
        self.hidden_layer2 = None
        self.output_layer = None
        self.relu = None
        self.softmax = None

    def construct_model(self):
        """
            1st: input layer
            2nd: the first hidden layer with *** neurons
            3rd: the second hidden layer with *** neurons
            4th: output layer
        """
        # hidden layers
        self.hidden_layer1 = nn.Linear(self.feature_no,
                                       self.neurons_of_hidden_layers[0])
        self.hidden_layer2 = nn.Linear(self.neurons_of_hidden_layers[0],
                                       self.neurons_of_hidden_layers[1])
        # output layer
        self.output_layer = nn.Linear(self.neurons_of_hidden_layers[1],
                                      self.class_no)
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
        # import pdb; pdb.set_trace()
        input = input.reshape(-1, self.feature_no)
        hidden_layer1_output = self.hidden_layer1(input)
        hidden_layer1_output = self.relu(hidden_layer1_output)

        # the second hidden layer computation
        hidden_layer2_output = self.hidden_layer2(hidden_layer1_output)
        hidden_layer2_output = self.relu(hidden_layer2_output)

        # the output layer computation
        output_layer_result = self.output_layer(hidden_layer2_output)
        # output_layer_result = self.softmax(output_layer_result)
        return output_layer_result
