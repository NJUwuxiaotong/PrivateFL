from data_process.data_mnist_read import MnistInput


mnist = MnistInput()
mnist.read_data()
mnist.show_example(mnist.training_examples[0])
