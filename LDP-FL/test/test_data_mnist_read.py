from data_process.data_mnist_read import DatasetMnist


mnist = DatasetMnist()
mnist.read_data()
mnist.show_example(mnist.training_examples[0])
