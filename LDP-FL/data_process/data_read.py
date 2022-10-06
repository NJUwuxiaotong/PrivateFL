from constant import consts as const


class DataInput():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self._check_dataset_name()
        self.training_examples = None
        self.training_examples_no = None
        self.training_labels = None
        self.training_labels_no = None
        self.test_examples = None
        self.test_examples_no = None
        self.test_labels = None
        self.test_labels_no = None

    def _check_dataset_name(self):
        if self.dataset_name not in const.DATASET_NAMES:
            print("ERROR: Dataset [%s] does not exist!" % self.dataset_name)
            exit(1)

    def read_data(self):
        pass
