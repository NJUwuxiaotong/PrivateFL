import numpy as np

from pub_lib.pub_libs import analyze_dist_of_single_att


def data_dispatcher(is_balanced, is_iid, client_no, training_labels):
    """
    Function: execute the data assignment for the clients.
    client_data_dispatch - type: array, shape: client_no * example_no
    """
    print("---------- Data Distribution -------------")
    print("Dist of Total Examples is %s" %
          analyze_dist_of_single_att(training_labels))

    if not is_balanced:   # unbalanced and non-iid
        # 80%/20%
        example_no = len(training_labels)
        large_num = int((example_no/client_no)*2*0.8)
        low_num = int((example_no/client_no)*2*0.2)
        pair_num = int(client_no/2)
        examples_dispatch_no = np.array([large_num, low_num]*pair_num)

        client_data_dispatch = np.arange(example_no)
        np.random.shuffle(client_data_dispatch)

        final_dispatch = list()
        start_pos = 0
        for dispatch_no in examples_dispatch_no:
            final_dispatch.append(
                client_data_dispatch[start_pos:
                                     start_pos+dispatch_no.tolist()].tolist())
            start_pos = start_pos + dispatch_no.tolist()
        return final_dispatch

    if is_iid:
        # training examples shuffle
        example_no = len(training_labels)
        client_data_dispatch = np.arange(example_no)
        np.random.shuffle(client_data_dispatch)
        client_data_dispatch = \
            client_data_dispatch.reshape(client_no, -1)
    else:
        """
        Dispatch method 1:
        self.client_data_dispatch = \
            self.training_labels.reshape(1, -1).numpy().argsort()[0]
        """
        # 10 is ok.
        example_block_no = 1
        client_order = np.arange(client_no * example_block_no)
        np.random.shuffle(client_order)
        client_order = client_order.reshape(-1, example_block_no)

        client_data_dispatch = \
            training_labels.reshape(1, -1).numpy().argsort()[0]
        client_data_dispatch = \
            client_data_dispatch.reshape(client_no * example_block_no, -1)
        client_data_dispatch = client_data_dispatch[client_order]
        client_data_dispatch = client_data_dispatch.reshape(client_no, -1)

    """
    for i in range(client_no):
        label_dist = analyze_dist_of_single_att(
            training_labels[client_data_dispatch[i]])
        print("Client %s - Dist of Examples: %s" % (i, label_dist))
    print("--------------- End ----------------------")
    """
    return client_data_dispatch
