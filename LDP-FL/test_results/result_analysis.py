
file_path = "C:\\workspace\\workspace\\projects\\PrivateFL\\LDP-FL" \
            "\\test_results\\fd_resnet_false_ratio_0.1_lr_0.001_dd_10"


def add_a(a, b):
    return a+b


"data analysis"
with open(file_path, "r") as fl:
    result = fl.readlines()

    for line in result:
        import pdb; pdb.set_trace()

        add_a(1, 2)

        if "[" in line:
            left_pos = line.find("[")
            right_pos = line.find("]")
            entries = line[left_pos + 1: right_pos].split(",")
            # import pdb; pdb.set_trace()
            for entry in entries:
                print("%4d" % int(entry), end=" ")

            for i in range(10-len(entries)):
                print("%4d" % 0, end=" ")

            print("")


with open(file_path, "r") as fl:
    result = fl.readlines()
    for line in result:
        if "Accuracy" in line:
            pos = line.find("Accuracy") + 8
            r = line[pos: pos + 6]
            print(r)