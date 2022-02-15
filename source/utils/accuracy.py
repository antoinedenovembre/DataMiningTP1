import matplotlib.pyplot as plt


def print_accuracy(accu_r, acc):
    plt.plot(accu_r, acc)
    plt.show()
    max_accu = max(acc)
    print(max_accu, " is the maximum accuracy, reached for index ", acc.index(max_accu))
