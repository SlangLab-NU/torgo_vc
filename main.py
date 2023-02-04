

import numpy as np
from numpy import array


class myList(list):

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 10

    def __getitem__(self, item):
        return [1,2,3]






def main():

    print("Hello world")

    x = myList()
    x.append(1)
    x.append(2)
    print(x)
    print(f"Number at 1 {x[1]}")
    print(len(x))

if __name__ == "__main__":
    main()