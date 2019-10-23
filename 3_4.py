import numpy as np
import pandas as pd
import random as rd


# Choose iris dataset.
# Prepare data.
def loadData(filename):
    dataset = pd.read_excel(filename)
    dataset['b'] = 1
    x = np.array(dataset[['Column1', 'Column2', 'Column3', 'Column4', 'b']])
    y = np.array(dataset[['Column5']].replace(['Iris-setosa', 'Iris-versicolor',
                                               'Iris-virginica'], [1, 2, 3]))

    return x, y


x, y = loadData('3_4data.xlsx')

x1, x2, x3 = x[:50], x[50:100], x[100:150]
y1, y2, y3 = y[:50], y[50:100], y[100:150]

def shuffle(x1,y1,x2,y2,k_fold=10):
    idx1 = rd.shuffle(list(range(0,50)))
    idx2 = rd.shuffle(list(range(50,100)))
    test_idx = idx1[:5] + idx2[:5]
    train_idx = idx1[5:50] + idx2[50:100]
# To be continued...
