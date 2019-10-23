import numpy as np
import pandas as pd


# Choose iris dataset.
# Prepare data.
def loadData(filename):
    dataset = pd.read_excel(filename)
    dataset['b'] = 1
    x = np.array(dataset[['Column1', 'Column2', 'Column3', 'Column4', 'b']])
    y = np.array(dataset[['Column5']].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, 2, 3]))

    return x, y

x,y=loadData('3_4data.xlsx')
print(x.shape)
print(x)
print(y)
# To be continued...