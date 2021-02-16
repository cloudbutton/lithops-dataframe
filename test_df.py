# See https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/

from lithops.dataframe import read_csv
from lithops import Storage
import numpy as np


if __name__ == '__main__':

    #st = Storage(backend='ibm_cos')
    #with open('test_integers.csv', 'rb') as f:
    #    st.put_object('lithops-data-us-east', 'test_integers.csv', f.read())

    file = 'cos://lithops-data-us-east/test_integers.csv'
    df = read_csv(file, delimiter=',', names=['A', 'B', 'C', 'D', 'E', 'F'])

    def myadd(row, a, b=1):
        return row.sum() + a + b

    df.apply(myadd, axis=1, args=(2,), b=1.5)
