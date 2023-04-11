import pandas as pd
import numpy
import matplotlib.pyplot as plt   

df = pd.read_csv('./data.csv',sep=',')
df = df.drop_duplicates()
df = df.dropna()
