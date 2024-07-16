import pandas as pd

path = 'C:/Users/pc/Desktop/jobs_dataset_with_features.csv'

df = pd.read_csv(path)
print(df.head(), '\n', df.shape, '\n', df.columns)