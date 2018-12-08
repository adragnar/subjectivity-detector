"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/data.tsv', sep='\t')
subj = data[:5000]
obj = data[5000:]

train_s, val_test_s = train_test_split(subj, test_size=0.36)
val_s, test_s = train_test_split(val_test_s, test_size=20/36)

train_o, val_test_o = train_test_split(obj, test_size=0.36)
val_o, test_o = train_test_split(val_test_o, test_size=20/36)

train = pd.concat([train_s, train_o])
val = pd.concat([val_s, val_o])
test = pd.concat([test_s, test_o])

train.to_csv('train.tsv', sep='\t', index=False)
val.to_csv('validation.tsv', sep='\t', index=False)
test.to_csv('test.tsv', sep='\t', index=False)

print(train_s.shape, train_o.shape, val_s.shape, val_o.shape, test_s.shape, test_o.shape)
