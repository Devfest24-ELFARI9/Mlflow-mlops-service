import numpy as np
import pandas as pd


def sample(data, target_variable = "Status", test_factor= 0.1, train_factor=0.4):
    class_counts = data[target_variable].value_counts()
    min_class_size = class_counts.min()  # Get the size of the smallest class
    min_class_size -= np.floor(test_factor * min_class_size)
    train_data = pd.DataFrame()
    for target, count in class_counts.items():
      class_data = data[target_variable] == target
      if np.floor(count * test_factor) > min_class_size:
          sampled_df = data[class_data].sample(np.ceil(min_class_size + train_factor * min_class_size).astype(int), replace=False)
      else :
          sampled_df = data[class_data].sample(min_class_size.astype(int), replace=False)
      train_data = pd.concat([train_data, sampled_df], ignore_index=False)
    print(train_data[target_variable].value_counts())
    return train_data



