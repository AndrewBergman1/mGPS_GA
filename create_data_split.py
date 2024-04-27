import pandas as pd
from sklearn.model_selection import train_test_split



tot_data = pd.read_csv("metasub_taxa_abundance.csv")

fold = 1



train_data, test_data = train_test_split(tot_data, test_size=0.20)


train_data.to_csv(f'training_data.csv', index=False)
test_data.to_csv(f'testing_data.csv', index=False)