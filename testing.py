import pandas as pd
import sys

preds = pd.read_csv("./metasub_global_git.csv")
df = pd.read_csv("metasub_global_results.csv")

print(df.columns)

print(df["uuid"])


predictors = preds['taxa']

df = df.drop('Unnamed: 0', axis=1)

columns = predictors.tolist() + ['uuid']


df_filtered = df[columns]

print(df_filtered)

df_filtered.to_csv("200_preds.csv")