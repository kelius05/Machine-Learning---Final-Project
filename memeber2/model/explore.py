import pandas as pd

df = pd.read_csv("../../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")  #Load the dataset

print(f"Shape: {df.shape} ")   #prints(rows,columns) - expect (2111, 17 )
print(df.head())               #Shows the first 5 rows so you can see what the data looks like
print(df.dtypes)   #Shows each columns data type - 'object' means text, 'float' means numeric
print(df.isnull().sum())   #Counts missing values per column - want all zeroes
print(df['NObeyesdad'].value_counts())   #Shows how many samples belong to each obesity category
print(df.select_dtypes(include='object').columns.tolist())   #Lists all text columns - these ar ethe ones that need encoding