import pandas as pd

df = pd.read_csv("A2Z Insurance.csv")

df.isnull().sum()

df = df[df["Brithday Year"]>1900] #Drop one case where birthday year <1900 


df = pd.read_csv("A2Z Insurance.csv")

df = df.set_index("Customer Identity")
