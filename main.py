import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("A2Z Insurance.csv")

df.isnull().sum()

df = df[df["Brithday Year"]>1900] #Drop one case where birthday year <1900 


df = df.set_index("Customer Identity")

plt.scatter(df.index, df["First Policy´s Year"])
plt.show()