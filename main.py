import pandas as pd
<<<<<<< HEAD

=======
#michael is very stupid
#chuck norris is dead
# but he survived
>>>>>>> 7e7c48ad41c2e73e643e843a037640395d7314b2
df = pd.read_csv("A2Z Insurance.csv")

df.isnull().sum()

df =df[df["Brithday Year"]>1900] #Drop one case where birthday year <1900 

#asdasd