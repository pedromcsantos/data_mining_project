import pandas as pd
<<<<<<< HEAD
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
=======

df = pd.read_csv("A2Z Insurance.csv")

df = df.set_index("Customer Identity")








>>>>>>> 27e86d451c43d6456f21fe7eadfb7c990bbb0d19
#"asadasda" 

#sada asdasdas 