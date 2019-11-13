import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
df = pd.read_csv("A2Z Insurance.csv")

df = df.set_index("Customer Identity")

df.isnull().sum()

df = df[df["Brithday Year"]>1900] #Drop one case where birthday year <1900 

newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]

df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)

describe = df.describe()

#multiple salary by 12 to have everything in the same unit (year)

df["salary_monthly"] = df["salary_monthly"]*12
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
plt.scatter(x= df.index, y=df["first_policy"])
plt.show() #only one outlier, we will delete it

df = df[df["first_policy"]<10000]
plt.scatter(x= df.index, y=df["first_policy"]) 
plt.show() #everything looks fine now

plt.scatter(x= df.index, y=df["salary_year"])
plt.show()

salary_outlier = df[df["salary_year"]>30000*12] #maybe very rich, we will keep him for now

#Check for every customer if the first policy was made before he was born

check_policy = df[df["first_policy"]<df["birth_year"]] #~2000 values, we have to look into it

plt.scatter(x= df.index, y=df["mon_value"]) 
plt.show()

mon_value_outlier = df[df["mon_value"]<-25000]  #we will also keep them to further analysis

plt.scatter(x= df.index, y=df["claims_rate"]) 
plt.show()
claims_rate_outlier = df[df["claims_rate"]>20] #theres a correlation between claims and monetary so individuals overlap

plt.scatter(x= df.index, y=df["premium_motor"]) 
plt.show()

motor_outlier = df[df["premium_motor"]>2000]

plt.scatter(x= df.index, y=df["premium_household"]) 
plt.show()

household_outlier = df[df["premium_household"]>5000] #ask Jorge what should we do in this case 

plt.scatter(x= df.index, y=df["premium_health"]) 
plt.show()
health_outlier = df[df["premium_health"]>5000]

plt.scatter(x= df.index, y=df["premium_work_comp"]) 
plt.show()
work_outlier = df[df["premium_work_comp"]>1750] #not rly an outlier

plt.scatter(x= df.index, y=df["premium_life"]) 
plt.show() #looks ok

#filling NAN

dfisnull = df.isnull().sum()

sb.heatmap(df.corr())
corr = df.corr()

educ_nan = df[df["educ"]!=df["educ"]]

from sklearn.neighbors import NearestNeighbors

lifemissing = df[(df["premium_life"]!=df["premium_life"])|(df["premium_life"]==0)]

#Assumption: nan values in premium mean no contract 

df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)


educ_grouped = df[["educ","birth_year", "salary_year"]].groupby(["educ","birth_year"]).mean()

from sklearn import preprocessing
le_status = preprocessing.LabelEncoder()
df['loc_encoded'] = le_status.fit_transform(df["location"])
le_status.classes_

from sklearn.preprocessing import OneHotEncoder
#define encoder
status_ohe = OneHotEncoder()

#apply encoder


Status = status_ohe.fit_transform(df.loc_encoded.values.reshape(-1,1)).toarray()

my_data_to_label_OneHot_s = pd.DataFrame(Status, columns = ["location_"+str(i) for i in range(4)] )


new_data = df.join(my_data_to_label_OneHot_s, on=df.index)

df_incomplete = df[df["educ"]==df["educ"]]

Status_2 = status_ohe.fit_transform(df_incomplete.educ.values.reshape(-1,1)).toarray()
s
my_data_to_label_OneHot_s_2 = pd.DataFrame(Status_2, columns = ["educ_"+str(i) for i in range(4)] )

new_data = new_data.join(my_data_to_label_OneHot_s_2, on=d)
