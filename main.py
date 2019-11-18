import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from itertools import chain
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from kmodes.kprototypes import KPrototypes

df = pd.read_csv("A2Z Insurance.csv")
df = df.set_index("Customer Identity")
newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)


#####################################################################################
describe = df.describe()

# Detect potential outlier and drop rows with obvious mistakes
df = df[df["birth_year"]>1900] #Drop one case where birthday year <1900 

df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
salary_outlier = df[df["salary_year"]>30000*12] #maybe very rich, we will keep him for now

#plt.scatter(x= df.index, y=df["first_policy"])
#plt.show() #only one outlier, we will delete it
df = df[df["first_policy"]<10000]

#plt.scatter(x= df.index, y=df["salary_year"])
#plt.show()

#plt.scatter(x= df.index, y=df["mon_value"]) 
#plt.show()

mon_value_outlier = df[df["mon_value"]<-25000]  #we will also keep them to further analysis

#plt.scatter(x= df.index, y=df["claims_rate"]) 
#plt.show()
claims_rate_outlier = df[df["claims_rate"]>20] #theres a correlation between claims and monetary so individuals overlap

#plt.scatter(x= df.index, y=df["premium_motor"]) 
#plt.show()

motor_outlier = df[df["premium_motor"]>2000]

#plt.scatter(x= df.index, y=df["premium_household"]) 
#plt.show()

household_outlier = df[df["premium_household"]>5000] #ask Jorge what should we do in this case 

#plt.scatter(x= df.index, y=df["premium_health"]) 
#plt.show()
health_outlier = df[df["premium_health"]>5000]

#plt.scatter(x= df.index, y=df["premium_work_comp"]) 
#plt.show()
work_outlier = df[df["premium_work_comp"]>1750] 

#plt.scatter(x= df.index, y=df["premium_life"]) 
#plt.show() #looks ok

#Check for every customer if the first policy was made before he was born
check_policy = df[df["first_policy"]<df["birth_year"]] #~2000 values, we treat the wrong dates as nan values
def birth_helpfunc(x):
    if x["birth_year"] > x["first_policy"]:
        x["birth_year"] = np.nan
    return x
df = df.apply(lambda x: birth_helpfunc(x), axis=1)

#####################################################################################
################# Outlier #################
df.reset_index(inplace=True,drop=True)
df_num = pd.DataFrame(df[['first_policy','birth_year', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])

def calc_threshold(column_name, multiplier):
	Q1 = df_num[column_name].quantile(0.25)
	Q3 = df_num[column_name].quantile(0.75)
	IQR = Q3 - Q1
	return IQR * multiplier

#multi = 1.5
#multi = 3
multi = 4.5

outliers = []
for col in df_num.columns.values[1:]:
	th_pos = calc_threshold(col, multi) + df_num[col].mean()
	th_neg = df_num[col].mean() - calc_threshold(col, multi)
	outliers.append(df_num[(df_num[col] >= th_pos) | (df_num[col] <= th_neg)].index.values)

df_outlier = df_num.iloc[list(set([o for l in outliers for o in l]))]

df = df[~df.index.isin(df_outlier.index.values)]

#####################################################################################
################# filling NAN #################

dfisnull = df.isnull().sum()

# Filling nan values in premium columns
#Assumption: nan values in premium mean no contract 
df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)


###### Option 1 ######

# Filling nan values in educ, salary, has_children and birth_year (the wrong one) with k-prototype
# For this step we remove all rows with nan-values and outliers from the dataframe
df_fill = df.drop(outliers)
df_fill = df_fill.dropna()
df_fill = df_fill.reset_index(drop=True)
# Normalization
scaler = StandardScaler()
num_norm = scaler.fit_transform(df_fill[['first_policy','birth_year', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])
df_num_norm = pd.DataFrame(num_norm, columns = ['first_policy','birth_year', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp'])
df_fill_norm = df_num_norm.join(df_fill[["educ", "location","has_children"]])

# Fit model 
kproto = KPrototypes(n_clusters=9, init='random', random_state=1, n_init=1)
observ_cluster = kproto.fit_predict(df_fill_norm, categorical=[10,11,12])

# Inverse Normalization for Interpretation
cluster_centroids_num = pd.DataFrame(scaler.inverse_transform(X = kproto.cluster_centroids_[0]), columns = df_num_norm.columns)
cluster_centroids = pd.concat([cluster_centroids_num,pd.DataFrame(kproto.cluster_centroids_[1])], axis=1)
cluster_centroids.columns = df_fill_norm.columns

# Fill missing values with mean of column, predict cluster of customer and change missing value to centroid of cluster
df_isnan = df[df.isnull().any(axis=1)].reset_index(drop=True)
isnan_check = df_isnan.isnull() # to check if a value was a nan-value before imputation

# Fill missing values with mean/mode
df_isnan.loc[:,['first_policy','birth_year', 'salary_year']] = df_isnan.loc[:,['first_policy','birth_year', 'salary_year']].fillna(df_isnan[['first_policy','birth_year', 'salary_year']].mean())
df_isnan.loc[:,["educ", "location","has_children"]] = df_isnan.loc[:,["educ", "location","has_children"]].fillna(df_isnan[["educ", "location","has_children"]].mode().iloc[0,:])


# Predict clusters of customers with missing values
df_isnan["cluster"] = kproto.predict(df_isnan, categorical=[0,1,2])

# Change nan-values to centroids of cluster
df_filled = df_isnan.copy()
for i in isnan_check.columns.values: 
	for j in isnan_check.index.values:
		if isnan_check.loc[j,i]:
			df_filled.loc[j,i] = cluster_centroids.loc[df_filled.loc[j, "cluster"], i]

##############################################################
###### Option 2 ######
# Fill nan-value individually by column
# Use correlation matrix
corrmap = df.corr()

# Drop customers with nan values in "salary_year","educ" or "has_children" because these are only a few customers and we do not have any reasonable correlation to fill it 
df = df.dropna(subset=["salary_year","educ","has_children"])

# birth_year
# correlation with salary is quite high
# Fill values with mean of customers with same salary +/-1000, drop customers if salaray is nan
def get_birthyear(salary):
	 return round(df.loc[df["birth_year"].notnull() & (df["salary_year"] <= salary + 1000) & (df["salary_year"] <= salary + 1000)].mean())

df.loc[df["birth_year"].isnull(),"birth_year"] = df.loc[df["birth_year"].isnull(),"salary_year"].apply(lambda s: get_birthyear(s))

##############################################################


################# Feature selection #################

# Split the features in customer- and product-related. 
customer_related = ['first_policy', 'birth_year', 'educ', 'salary_year', 'location','has_children', 'mon_value', 'claims_rate']
product_related = ['premium_motor','premium_household', 'premium_health', 'premium_life','premium_work_comp']


################# Similarity measures #################

# euclidean or cosine





################# Choose algorithm #################
##### Customer-related #####
### K-Prototype ###

# Normalization
scaler = StandardScaler()
num_norm = scaler.fit_transform(df_fill[['first_policy','birth_year', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])
df_num_norm = pd.DataFrame(num_norm, columns = ['first_policy','birth_year', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp'])
df_fill_norm = df_num_norm.join(df_fill[["educ", "location","has_children"]])

# Find number of clusters
# Elbow graph
C = []

for i in range(1,20):
	kproto = KPrototypes(n_clusters=i, init='random', random_state=1).fit(df_fill_norm, categorical=[10,11,12])
	C.append(kproto.cost_)

plt.plot(range(1,20), C)	
	




# Inverse Normalization for Interpretation
cluster_centroids_num = pd.DataFrame(scaler.inverse_transform(X = kproto.cluster_centroids_[0]), columns = df_num_norm.columns)
cluster_centroids = pd.concat([cluster_centroids_num,pd.DataFrame(kproto.cluster_centroids_[1])], axis=1)
cluster_centroids.columns = df_fill_norm.columns











