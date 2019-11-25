import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score  #avg of avgs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from helperFunctions import create_silgraph, get_outliers_i, create_elbowgraph


df = pd.read_csv("A2Z Insurance.csv")

# Preprocessing
df = df.set_index("Customer Identity")
newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)
df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
df = df[df["birth_year"]>1900] #Drop one case where birthday year <1900 
df = df[df["first_policy"]<50000] #Drop one case where first_policy year <50000 


#####################################################################################
################# Outlier #################
df.reset_index(inplace=True,drop=True)
df_num = pd.DataFrame(df[['first_policy','birth_year', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])
	
# Define individual multipliers for features
multipliers = {'first_policy': 0,'birth_year': 0, 'salary_year': 5,'mon_value': 5,'claims_rate': 5,'premium_motor': 5,'premium_household': 5,'premium_health': 5,'premium_life': 5,'premium_work_comp': 5}

outliers = []
for col, multi in multipliers.items():
	outliers.append(get_outliers_i(df_num, col, multi))

df_outlier = df_num.iloc[list(set([o for l in outliers for o in l]))]
df = df[~df.index.isin(df_outlier.index.values)]

#####################################################################################
################# filling NAN #################

# Filling nan values in premium columns
#Assumption: nan values in premium mean no contract 
df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)

# Drop customers with nan values in "salary_year","educ" or "has_children" because these are only a few customers and we do not have any reasonable correlation to fill it 
df = df.dropna(subset=["salary_year","educ","has_children"])

#######################################################################
######### Feature-engineering and -selection #########
df.reset_index(drop=True, inplace=True)

# We assume that policies can be passed from the parents to their children. We therefore drop first_policy and create two new features:
# customer_since feature: Either the first_policy date or the birth_year if the first_policy year is older than the customer
df["customer_since"] = [df.loc[i, "first_policy"] if df.loc[i, "first_policy"] > df.loc[i, "birth_year"] else df.loc[i, "birth_year"] for i in range(len(df.index.values))]
# is_family_policy: True if first_policy is older than customer
df["is_family_policy"] = [0 if df.loc[i, "first_policy"] > df.loc[i, "birth_year"] else 1 for i in range(len(df.index.values))]
# Drop first_policy
df = df.drop("first_policy", axis=1)

# Calculate total amount paid for premiums per year per customer
df["premium_total"] = [sum(p for p in premiums if p > 0) for i, premiums in df[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]

# Number of parts which customers cancelled this year (has a negative value in the premium-related columns)
df["cancelled_contracts"] = [sum(1 for p in premiums if p < 0) for i, premiums in df[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]

# Split the features in customer- and product-related. 
customer_related_num = ['birth_year', 'salary_year',  'mon_value', 'claims_rate', 'customer_since', 'premium_total']
customer_related_cat = ['location','has_children','is_family_policy', 'educ', 'cancelled_contracts']
customer_related = customer_related_num + customer_related_cat
product_related = ['premium_motor','premium_household', 'premium_health', 'premium_life','premium_work_comp']

#### We could transform education into a intervall scala 

################# Choose algorithm #################
######### Product-related #########
### K-Means ###

# Normalization for product-related variables
scaler = StandardScaler()
prod_norm = scaler.fit_transform(df[['premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])
df_prod_norm = pd.DataFrame(prod_norm, columns = ['premium_motor','premium_household','premium_health','premium_life','premium_work_comp'])

### Find number of clusters
# Elbow graph
create_elbowgraph(10, df_prod_norm)

#Silhouette
n_clusters = 3
kmeans = KMeans(n_clusters=3, random_state=1).fit(df_prod_norm)

silhouette_avg = silhouette_score(df_prod_norm, kmeans.labels_)
print("For n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_avg) 

# Compute the silhouette scores for each sample
create_silgraph(df_prod_norm,kmeans.labels_ )

# Inverse Normalization for Interpretation
cluster_centroids_num = pd.DataFrame(scaler.inverse_transform(X=kmeans.cluster_centers_), columns = df_prod_norm.columns)

######### Customer-related ##########
### K-Prototype with categorical and numerical Features ###
# Normalization for Customer
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[['birth_year', 'salary_year', 'mon_value', 'claims_rate', 'customer_since', 'premium_total', 'cancelled_contracts']])
df_num_norm = pd.DataFrame(cust_norm, columns = ['birth_year', 'salary_year', 'mon_value', 'claims_rate', 'customer_since', 'premium_total', 'cancelled_contracts'])
df_cust_norm =df_num_norm.join(df[["educ","location","has_children", 'is_family_policy']])

# Elbow graph
create_elbowgraph(10, df_cust_norm, "kproto", [7,8,9,10] )

kproto = KPrototypes(n_clusters=3, init='random', random_state=1).fit(df_cust_norm, categorical=[7,8,9,10])

# Inverse Normalization for Interpretation
cluster_centroids_num_c = pd.DataFrame(scaler.inverse_transform(X = kproto.cluster_centroids_[0]), columns = df_num_norm.columns)
cluster_centroids_c = pd.concat([cluster_centroids_num_c,pd.DataFrame(kproto.cluster_centroids_[1])], axis=1)
cluster_centroids_c.columns = df_cust_norm.columns

#3 or 4 clusters
#if 4: 2 clusters for old ppl w/ low and high claim raie; 2 clusters for "younger" (70s)ppl with low and high claim rate
#if 3: 3 different age groups (40s, 60s, 70s) with mid high, low and very high claim rates respctivlly


################ K-Means with only numerical Features #################
# Normalization
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)

# Model fit
kmeans_cust = KMeans(n_clusters=3, random_state=1).fit(df_cust_num_norm)

# Inverse Normalization for Interpretation
cluster_centroids_cust_num = pd.DataFrame(scaler.inverse_transform(X=kmeans_cust.cluster_centers_), columns = customer_related_num)

####################################################################################
#### first kmeans with large number of clusters and then apply hierarchical clusters 

# Normalization
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)

# Kmeans fit
kmeans_cust_l = KMeans(n_clusters=100, random_state=1).fit(df_cust_num_norm)

# KMeans predict
df_cust = df[customer_related_num].copy()
df_cust["k_cluster"] = kmeans_cust_l.predict(df_cust_num_norm)

# Cluster centroids
cc_cust_num_l = pd.DataFrame(kmeans_cust_l.cluster_centers_, columns = customer_related_num)

# Create dendogram
dend = shc.dendrogram(shc.linkage(cc_cust_num_l, method='ward'))
# 4 clusters

# Agglomerative Hierarchical Clustering 
cc_cust_num_l["cluster"] = AgglomerativeClustering(n_clusters=4).fit_predict(cc_cust_num_l)

# Calculate centroids of clusters and inverse scaling for interpretation
cc_cust_num_norm = cc_cust_num_l.groupby("cluster").mean()
cc_cust_num = pd.DataFrame(scaler.inverse_transform(X=cc_cust_num_norm), columns = customer_related_num)

# Assign customer to cluster generated by hierarchical clustering
df_cust["cluster"] = [cc_cust_num_l.loc[i,"cluster"] for i in df_cust["k_cluster"].values]

# Silhoutte graph
create_silgraph(df_cust_num_norm, df_cust["cluster"])


#################################################################
################## Decision Tree classifier #####################

X = df_cust.iloc[:,:-1]
y = df_cust.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

clf = DecisionTreeClassifier()
# Fit model
clf = clf.fit(X_train,y_train)
#Predict the cluster for test data
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True,
                special_characters=True,feature_names = X.columns.values,class_names=['0','1', '3', '4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree_cluster.png')

# Predict cluster of outliers 







