import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score  #avg of avgs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydotplus
from kmodes.kmodes import KModes
from helperFunctions import create_silgraph, get_outliers_i, create_elbowgraph


df = pd.read_csv("data/A2Z Insurance.csv")

# Preprocessing
df = df.set_index("Customer Identity")
newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)
df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
df = df.drop("birth_year", axis=1) #Drop birth_year for clustering; consider it for interpretation
df = df[df["first_policy"]<50000] #Drop one case where first_policy year <50000 


#####################################################################################
################# Outlier #################
df.reset_index(inplace=True,drop=True)
df_num = pd.DataFrame(df[['first_policy', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])
	
# Define individual multipliers for features
thresholds = {'salary_year': 200000,'mon_value': -1000,'claims_rate': 3,'premium_motor': 600,'premium_household': 1600,'premium_health': 400,'premium_life': 300,'premium_work_comp': 300}
outliers = []
for col, th in thresholds.items():
	direct = "pos"
	if col == "mon_value":
		direct = "neg"
	outliers.append(get_outliers_i(df_num, col, th, direct))

df_outlier = df.iloc[list(set([o for l in outliers for o in l]))]
df = df[~df.index.isin(df_outlier.index.values)]

#####################################################################################
################# filling NAN #################

# Filling nan values in premium columns
#Assumption: nan values in premium mean no contract 
df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)

# Drop customers with nan values in "salary_year","educ" or "has_children" because these are only a few customers and we do not have any reasonable correlation to fill it 
df_dropped = df[df[["salary_year","educ","has_children"]].isna().any(axis=1)]
df = df.dropna(subset=["salary_year","educ","has_children"])

#######################################################################
######### Feature-engineering and -selection #########
df.reset_index(drop=True, inplace=True)

# Calculate total amount paid for premiums per year per customer
df["premium_total"] = [sum(p for p in premiums if p > 0) for i, premiums in df[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]

# True if customer cancelled contract this year (has a negative value in the premium-related columns)
temp = [sum(1 for p in premiums if p < 0) for i, premiums in df[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]
df["cancelled_contracts"] = [1 if i != 0 else 0 for i in temp]

# True if customers has premium for every part
temp = [sum(1 for p in premiums if p > 0) for i, premiums in df[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]
df["has_all"] = [1 if i == 5 else 0 for i in temp]

# Split the features in customer- and product-related. 
customer_related_num = ['salary_year', 'mon_value',  'claims_rate', 'premium_total'] # dont use first_policy because the clusters are clearer without
customer_related_cat = ['location','has_children', 'educ', 'cancelled_contracts', 'has_all']
customer_related = customer_related_num + customer_related_cat
product_related = ['premium_motor','premium_household', 'premium_health', 'premium_life','premium_work_comp']

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
df["p_cluster"] = kmeans.labels_

silhouette_avg = silhouette_score(df_prod_norm, kmeans.labels_)
print("For n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_avg) 

# Compute the silhouette scores for each sample
create_silgraph(df_prod_norm,kmeans.labels_ )

# Inverse Normalization for Interpretation
cluster_centroids_num = pd.DataFrame(scaler.inverse_transform(X=kmeans.cluster_centers_), columns = df_prod_norm.columns)

######### Customer-related ##########
################ K-Means with only numerical Features #################
# Normalization
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)

create_elbowgraph(10, df_cust_num_norm)

# Model fit
kmeans_cust = KMeans(n_clusters=4, random_state=1).fit(df_cust_num_norm)

# Model predict
df["c_cluster"] = kmeans_cust.labels_

create_silgraph(df_cust_num_norm,kmeans_cust.labels_ )
silhouette_avg = silhouette_score(df_cust_num_norm, kmeans_cust.labels_)
print("the average silhouette_score is :", silhouette_avg) 

# Inverse Normalization for Interpretation
cluster_centroids_cust_num = pd.DataFrame(scaler.inverse_transform(X=kmeans_cust.cluster_centers_), columns = customer_related_num)


#################################################################
################## Decision Tree classifier #####################
# Find most important features

X = df[customer_related_num]
y = df["c_cluster"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

clf = DecisionTreeClassifier(max_depth=4)
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

# Predict clusters of outliers and dropped customers
# c_cluster
df_add = pd.concat([df_outlier,df_dropped], axis=0)
num_cols = ['salary_year',  'mon_value', 'claims_rate']
df_topredc = pd.DataFrame(df_add[num_cols])
trained_models = {}
pred_cclusters = []
df_topredc.reset_index(drop=True, inplace=True)
df_add.reset_index(drop=True, inplace=True)

for i in df_topredc.index.values:
	isna = df_topredc.iloc[i,:].isna()
	cols = [num_cols[j] for j in range(0,len(num_cols)) if isna[j] == False]
	if ', '.join(cols) in trained_models.keys():
		y_pred = trained_models[', '.join(cols)].predict([df_topredc.loc[i,cols]])
		pred_cclusters.append(y_pred[0])
		continue	
	else:	
		X = df[cols]
		y = df["c_cluster"]
		clf = DecisionTreeClassifier()
		clf = clf.fit(X,y)
		y_pred = clf.predict([df_topredc.loc[i,cols]])
		pred_cclusters.append(y_pred[0])
		trained_models[', '.join(cols)] = clf

# p_cluster 
pcols = ["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
#Assumption: nan values in premium mean no contract 
df_topredp = pd.DataFrame(df_add[pcols])
df_topredp = df_topredp.fillna(0)

X = df[pcols]
y = df["p_cluster"]
clf = DecisionTreeClassifier()
clf = clf.fit(X,y)
pred_pclusters = clf.predict(df_topredp)

# Add outliers and dropped customers to main dataframe 

df_add["premium_total"] = [sum(p for p in premiums if p > 0) for i, premiums in df_add[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]
temp = [sum(1 for p in premiums if p < 0) for i, premiums in df_add[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]
df_add["cancelled_contracts"] = [1 if i != 0 else 0 for i in temp]
temp = [sum(1 for p in premiums if p > 0) for i, premiums in df_add[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]
df_add["has_all"] = [1 if i == 5 else 0 for i in temp]

df_add["c_cluster"] = pred_cclusters
df_add["p_cluster"] = pred_pclusters

df = df.append(df_add)

#filling NAN in premiums with zeros (see assumption above)
df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)

#Missing values in salary_year(numerical): calculate salary_year mean for each cluster
salary_mean = df.pivot_table(values="salary_year",index="c_cluster",aggfunc="mean")
#Missing values in education and has_children(categorival): calculate mode for each cluster
cat_mode = df[["has_children", "educ", "c_cluster"]].groupby("c_cluster").agg(lambda x: x.mode())

#filling missing data by mean and mode of each cluster
for i in range(4):
    df["salary_year"][df["c_cluster"]==i]=df["salary_year"][df["c_cluster"]==i].fillna(salary_mean.iloc[i][0])
    df["educ"][df["c_cluster"]==i] = df["educ"][df["c_cluster"]==i].fillna(cat_mode.iloc[i][1])
    df["has_children"][df["c_cluster"]==i]=df["has_children"][df["c_cluster"]==i].fillna(cat_mode.iloc[i][0])
	
print(df.isnull().sum())
df.to_csv("data/insurance_clusters.csv")



