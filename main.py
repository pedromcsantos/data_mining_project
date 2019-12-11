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
from helperFunctions import create_silgraph, get_outliers_i, create_elbowgraph
from sompy.sompy import SOMFactory
import matplotlib.pyplot as plt
from sompy.visualization.mapview import View2DPacked
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView


df = pd.read_csv("data/A2Z Insurance.csv")

# Preprocessing
df = df.set_index("Customer Identity")
newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)
df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
df = df.drop("birth_year", axis=1) #Drop birth_year for clustering; consider it for interpretation
df = df[df["first_policy"]<50000] #Drop one case where first_policy year <50000 
#df = df[~(df["premium_motor"]==0)&~(df["premium_household"]==0)&~(df["premium_health"]==0)&~(df["premium_life"]==0)&~(df["premium_work_comp"]==0)]
df = df[df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].sum(axis=1)!=0]

#####################################################################################
################# Outlier #################
df.reset_index(inplace=True,drop=True)
df_num = pd.DataFrame(df[['first_policy', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])
# Define individual multipliers for features
thresholds = {'salary_year': 200000,'mon_value': -200,'claims_rate': 3,'premium_motor': 600,'premium_household': 1600,'premium_health': 400,'premium_life': 300,'premium_work_comp': 300}
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

#Calculate if customers are profitable
df["is_profit"] = [1 if mon_value > 0 else 0 for mon_value in df.mon_value.values]

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
prod_norm = scaler.fit_transform(df[product_related])
df_prod_norm = pd.DataFrame(prod_norm, columns = product_related)

### Find number of clusters
# Elbow graph
create_elbowgraph(10, df_prod_norm)

#Silhouette
kmeans = KMeans(n_clusters=2, random_state=1).fit(df_prod_norm)
df["p_cluster"] = kmeans.labels_

create_silgraph(df_prod_norm, df["p_cluster"])
silhouette_avg = silhouette_score(df_prod_norm, kmeans.labels_)
print("For n_clusters =", str(2), "the average silhouette_score is :", silhouette_avg) 

# Compute the silhouette scores for each sample
#create_silgraph(df_prod_norm,kmeans.labels_ )

# Inverse Normalization for Interpretation
pcluster_centroids_num = pd.DataFrame(scaler.inverse_transform(X=kmeans.cluster_centers_), columns = df_prod_norm.columns)


######### Customer-related ##########
############ SOM and Hierarchical Clustering #####################
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_cust_norm = pd.DataFrame(cust_norm, columns = customer_related_num)

X = df_cust_norm.values

sm = SOMFactory().build(data = X,
               mapsize=(8,8),
               normalization = 'var',
               initialization="pca",
               component_names=customer_related_num,
               lattice="hexa",
               training ="batch" )

sm.train(n_job=5,
         verbose='info',
         train_rough_len=40,
         train_finetune_len=100)

final_clusters = pd.DataFrame(sm._data, columns = customer_related_num)
my_labels = pd.DataFrame(sm._bmu[0])    
final_clusters = pd.concat([final_clusters,my_labels], axis = 1)
cluster_cols = customer_related_num  + ["Labels"]
final_clusters.columns = cluster_cols

view2D  = View2DPacked(20,20,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
plt.show()

view2D  = View2D(20,20,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
plt.show()
vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="autumn", logaritmic=False)

## Hierarchical Clustering ##
som_cluster = final_clusters.groupby("Labels").mean()
dend = shc.dendrogram(shc.linkage(som_cluster, method='ward'))

som_cluster["h_cluster"] = AgglomerativeClustering(n_clusters=3).fit_predict(som_cluster)
# Calculate centroids of clusters and inverse scaling for interpretation
h_cluster = som_cluster.groupby("h_cluster").mean()
h_cluster = pd.DataFrame(scaler.inverse_transform(X=h_cluster), columns = customer_related_num)
# Assign customer to cluster generated by hierarchical clustering
final_clusters["h_cluster"] = [som_cluster.loc[label,"h_cluster"] for label in final_clusters["Labels"].values]
# Silhoutte graph
create_silgraph(df_cust_norm, final_clusters["somh_cluster"])
silhouette_avg = silhouette_score(df_cust_norm, final_clusters["h_cluster"])
print("the average silhouette_score is :", silhouette_avg) 
df["c_cluster"] = final_clusters["h_cluster"]

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
#graph.write_png('decision_tree_cluster.png')

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
df = df.reset_index(drop=True)

#filling NAN in premiums with zeros (see assumption above)
df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)

#Missing values in salary_year(numerical): calculate salary_year mean for each cluster
salary_mean = df.pivot_table(values="salary_year",index="c_cluster",aggfunc="mean")
#Missing values in education and has_children(categorival): calculate mode for each cluster
cat_mode = df[["has_children", "educ", "c_cluster"]].groupby("c_cluster").agg(lambda x: x.mode())

#filling missing data by mean and mode of each cluster
for i in range(3):
    df["salary_year"][df["c_cluster"]==i]=df["salary_year"][df["c_cluster"]==i].fillna(salary_mean.iloc[i][0])
    df["educ"][df["c_cluster"]==i] = df["educ"][df["c_cluster"]==i].fillna(cat_mode.iloc[i][1])
    df["has_children"][df["c_cluster"]==i]=df["has_children"][df["c_cluster"]==i].fillna(cat_mode.iloc[i][0])

df["is_profit"] = [1 if mon_value > 0 else 0 for mon_value in df.mon_value.values]

print(df.isnull().sum())
df.to_csv("data/insurance_clusters.csv")


