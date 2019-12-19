import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score
from helperFunctions import create_silgraph, get_outliers_i, create_elbowgraph
from sompy.sompy import SOMFactory
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

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
################# filling NAN #################
# Filling nan values in premium columns
#Assumption: nan values in premium mean no contract 
df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)
# Drop customers with nan values in "salary_year","educ" or "has_children" because these are only a few customers and we do not have any reasonable correlation to fill it 
df_dropped = df[df[["salary_year","educ","has_children"]].isna().any(axis=1)]
df = df.dropna(subset=["salary_year","educ","has_children"])
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
customer_related_cat = ['location','has_children', 'educ', 'cancelled_contracts', 'has_all', "is_profit"]
customer_related = customer_related_num + customer_related_cat
product_related = ['premium_motor','premium_household', 'premium_health', 'premium_life','premium_work_comp']


############################# Approaches ###################################
#############Customers#############
########Numerical###########
###1. Approach: K-Means 
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)
create_elbowgraph(10, df_cust_num_norm)
kmeans_cust = KMeans(n_clusters=3, random_state=1).fit(df_cust_num_norm)
df["kmc_cluster"] = kmeans_cust.labels_
create_silgraph(df_cust_num_norm,kmeans_cust.labels_ )
silhouette_avg = silhouette_score(df_cust_num_norm, kmeans_cust.labels_)
cc_kmeans = pd.DataFrame(scaler.inverse_transform(X=kmeans_cust.cluster_centers_), columns = customer_related_num)
sizes = df["kmc_cluster"].value_counts() / len(df["kmc_cluster"])
ap_kmeans = {"cc": cc_kmeans, "sil_score": silhouette_avg, "sizes": sizes}


### 2. Approach: SOM followed by K-Means
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
som_cluster = final_clusters.groupby("Labels").mean()
#create_elbowgraph(10, som_cluster)
kmeans = KMeans(n_clusters=3, random_state=1).fit(som_cluster)
som_cluster["somk_cluster"] = kmeans.labels_
k_cluster = som_cluster.groupby("somk_cluster").mean()
k_cluster = pd.DataFrame(scaler.inverse_transform(X=k_cluster), columns = customer_related_num)
final_clusters["somk_cluster"] = [som_cluster.loc[i, "somk_cluster"] for i in final_clusters["Labels"].values ]
#create_silgraph(df_cust_norm, final_clusters["k_cluster"])
silhouette_avg = silhouette_score(df_cust_norm, final_clusters["somk_cluster"])
df["somkmc_cluster"] = final_clusters["somk_cluster"]
sizes = df["somkmc_cluster"].value_counts()/ len(df["somkmc_cluster"])
ap_somkmeans = {"cc": k_cluster, "sil_score": silhouette_avg, "sizes": sizes}


#### 3.Approach: K-means with large number of clusters and then apply hierarchical clustering 
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
plt.title("Dendogram")
# Agglomerative Hierarchical Clustering 
cc_cust_num_l["h_cluster"] = AgglomerativeClustering(n_clusters=3).fit_predict(cc_cust_num_l)
# Calculate centroids of clusters and inverse scaling for interpretation
cc_cust_num_norm = cc_cust_num_l.groupby("h_cluster").mean()
cc_cust_num = pd.DataFrame(scaler.inverse_transform(X=cc_cust_num_norm), columns = customer_related_num)
# Assign customer to cluster generated by hierarchical clustering
df_cust["h_cluster"] = [cc_cust_num_l.loc[i,"h_cluster"] for i in df_cust["k_cluster"].values]
# Silhoutte graph
#create_silgraph(df_cust_num_norm, df_cust["cluster"])
silhouette_avg = silhouette_score(df_cust_num_norm, df_cust["h_cluster"])
cc_cust_num = df_cust.groupby("h_cluster").mean()
sizes = df_cust["h_cluster"].value_counts()/ len(df_cust["h_cluster"])
ap_kmeanshierach = {"cc": cc_cust_num, "sil_score": silhouette_avg, "sizes": sizes}
		
############ 4. Approach: SOM and Hierarchical Clustering #####################
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
## Hierarchical Clustering ##
som_cluster = final_clusters.groupby("Labels").mean()
dend = shc.dendrogram(shc.linkage(som_cluster, method='ward'))
som_cluster["h_cluster"] = AgglomerativeClustering(n_clusters=3).fit_predict(som_cluster)
# Calculate centroids of clusters and inverse scaling for interpretation
h_cluster_norm = som_cluster.groupby("h_cluster").mean()
h_cluster = pd.DataFrame(scaler.inverse_transform(X=h_cluster_norm), columns = customer_related_num)
# Assign customer to cluster generated by hierarchical clustering
final_clusters["h_cluster"] = [som_cluster.loc[label,"h_cluster"] for label in final_clusters["Labels"].values]
# Silhoutte graph
#create_silgraph(df_cust_norm, final_clusters["somh_cluster"])
silhouette_avg = silhouette_score(df_cust_norm, final_clusters["h_cluster"])
df["c_cluster"] = final_clusters["h_cluster"]	
sizes = df["c_cluster"].value_counts()/ len(df["c_cluster"])
ap_somhierarch = {"cc": h_cluster, "sil_score": silhouette_avg, "sizes": sizes}

# Applying EM on our final clusters
from sklearn import mixture
gmm = mixture.GaussianMixture(n_components= 3, means_init=h_cluster_norm)
gmm.fit(df_cust_norm)
EM_labels_ = gmm.predict(df_cust_norm)
d_prob = gmm.predict_proba(df_cust_norm)
sit = scaler.inverse_transform(gmm.means_)

#### 5.Approach: DBSCAN 
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)
db = DBSCAN(eps= 0.36,min_samples=15).fit(df_cust_num_norm)

df["labels"] = db.labels_
cols = customer_related_num + ["labels"]
cc_dbscan = df[cols].groupby("labels").mean()
sizes = df["labels"].value_counts()

#### 6.Approach: Mean shift
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)

my_bandwidth = estimate_bandwidth(df_cust_num_norm,quantile=0.2,n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth, bin_seeding=True)

ms.fit(df_cust_num_norm)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
df["Labels"] = ms.predict(df_cust_num_norm)
cols = customer_related_num + ["labels"]
cc_mshift = df[cols].groupby("labels").mean()
sizes = df["labels"].value_counts()

######## Categorical ###########
### 1. Approach: K-Prototype with categorical and numerical Features
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[customer_related_num])
df_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)
df_cust_norm =df_num_norm.join(df[customer_related_cat])
# create_elbowgraph(10, df_cust_norm, "kproto", [4,5,6,7,8] )
kproto = KPrototypes(n_clusters=3, init='random', random_state=1)
model = kproto.fit(df_cust_norm, categorical=[4,5,6,7,8,9])
# Inverse Normalization for Interpretation
cc_kproto_num = pd.DataFrame(scaler.inverse_transform(X = model.cluster_centroids_[0]))
cc_kproto = pd.concat([cc_kproto_num,pd.DataFrame(model.cluster_centroids_[1])], axis=1)
cc_kproto.columns = customer_related



###### 2. Approach: Categorical Kmodes ########
kmodes = KModes(n_clusters=4)
temp_kmodes = kmodes.fit_predict(df[customer_related_cat])
kmcc = pd.DataFrame(kmodes.cluster_centroids_, columns=customer_related_cat)

df["cat_cluster"] = temp_kmodes

########################################################################################################################################################################
########################################################################################################################################################################
############# Products #############
#### 1. Approach: SOM and Hierarchical Clustering #####################
scaler = StandardScaler()
prod_norm = scaler.fit_transform(df[product_related])
df_prod_norm = pd.DataFrame(prod_norm, columns = product_related)
X = df_prod_norm.values

sm = SOMFactory().build(data = X,mapsize=(8,8), normalization = 'var', initialization="pca",
               component_names=product_related,lattice="hexa",training ="batch" )
sm.train(n_job=5,verbose='info',train_rough_len=40,train_finetune_len=100)

final_clusters = pd.DataFrame(sm._data, columns = product_related)
my_labels = pd.DataFrame(sm._bmu[0])    
final_clusters = pd.concat([final_clusters,my_labels], axis = 1)
cluster_cols = product_related  + ["Labels"]
final_clusters.columns = cluster_cols
som_cluster = final_clusters.groupby("Labels").mean()
dend = shc.dendrogram(shc.linkage(som_cluster, method='ward'))
plt.title("Dendogram after SOM with product features")
som_cluster["h_cluster"] = AgglomerativeClustering(n_clusters=2).fit_predict(som_cluster)
# Calculate centroids of clusters and inverse scaling for interpretation
h_cluster = som_cluster.groupby("h_cluster").mean()
h_cluster = pd.DataFrame(scaler.inverse_transform(X=h_cluster), columns = product_related)
# Assign customer to cluster generated by hierarchical clustering
final_clusters["h_cluster"] = [som_cluster.loc[label,"h_cluster"] for label in final_clusters["Labels"].values]
#create_silgraph(df_cust_norm, final_clusters["somh_cluster"])
silhouette_avg = silhouette_score(df_prod_norm, final_clusters["h_cluster"])
df["p_cluster"] = final_clusters["h_cluster"]

sizes = df["p_cluster"].value_counts()/ len(df["p_cluster"])
ap_psomhierarch = {"cc": h_cluster, "sil_score": silhouette_avg, "sizes": sizes}

##### 2. Approach: Kmeans
scaler = StandardScaler()
prod_norm = scaler.fit_transform(df[product_related])
df_prod_norm = pd.DataFrame(prod_norm, columns = product_related)
create_elbowgraph(10, df_prod_norm)

kmeans = KMeans(n_clusters=2, random_state=1).fit(df_prod_norm)
df["p_cluster"] = kmeans.labels_
silhouette_avg = silhouette_score(df_prod_norm, kmeans.labels_)
cc_pkmeans = pcluster_centroids_num = pd.DataFrame(scaler.inverse_transform(X=kmeans.cluster_centers_), columns = df_prod_norm.columns)
sizes = df["p_cluster"].value_counts()/ len(df["p_cluster"])
ap_pkmeans = {"cc": cc_pkmeans, "sil_score": silhouette_avg, "sizes": sizes}

### 3. Approach: SOM followed by K-Means
scaler = StandardScaler()
prod_norm = scaler.fit_transform(df[product_related])
df_prod_norm = pd.DataFrame(prod_norm, columns = product_related)

X = df_prod_norm.values
sm = SOMFactory().build(data = X,
               mapsize=(8,8),
               normalization = 'var',
               initialization="pca",
               component_names=product_related,
               lattice="hexa",
               training ="batch" )
sm.train(n_job=5,
         verbose='info',
         train_rough_len=40,
         train_finetune_len=100)
final_clusters = pd.DataFrame(sm._data, columns = product_related)
my_labels = pd.DataFrame(sm._bmu[0])    
final_clusters = pd.concat([final_clusters,my_labels], axis = 1)
cluster_cols = product_related  + ["Labels"]
final_clusters.columns = cluster_cols
som_cluster = final_clusters.groupby("Labels").mean()
#create_elbowgraph(10, som_cluster)
kmeans = KMeans(n_clusters=3, random_state=1).fit(som_cluster)
som_cluster["somk_cluster"] = kmeans.labels_
k_cluster = som_cluster.groupby("somk_cluster").mean()
k_cluster = pd.DataFrame(scaler.inverse_transform(X=k_cluster), columns = product_related)
final_clusters["somk_cluster"] = [som_cluster.loc[i, "somk_cluster"] for i in final_clusters["Labels"].values ]
#create_silgraph(df_cust_norm, final_clusters["k_cluster"])
silhouette_avg = silhouette_score(df_prod_norm, final_clusters["somk_cluster"])
df["somkmc_cluster"] = final_clusters["somk_cluster"]
sizes = df["somkmc_cluster"].value_counts()/ len(df["somkmc_cluster"])
ap_psomkmeans = {"cc": k_cluster, "sil_score": silhouette_avg, "sizes": sizes}

#### 4.Approach: K-means with large number of clusters and then apply hierarchical clustering 
scaler = StandardScaler()
prod_norm = scaler.fit_transform(df[product_related])
df_prod_norm = pd.DataFrame(prod_norm, columns = product_related)

# Kmeans fit
kmeans_prod_l = KMeans(n_clusters=100, random_state=1).fit(df_prod_norm)
# KMeans predict
df_prod = df[product_related].copy()
df_prod["k_cluster"] = kmeans_prod_l.predict(df_prod_norm)
# Cluster centroids
cc_prod_num_l = pd.DataFrame(kmeans_prod_l.cluster_centers_, columns = product_related)
# Create dendogram
#dend = shc.dendrogram(shc.linkage(cc_prod_num_l, method='ward'))
#plt.title("Dendogram")
# Agglomerative Hierarchical Clustering 
cc_prod_num_l["h_cluster"] = AgglomerativeClustering(n_clusters=3).fit_predict(cc_prod_num_l)
# Calculate centroids of clusters and inverse scaling for interpretation
cc_prod_num_norm = cc_prod_num_l.groupby("h_cluster").mean()
cc_prod_num = pd.DataFrame(scaler.inverse_transform(X=cc_prod_num_norm), columns = product_related)
# Assign customer to cluster generated by hierarchical clustering
df_prod["h_cluster"] = [cc_prod_num_l.loc[i,"h_cluster"] for i in df_prod["k_cluster"].values]
# Silhoutte graph
#create_silgraph(df_prod_norm, df_prod["cluster"])
silhouette_avg = silhouette_score(df_prod_norm, df_prod["h_cluster"])
cc_prod = df_prod.groupby("h_cluster").mean()
sizes = df_prod["h_cluster"].value_counts()/ len(df_prod["h_cluster"])
ap_pkmeanshierach = {"cc": cc_prod, "sil_score": silhouette_avg, "sizes": sizes}


#### 5.Approach: DBSCAN 
scaler = StandardScaler()
prod_norm = scaler.fit_transform(df[product_related])
df_prod_norm = pd.DataFrame(prod_norm, columns = product_related)
db = DBSCAN(eps= 0.36,min_samples=15).fit(df_prod_norm)

df["labels"] = db.labels_
cols = product_related + ["labels"]
cc_dbscan = df[cols].groupby("labels").mean()
sizes = df["labels"].value_counts()

#### 6.Approach: Mean shift
scaler = StandardScaler()
prod_norm = scaler.fit_transform(df[product_related])
df_prod_norm = pd.DataFrame(prod_norm, columns = product_related)

my_bandwidth = estimate_bandwidth(df_prod_norm,quantile=0.2,n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth, bin_seeding=True)

ms.fit(df_prod_norm)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
df["Labels"] = ms.predict(df_prod_norm)
cols = customer_related_num + ["labels"]
cc_mshift = df[cols].groupby("labels").mean()
sizes = df["labels"].value_counts()




