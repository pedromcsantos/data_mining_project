#### after defining the clusters we use knn to assign the outlier-samples to the clusters





###############Detect potential outlier and drop rows with obvious mistakes###############
describe = df.describe()

salary_outlier = df[df["salary_year"]>30000*12] #maybe very rich, we will keep him for now

#plt.scatter(x= df.index, y=df["first_policy"])
#plt.show() #only one outlier, we will delete it
df = df[df["first_policy"]<10000]

#plt.scatter(x= df.index, y=df["salary_year"])
#plt.show()

#plt.scatter(x= df.index, y=df["mon_value"]) 
#plt.show()

mon_value_outlier = df[df["mon_value"]<-25000]  #we will also keep them for further analysis

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




###### Option 1 ######
# Filling nan values in educ, salary, has_children and birth_year (the wrong one) with k-prototype
# For this step we remove all rows with nan-values and outliers from the dataframe
#df_fill = df.drop(df_outlier) not necessary
df_fill = df.dropna()
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


### K-Prototype with categorical and numerical Features ###
# Normalization for Customer
scaler = StandardScaler()
cust_norm = scaler.fit_transform(df[[ 'salary_year', 'mon_value', 'claims_rate', 'premium_total']])
df_num_norm = pd.DataFrame(cust_norm, columns = [ 'salary_year', 'mon_value', 'claims_rate',  'premium_total'])
df_cust_norm =df_num_norm.join(df[["educ","location","has_children", 'cancelled_contracts', "nr_contracts"]])

# Elbow graph
# create_elbowgraph(10, df_cust_norm, "kproto", [6,7,8,9,10,11] )

kproto = KPrototypes(n_clusters=3, init='random', random_state=1).fit(df_cust_norm, categorical=[6,7,8,9,10,11])

# Inverse Normalization for Interpretation
cluster_centroids_num_c = pd.DataFrame(scaler.inverse_transform(X = kproto.cluster_centroids_[0]), columns = df_num_norm.columns)
cluster_centroids_c = pd.concat([cluster_centroids_num_c,pd.DataFrame(kproto.cluster_centroids_[1])], axis=1)
cluster_centroids_c.columns = df_cust_norm.columns

########################################################################################################################################################################
########################################################################################################################################################################
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
silhouette_avg = silhouette_score(df_cust_num_norm, df_cust["cluster"])
print("the average silhouette_score is :", silhouette_avg) 

cc_cust_num = df_cust.groupby("cluster").mean()
		
##### Reason for not using it: Similar clusters but lower silhouettes score			
			