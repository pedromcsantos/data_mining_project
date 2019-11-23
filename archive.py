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
			
			
			