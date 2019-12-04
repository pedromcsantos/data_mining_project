import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score  #avg of avgs
from helperFunctions import create_silgraph, get_outliers_i, create_elbowgraph
import plotly.offline as pyo
import plotly.figure_factory as ff



def create_distplot(df, col, bin_size):
	x = df.loc[:,col]
	x = x.dropna()
	hist_data = [x]
	group_labels = ['distplot']

	fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size)
	fig.write_image(col + "_distplot.png")
	fig.show()
	
df = pd.read_csv("data/A2Z Insurance.csv")

# Preprocessing
df = df.set_index("Customer Identity")
newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)
df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
df = df.drop("birth_year", axis=1) #Drop birth_year for clustering; consider it for interpretation
df = df[df["first_policy"]<50000] #Drop one case where first_policy year <50000 


# Plot distribution
num_col = ['first_policy','salary_year', 'mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']
bin_size = [0, 1000, 10000, 0.4, 100,100,100,100,100]
for i in range(0,len(num_col)):
	create_distplot(df,num_col[i], bin_size[i])

plt.scatter(x= df.index, y=df["salary_year"])

plt.scatter(x= df.index, y=df["first_policy"])
plt.show() #only one outlier, we will delete it
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



# After plotting the features we decided to test the following outlier variations:
# Customer related features

cluster_centroids = []

variations = [
		{'salary_year': 0,'mon_value': 0,'claims_rate': 0,'premium_motor': 0,'premium_household': 0,'premium_health': 0,'premium_life': 0,'premium_work_comp': 0}, # include all, good sil score but centroids too skewed by outliers
		{'salary_year': 0,'mon_value': 5,'claims_rate': 5,'premium_motor': 5,'premium_household': 5,'premium_health': 5,'premium_life': 5,'premium_work_comp': 5},
		{'salary_year': 5,'mon_value': 0,'claims_rate': 0,'premium_motor': 5,'premium_household': 5,'premium_health': 5,'premium_life': 5,'premium_work_comp': 5},
		{'salary_year': 5,'mon_value': 5,'claims_rate': 5,'premium_motor': 5,'premium_household': 5,'premium_health': 5,'premium_life': 5,'premium_work_comp': 5},
		{'salary_year': 10,'mon_value': 10,'claims_rate': 10,'premium_motor': 10,'premium_household': 10,'premium_health': 10,'premium_life': 10,'premium_work_comp': 10},
		{'salary_year': 3,'mon_value': 3,'claims_rate': 3,'premium_motor': 3,'premium_household': 3,'premium_health': 3,'premium_life': 3,'premium_work_comp': 3},
		{'salary_year': 1.5,'mon_value': 1.5,'claims_rate': 1.5,'premium_motor': 1.5,'premium_household': 1.5,'premium_health': 1.5,'premium_life': 1.5,'premium_work_comp': 1.5} # very strict
		]


for v in variations:
	df = pd.read_csv("A2Z Insurance.csv")

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
	multipliers = v
	outliers = []
	for col, multi in multipliers.items():
		outliers.append(get_outliers_i(df_num, col, multi))
	
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

	# Split the features in customer- and product-related. 
	customer_related_num = ['salary_year', 'mon_value',  'claims_rate', 'premium_total'] # dont use first_policy because the clusters are clearer without
	
	######### Customer-related ##########
	################ K-Means with only numerical Features #################
	# Normalization
	scaler = StandardScaler()
	cust_norm = scaler.fit_transform(df[customer_related_num])
	df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)
		
	# Model fit
	kmeans_cust = KMeans(n_clusters=4, random_state=1).fit(df_cust_num_norm)
	
	# Model predict
	df["c_cluster"] = kmeans_cust.labels_
	
	silhouette_avg = silhouette_score(df_cust_num_norm, kmeans_cust.labels_)
	print("the average silhouette_score for " + str(v) + " is " + str(silhouette_avg)) 
	
	# Inverse Normalization for Interpretation
	cc = pd.DataFrame(scaler.inverse_transform(X=kmeans_cust.cluster_centers_), columns = customer_related_num)
	
	cluster_centroids.append(dict(key=v, value=cc))


