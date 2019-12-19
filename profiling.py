import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objs as go
import plotly.offline as pyo

df = pd.read_csv("data/insurance_clusters.csv", index_col = 0)

categoricals = ["educ","location","has_children","cancelled_contracts","has_all","is_profit"]
numericals = ["salary_year","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp","premium_total"]
clusters =  ["p_cluster","c_cluster"]
df.is_profit.value_counts()

customer_related_num = ['salary_year',  'mon_value', 'claims_rate', 'premium_total', "c_cluster"]
customer_related_cat = ['location','has_children', 'educ', 'cancelled_contracts', 'has_all','is_profit']
product_related_num = ["premium_motor","premium_household","premium_health","premium_life","premium_work_comp", "p_cluster"]

customer_cluster = df[customer_related_num].groupby("c_cluster").mean()
product_cluster = df[product_related_num].groupby("p_cluster").mean()
#### Checking if clusters overlap
## create a table/Matrix
cluster_matrix = pd.crosstab(df["p_cluster"], df["c_cluster"])
#plot
trace1 = go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,0], name = "c_cluster_0") 
trace2=go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,1], name = "c_cluster_1") 
trace3=go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,2], name = "c_cluster_2")
#trace4=go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,3], name = "c_cluster_3")  
data = [trace1, trace2,trace3] #, trace4]
layout = go.Layout(title = "Cluster Combination", template = "plotly_dark", xaxis=dict(title="Product Cluster"), yaxis=dict(title="Frequency"))
fig = go.Figure(data = data, layout = layout)

fig.write_image("images/profiling/Cluster_combination.png", width=1200, height=500)


#profiling categoricals + numericals
for i in df[categoricals]:
    data = []
    for b in range(len(df[i].unique())):
        trace1 = go.Bar(x= df["c_cluster"].unique().sort(), y = (pd.crosstab(df[i], df["c_cluster"])/pd.crosstab(df[i], df["c_cluster"]).sum()).iloc[b,:], name = b) 
        data.append(trace1)
    layout = go.Layout(title = i, template = "plotly_dark", xaxis=dict(title="clusters"), yaxis=dict(title="Relative Freq."))
    fig = go.Figure(data = data, layout = layout)
    fig.update_layout(barmode = "stack")
    fig.write_image("images/profiling/"+i+".png",width=1200, height=500)

education = pd.crosstab(df["c_cluster"],df["educ"])
trace1 = go.Bar(x= education.index, y = education.iloc[:,0], name = "Basic") 
trace2=go.Bar(x= education.index, y = education.iloc[:,1], name = "HS") 
trace3=go.Bar(x= education.index, y = education.iloc[:,2], name = "BSc/MSc")
trace4=go.Bar(x= education.index, y = education.iloc[:,3], name = "PhD")
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(title = "Cluster and educ", template = "plotly_dark", xaxis=dict(title="Cluster"), yaxis=dict(title="Frequency"))
fig = go.Figure(data = data, layout = layout)
fig.update_layout(barmode = "stack")

fig.write_image("images/profiling/Education_dist.png", width=1200, height=500)

#Compute new clusters bearing in mind the profitability of the clients
profitability = pd.crosstab(df.c_cluster, df.is_profit)
#Assign clustr 0 + profit 0 to a new cluster 3
df_profiled = df.copy()
df_profiled["c_cluster"] = [3 if df_profiled.loc[i,"is_profit"] == 0 else df_profiled.loc[i,"c_cluster"] for i in df_profiled.index.values]
df_profiled["c_cluster"].value_counts()

customer_cluster_prof = df_profiled[customer_related_num].groupby("c_cluster").mean()
#for report, add plot with distibution and comment, for instance, we can see that within cluster 1 and 3 there
#are several guys with not all contracts and therefore premium total is smaller

cluster_matrix_prof = pd.crosstab(df_profiled["c_cluster"], df_profiled["p_cluster"])

### Moving clusters around 
df_profiled["p1_cluster"] = [1 if ((df_profiled.loc[i,"c_cluster"] == 0) | (df_profiled.loc[i,"c_cluster"] == 2))   else  0 for i in df_profiled.index.values]
df_profiled.p1_cluster.value_counts()

cluster_matrix_prof2 = pd.crosstab(df_profiled["c_cluster"], df_profiled["p_cluster"])

product_centroids2= df_profiled[['premium_health', 'premium_household','premium_life', 'premium_motor', 'p1_cluster', 'premium_work_comp']].groupby("p1_cluster").mean()


#Final decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
import pydotplus
import graphviz
from sklearn import tree

X = df_profiled[['salary_year', 'mon_value', 'claims_rate', 'premium_total']]
y = df_profiled["c_cluster"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

clf = DecisionTreeClassifier(max_depth=4)
# Fit model
clf = clf.fit(X_train,y_train)
#Predict the cluster for test data
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X.columns.values,
                                class_names = ['0','1', '3', '4'] ,
                                filled=True,
                                rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)


"""
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True,
                special_characters=True,feature_names = X.columns.values,class_names=['0','1', '3', '4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree_cluster.png')
"""

### Compute probability of each point to belong to the remaining clusters

#Re-assign cluster 0 + profit 0 to a new cluster 3
df_profiled_p = df.copy()
df_profiled_p["c_cluster"] = [3 if df_profiled_p.loc[i,"is_profit"] == 0 else df_profiled_p.loc[i,"c_cluster"] for i in df_profiled_p.index.values]
df_profiled_p["c_cluster"].value_counts()

customer_cluster_prof_p = df_profiled_p[customer_related_num].groupby("c_cluster").mean()
cluster_matrix_prof_p = pd.crosstab(df_profiled_p["c_cluster"], df_profiled_p["p_cluster"])


#Normalizing data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_profiled_p = df_profiled_p[product_related_num + customer_related_num].drop(["c_cluster","p_cluster"], axis=1)

df_profiled_norm = scaler.fit_transform(df_profiled_p)
df_profiled_norm = pd.DataFrame(df_profiled_norm, columns = df_profiled_p.columns.values)
df_profiled_norm["c_cluster"] = df_profiled["c_cluster"]
df_profiled_norm["p_cluster"] = df_profiled["p_cluster"]

customer_cluster_norm = df_profiled_norm[customer_related_num].groupby("c_cluster").mean()
product_cluster_norm = df_profiled_norm[product_related_num].groupby("p_cluster").mean()

customer_std_norm = df_profiled_norm[customer_related_num].groupby("c_cluster").std()
product_std_norm = df_profiled_norm[product_related_num].groupby("p_cluster").std()

from scipy.stats import norm
norm.pdf(x = df_profiled_norm.loc[10251,"salary_year"],loc = customer_cluster_norm.iloc[0][0], scale = customer_std_norm.iloc[0][0])
norm.pdf(x = df_profiled_norm.loc[10251,"salary_year"],loc = customer_cluster_norm.iloc[1][0], scale = customer_std_norm.iloc[1][0])
norm.pdf(x = df_profiled_norm.loc[10251,"salary_year"],loc = customer_cluster_norm.iloc[2][0], scale = customer_std_norm.iloc[2][0])
norm.pdf(x = df_profiled_norm.loc[10251,"salary_year"],loc = customer_cluster_norm.iloc[3][0], scale = customer_std_norm.iloc[3][0])

df_profiled_p_mov = df_profiled[((df_profiled["c_cluster"]==0)&(df_profiled["p_cluster"]==0))|((df_profiled["c_cluster"]==1)&(df_profiled["p_cluster"]==1))]

cc_01 = customer_cluster_norm.iloc[0].append(product_cluster_norm.iloc[1])
cc_10 = customer_cluster_norm.iloc[1].append(product_cluster_norm.iloc[0])
cc_21 = customer_cluster_norm.iloc[2].append(product_cluster_norm.iloc[1])
cc_31 = customer_cluster_norm.iloc[3].append(product_cluster_norm.iloc[1])

std_01 = customer_std_norm.iloc[0].append(product_std_norm.iloc[1])
std_10 = customer_std_norm.iloc[1].append(product_std_norm.iloc[0])
std_21 = customer_std_norm.iloc[2].append(product_std_norm.iloc[1])
std_31 = customer_std_norm.iloc[3].append(product_std_norm.iloc[1])

ccf_01 = pd.DataFrame([cc_01,std_01])
ccf_10 = pd.DataFrame([cc_10,std_10])
ccf_21 = pd.DataFrame([cc_21,std_21])
ccf_31 = pd.DataFrame([cc_31,std_31])

df_list = [ccf_01, ccf_10, ccf_21, ccf_31]
ccpc = [[0,1],[1,0],[2,1],[3,1]]
df_profiled_p_mov["new_cluster"] = np.nan
count=0
for i in df_profiled_p_mov.index:
    if count%100 == 0: print(count)
    count += 1
    cust = df_profiled_norm.iloc[i]
    maxprob = 0
    maxi = 0
    j = 0
    for field in df_list:
        prob = 0
        for col in field.columns.values:
            prob += norm.pdf(x = cust[col],loc = field.loc[0,col], scale = field.loc[1,col])
        if prob > maxprob:
            maxprob = prob
            maxi = j
        j += 1
    df_profiled_p_mov.loc[i,"new_cluster"] = maxi
            
    
df_profiled_p_mov.new_cluster.value_counts()
cross_test = pd.crosstab(df_profiled_p_mov["new_cluster"], df_profiled_p_mov["p_cluster"])

df_profiled["c2_cluster"] = [ccpc[int(df_profiled_p_mov.loc[c, "new_cluster"])][0]  if c in df_profiled_p_mov.index.values else df_profiled.loc[c,"c_cluster"] for c in df_profiled.index.values]
df_profiled["p2_cluster"] = [ccpc[int(df_profiled_p_mov.loc[c, "new_cluster"])][1]  if c in df_profiled_p_mov.index.values else df_profiled.loc[c,"p1_cluster"] for c in df_profiled.index.values]

cluster_matrix_prof_final = pd.crosstab(df_profiled["c2_cluster"], df_profiled["p2_cluster"])

#CHANGE TREE TO HERE 