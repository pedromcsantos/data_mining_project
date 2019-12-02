import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objs as go
import plotly.offline as pyo

df = pd.read_csv("insurance_clusters.csv", index_col = 0)

categoricals = ["educ","location","has_children","is_family_policy","cancelled_contracts","nr_contracts", "is_maxed"]
numericals = ["birth_year","salary_year","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp","customer_since","premium_total"]
clusters =  ["p_cluster","c_cluster"]

customer_related_num = ['birth_year', 'salary_year',  'mon_value', 'claims_rate', 'customer_since', 'premium_total', "c_cluster"]
customer_related_cat = ['location','has_children','is_family_policy', 'educ', 'cancelled_contracts', 'nr_contracts']
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
trace4=go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,3], name = "c_cluster_3")  
data = [trace1, trace2,trace3, trace4]
layout = go.Layout(title = "Cluster Combination", template = "plotly_dark", xaxis=dict(title="Product Cluster"), yaxis=dict(title="Frequency"))
fig = go.Figure(data = data, layout = layout)

fig.write_image("Cluster_combination.png", width=1200, height=500)
###Plot customer cluster by categoricals and see if there are too differentclusters inside the clusters -> probably not

customer_related = customer_related_num + customer_related_cat

education = df[customer_related].groupby(["c_cluster","educ"]).mean() #Education doesnt affect that much the clusters
location = df[customer_related].groupby(["c_cluster","location"]).mean() #Location doesnt affect clusters
has_children = df[customer_related].groupby(["c_cluster","has_children"]).mean() #some influence
is_family_policy = df[customer_related].groupby(["c_cluster","is_family_policy"]).mean() #some influence 
cancelled_contracts = df[customer_related].groupby(["c_cluster","cancelled_contracts"]).mean() #some influence
nr_contracts = df[customer_related].groupby(["c_cluster","nr_contracts"]).mean() #some influence but groups are small/not representative below 4 contracts

nr_contracts_matrix = pd.crosstab(df["c_cluster"], df["nr_contracts"])

##Extra exploration in Number of Contracts
for i in df.index.values:
    b = df.loc[i,"nr_contracts"]
    if b == 5:
        df.loc[i,"is_maxed"] = 1 #max number of contracts 
    else:
        df.loc[i,"is_maxed"] = 0

customer_related = customer_related + ["is_maxed"]
     
is_maxed = df[customer_related].groupby(["c_cluster","is_maxed"]).mean()

#profiling categoricals + numericals
for i in df[categoricals]:
    trace1 = go.Bar(x= df[i].unique().sort(), y = pd.crosstab(df[i], df["c_cluster"])[0], name = "c_cluster_0") 
    trace2=go.Bar(x= df[i].unique().sort(), y =  pd.crosstab(df[i], df["c_cluster"])[1], name = "c_cluster_1") 
    trace3=go.Bar(x= df[i].unique().sort(), y =  pd.crosstab(df[i], df["c_cluster"])[2], name = "c_cluster_2")
    trace4=go.Bar(x= df[i].unique().sort(), y =  pd.crosstab(df[i], df["c_cluster"])[3], name = "c_cluster_3")
    
    data = [trace1, trace2,trace3, trace4]
    layout = go.Layout(title = i, template = "plotly_dark", xaxis=dict(title=i), yaxis=dict(title="Frequency"))
    fig = go.Figure(data = data, layout = layout)
    fig.write_image(i+".png",width=1200, height=500)

### Checcking for young ppl info
young_kids = df[df["birth_year"]>=1998]
young_kids["has_children"].value_counts()
young_kids["premium_motor"][young_kids["premium_motor"]>0].value_counts().sum() #195 out of 195 lol

