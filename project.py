#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[25]:


df = pd.read_csv(r"C:\Users\admin\Downloads/Mall_customers.csv")


# In[26]:


X = df[["Annual Income (k$)"]].values  # Independent variable
y = df["Spending Score (1-100)"].values  # Dependent variable


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[29]:


y_pred = model.predict(X_test)


# In[30]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[31]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# In[32]:


df_results = pd.DataFrame({'Annual Income (k$)': X_test.flatten(), 
                           'Actual Spending Score': y_test, 
                           'Predicted Spending Score': y_pred})

fig = px.bar(df_results.melt(id_vars="Annual Income (k$)", 
                             var_name="Type", 
                             value_name="Spending Score"), 
             x="Annual Income (k$)", 
             y="Spending Score", 
             color="Type", 
             title="Actual vs Predicted Spending Score")


# In[33]:


fig.show()


# In[41]:


X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values


# In[42]:


model = KMeans(n_clusters=5, random_state=0)
model.fit(X)
y = model.predict(X)


# In[43]:


k = range(2, 15)
sse = [KMeans(n_clusters=i, random_state=0).fit(X).inertia_ for i in k]


# In[44]:


fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(x=list(k), y=sse, mode='lines+markers', name="SSE"))
fig_elbow.update_layout(title="Elbow Method: Choosing Optimal Clusters",
                        xaxis_title="Number of Clusters",
                        yaxis_title="Sum of Squared Errors (SSE)")
fig_elbow.show()


# In[45]:


cluster_counts = pd.DataFrame({'Cluster': np.unique(y), 'Count': np.bincount(y)})
fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', title="Customer Count per Cluster",
                 labels={'Cluster': 'Cluster Number', 'Count': 'Number of Customers'})
fig_bar.show()



# In[46]:


centers = pd.DataFrame(model.cluster_centers_, columns=["Annual Income (k$)", "Spending Score (1-100)"])
fig_heatmap = px.imshow(centers, text_auto=True, aspect="auto",
                        labels={'x': 'Feature', 'y': 'Cluster'}, 
                        title="Cluster Centers Heatmap")
fig_heatmap.show()

