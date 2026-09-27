# Import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default')

# Upload dataset
data = pd.read_csv('Market_Basket_Optimisation-7bd7 (1).csv', header=None)

data.shape
data.head()
data.describe()
data[1]

# Item frequency
transaction = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        transaction.append(data.values[i, j])

transaction = np.array(transaction)

df = pd.DataFrame(transaction, columns=["items"])
df["incident_count"] = 1
df = df[df["items"] != "nan"]

df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

df_table.head(10).style.background_gradient(cmap='Blues')

df_table["all"] = "all"

fig = px.treemap(df_table.head(30), path=['all', "items"], values='incident_count',
                 color=df_table["incident_count"].head(30), hover_data=['items'],
                 color_continuous_scale='Blues')
fig.show()

# Check multiple records
transaction = []
for i in range(data.shape[0]):
    transaction.append([str(data.values[i, j]) for j in range(data.shape[1])])
transaction = np.array(transaction)

top20 = df_table["items"].head(20).values
df_top20_multiple_record_check = pd.DataFrame(columns=top20)

for item in top20:
    counts = []
    for j in range(transaction.shape[0]):
        counts.append(np.count_nonzero(transaction[j] == item))
    df_top20_multiple_record_check[item] = counts

df_top20_multiple_record_check.head(10)
df_top20_multiple_record_check.describe()

# First choices
transaction = [data.values[i, 0] for i in range(data.shape[0])]

df_first = pd.DataFrame(transaction, columns=["items"])
df_first["incident_count"] = 1
df_first = df_first[df_first["items"] != "nan"]

df_table_first = df_first.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
df_table_first["food"] = "food"
df_table_first = df_table_first.head(15)

plt.rcParams['figure.figsize'] = (20, 20)

first_choice = nx.from_pandas_edgelist(df_table_first, source='food', target="items", edge_attr=True)
pos = nx.spring_layout(first_choice)

nx.draw_networkx_nodes(first_choice, pos, node_size=12500, node_color="lavender")
nx.draw_networkx_edges(first_choice, pos, width=3, alpha=0.6, edge_color='black')
nx.draw_networkx_labels(first_choice, pos, font_size=18, font_family='sans-serif')

plt.axis('off')
plt.grid()
plt.title('Top 15 First Choices', fontsize=25)
plt.show()

# Second choices
transaction = [data.values[i, 1] for i in range(data.shape[0])]

df_second = pd.DataFrame(transaction, columns=["items"])
df_second["incident_count"] = 1
df_second = df_second[df_second["items"] != "nan"]

df_table_second = df_second.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
df_table_second["food"] = "food"
df_table_second = df_table_second.head(15)

second_choice = nx.from_pandas_edgelist(df_table_second, source='food', target="items", edge_attr=True)
pos = nx.spring_layout(second_choice)

nx.draw_networkx_nodes(second_choice, pos, node_size=12500, node_color="honeydew")
nx.draw_networkx_edges(second_choice, pos, width=3, alpha=0.6, edge_color='black')
nx.draw_networkx_labels(second_choice, pos, font_size=18, font_family='sans-serif')

plt.axis('off')
plt.grid()
plt.title('Top 15 Second Choices', fontsize=25)
plt.show()

# Third choices
transaction = [data.values[i, 2] for i in range(data.shape[0])]

df_third = pd.DataFrame(transaction, columns=["items"])
df_third["incident_count"] = 1
df_third = df_third[df_third["items"] != "nan"]

df_table_third = df_third.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
df_table_third["food"] = "food"
df_table_third = df_table_third.head(15)

fig = go.Figure(data=[go.Bar(x=df_table_third["items"], y=df_table_third["incident_count"],
                             hovertext=df_table_third["items"], text=df_table_third["incident_count"],
                             textposition="outside")])

fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.65)
fig.update_layout(title_text="Customers' Third Choices", template="plotly_dark")
fig.show()

# Prepare transactions
transaction = []
for i in range(data.shape[0]):
    transaction.append([str(data.values[i, j]) for j in range(data.shape[1]) if pd.notna(data.values[i, j])])

te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

dataset.shape

# Select top 50 items
first50 = df_table["items"].head(50).values
dataset = dataset.loc[:, first50]

# Convert to 0/1
dataset = dataset.astype(int)
dataset.head(10)

# Frequent itemsets
frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

frequent_itemsets

frequent_itemsets[(frequent_itemsets["length"] == 2) & (frequent_itemsets["support"] >= 0.05)]

frequent_itemsets[frequent_itemsets["length"] == 3].head()

# Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules["antecedents_length"] = rules["antecedents"].apply(len)
rules["consequents_length"] = rules["consequents"].apply(len)

rules.sort_values("lift", ascending=False)
rules.sort_values("confidence", ascending=False)

# Exclude mineral water
rules[~rules["consequents"].str.contains("mineral water", regex=False) &
      ~rules["antecedents"].str.contains("mineral water", regex=False)].sort_values(
          "confidence", ascending=False).head(10)

# Ground beef associations
rules[rules["antecedents"].apply(lambda x: "ground beef" in x) &
      (rules["antecedents_length"] == 1)].sort_values(
          "confidence", ascending=False).head(10)