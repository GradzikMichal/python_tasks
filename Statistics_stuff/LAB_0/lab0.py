import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#######################################################
# ZAD 1
print('#'*30)
print("Zad 1")
df = pd.read_csv("WeightLoss.csv")
print(df.head(20))
####################################
# ZAD 2
print('#'*30)
print("Zad 2")
print(df.describe())
#####################################
# ZAD 3
print('#'*30)
print("Zad 3")
df['sum_of_wl'] = df['wl1'] + df['wl2'] + df['wl3']
print(df.loc[df['sum_of_wl'].idxmax()])
########################################
# ZAD 4
print('#'*30)
print("Zad 4")
print(df.loc[df[['se1', 'se2', 'se3']].mean(axis=1).idxmin()])
#########################################
# ZAD5
print('#'*30)
print("Zad 5")
print(df.sort_values(by=['wl2'], ascending=False))
############################################
# ZAD 6
print('#'*30)
print("Zad 6")
print(df.loc[((df['wl1'] > 4) | (df['wl2'] > 4) | (df['wl3'] > 4))])
############################################
# ZAD 7
print('#'*30)
print("Zad 7")
df['se_mean'] = df[['se1', 'se2', 'se3']].mean(axis=1)
print(df['se_mean'])
############################################
# ZAD 8
print('#'*30)
print("Zad 8")
print(df.loc[((df['se_mean'] >= 12) & (df['se_mean'] <= 15))])
############################################
# ZAD 9
print('#'*30)
print("Zad 9")
print(df.loc[(((df['wl1'] > 3) & (df['se1'] >= 13)) | ((df['wl2'] > 3) & (df['se2'] >= 13)) | ((df['wl3'] > 3) & (df['se3'] >= 13)))])
############################################
# ZAD 10
print('#'*30)
print("Zad 10")
df['group_idx'] = df.groupby(['group']).ngroup()
print(df)
############################################
# ZAD 11
print('#'*30)
print("Zad 11")
df["se1_index"] = df.groupby(['se1']).ngroup()
print(df[['se1', 'se1_index']])
print(df.loc[((df['se1'] > 13) & (df['se1'] < 15))])
############################################
# ZAD 12
print('#'*30)
print("Zad 12")
print(df.sort_values(by=['se1','se2']).loc[((df['se1'] > 4) & (df['se2'] > 4))])
############################################
# ZAD 13
print('#'*30)
print("Zad 13")
df["Stable"] = ((df['se1'] == df['se2']) & (df['se2'] == df['se3']))
print(df[['se1', 'se2', 'se3', 'Stable']])
############################################
# ZAD 14
print('#'*30)
print("Zad 14")
print(df.loc[((df['se1'] > df['se3']))])
print(len(df.loc[((df['se1'] > df['se3']))]))
############################################
# ZAD 15
print('#'*30)
print("Zad 15")
print(df.groupby(['group_idx', 'group']).sum()['sum_of_wl'])
df.groupby(['group']).sum()['sum_of_wl'].plot(kind='bar' )
plt.show()
############################################
# ZAD 16
print('#'*30)
print("Zad 16")
df1 = df.drop(['se1', 'wl1'], axis=1)
print(df1)
############################################
# ZAD 17
print('#'*30)
print("Zad 17")
df2 = df[df['group'] != 'Control']
print(df2)
############################################
# ZAD 18
print('#'*30)
print("Zad 18")
df3 = df.drop_duplicates(subset=['wl1', 'wl2', 'wl3'], keep='first')
print(df3)
############################################
# ZAD 19
print('#'*30)
print("Zad 19")
df['cat_avg_se'] = np.where(df['se_mean'] > 14.5, 'high', (np.where(df['se_mean'] < 13, 'low', 'medium')))
print(df[['se_mean', 'cat_avg_se']])
print(df.groupby(['cat_avg_se', 'group']).agg({'rownames':'count', 'sum_of_wl':'mean'}).rename(columns={'rownames':'Row Count', 'sum_of_wl':'avg_wl'}))
############################################
# ZAD 20
print('#'*30)
print("Zad 20")
total_weight = df['sum_of_wl'].sum()
print(total_weight)
#let assume 1kg = 1g of 999 gold ~ 260zl and 1 donut = 4zl and 8 mld people alive
is_it_possible = ((total_weight * 260 / 4)/8000000000) > 1
print(is_it_possible)
#its not posiible


