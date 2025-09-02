import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


df = pd.read_csv('Salinan Salinan Online Retail Data.csv', header =0)

print(df)

# Data cleansing
df_clean = df.copy()
df_clean['date'] = pd.to_datetime(df_clean['order_date']).dt.date.astype('datetime64[ns]')

# menghapus semua baris tanpa customer_id
df_clean = df_clean[~df_clean['customer_id'].isna()]
# menghapus semua baris tanpa product_name
df_clean = df_clean[~df_clean['product_name'].isna()]
# membuat semua product_name berhuruf kecil
df_clean['product_name'] = df_clean['product_name'].str.lower()
# menghapus semua baris dengan product_code atau product_name test
df_clean = df_clean[(~df_clean['product_code'].str.lower().str.contains('test')) |
                    (~df_clean['product_name'].str.contains('test '))]
# membuat kolom order_status dengan nilai 'cancelled' jika order_id diawali dengan huruf 'c' dan 'delivered' jika order_id tanpa awalan huruf 'c'
df_clean['order_status'] = np.where(df_clean['order_id'].str[:1]=='C', 'cancelled', 'delivered')
# mengubah nilai quantity yang negatif menjadi positif karena nilai negatif tersebut hanya menandakan order tersebut cancelled
df_clean['quantity'] = df_clean['quantity'].abs()
# menghapus baris dengan price bernilai negatif
df_clean = df_clean[df_clean['price']>0]
# membuat nilai amount, yaitu perkalian antara quantity dan price
df_clean['amount'] = df_clean['quantity'] * df_clean['price']
# mengganti product_name dari product_code yang memiliki beberapa product_name dengan salah satu product_name-nya yang paling sering muncul
most_freq_product_name = df_clean.groupby(['product_code','product_name'], as_index=False).agg(order_cnt=('order_id','nunique')).sort_values(['product_code','order_cnt'], ascending=[True,False])
most_freq_product_name['rank'] = most_freq_product_name.groupby('product_code')['order_cnt'].rank(method='first', ascending=False)
most_freq_product_name = most_freq_product_name[most_freq_product_name['rank']==1].drop(columns=['order_cnt','rank'])
df_clean = df_clean.merge(most_freq_product_name.rename(columns={'product_name':'most_freq_product_name'}), how='left', on='product_code')
df_clean['product_name'] = df_clean['most_freq_product_name']
df_clean = df_clean.drop(columns='most_freq_product_name')
# mengkonversi customer_id menjadi string
df_clean['customer_id'] = df_clean['customer_id'].astype(str)
# menghapus outlier
from scipy import stats
df_clean = df_clean[(np.abs(stats.zscore(df_clean[['quantity','amount']]))<3).all(axis=1)]
df_clean = df_clean.reset_index(drop=True)
df_clean

# Agregat data transaksi ke bentuk summary total transaksi (order), total nilai order (order value)
# tanggal order terakhir dari setiap penggunanya

df_user = df_clean.groupby('customer_id', as_index=False).agg(order_count=('order_id','nunique'),
                                                              max_order_date=('date','max'),
                                                              total_order_value=('amount','sum'))

today = df_clean['date'].max()
df_user['day_since_last_order'] = (today-df_user['max_order_date']).dt.days

#Binning
df_user['recency_score'] = pd.cut(df_user['day_since_last_order'],bins=[df_user['day_since_last_order'].min(),
                                                                        np.percentile(df_user['day_since_last_order'],20),
                                                                        np.percentile(df_user['day_since_last_order'],40),
                                                                        np.percentile(df_user['day_since_last_order'],60),
                                                                        np.percentile(df_user['day_since_last_order'],80),
                                                                        df_user['day_since_last_order'].max()], 
                                  labels=[5,4,3,2,1], include_lowest = True).astype(int)

df_user['frequency_score'] = pd.cut(df_user['order_count'],bins=[0,
                                                                        np.percentile(df_user['order_count'],20),
                                                                        np.percentile(df_user['order_count'],40),
                                                                        np.percentile(df_user['order_count'],60),
                                                                        np.percentile(df_user['order_count'],80),
                                                                        df_user['order_count'].max()], 
                                  labels=[1,2,3,4,5], include_lowest = True).astype(int)

df_user['monetary_score'] = pd.cut(df_user['total_order_value'],bins=[df_user['total_order_value'].min(),
                                                                        np.percentile(df_user['total_order_value'],20),
                                                                        np.percentile(df_user['total_order_value'],40),
                                                                        np.percentile(df_user['total_order_value'],60),
                                                                        np.percentile(df_user['total_order_value'],80),
                                                                        df_user['total_order_value'].max()], 
                                  labels=[1,2,3,4,5], include_lowest = True).astype(int)

df_user['segment'] = np.select(
[(df_user['recency_score']==5) & (df_user ['frequency_score']>=4),
 (df_user['recency_score'].between(3, 4)) & (df_user ['frequency_score']>=4),
 (df_user['recency_score']>=4) & (df_user ['frequency_score'].between(2, 3)),
 (df_user['recency_score']<=2) & (df_user ['frequency_score']==5),
 (df_user['recency_score']==3) & (df_user ['frequency_score']==3),
 (df_user['recency_score']==5) & (df_user ['frequency_score']==1),
 (df_user['recency_score']==4) & (df_user ['frequency_score']==1),
 (df_user['recency_score']<=2) & (df_user ['frequency_score'].between(3, 4)),
 (df_user['recency_score']==3) & (df_user ['frequency_score']<=2),
 (df_user['recency_score']<=2) & (df_user ['frequency_score']<=2)],
['01-Champion', '02-Loyal Customer', '03-Potential Loyalists', '04-Cant Lose Them',
 '05-Need Attention', '06-New Customers', '07-Promising', '08-At Risk', '09-About to Sleep',
 '10-Hibernating'],default='Others')

# Summary dari RFM Segmentation
summary = pd.pivot_table(df_user,index='segment',values=['customer_id','day_since_last_order','order_count','total_order_value'],
                    aggfunc={
                        'customer_id': pd.Series.nunique,
                        'day_since_last_order': ['mean','median'],
                        'order_count': ['mean','median'],
                        'total_order_value': ['mean','median']
                    })

summary['pct_unique'] = (summary['customer_id'] / summary['customer_id'].sum() * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14,6))

sns.barplot(x=summary.index, y=summary['order_count']['mean'], ax=axes[0], palette="Blues")
axes[0].set_title("Rata-rata Order Count per Segmen")
axes[0].set_ylabel("Order Count")
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x=summary.index, y=summary['total_order_value']['mean'], ax=axes[1], palette="Greens")
axes[1].set_title("Rata-rata Order Value per Segmen")
axes[1].set_ylabel("Total Order Value")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
print(summary)
