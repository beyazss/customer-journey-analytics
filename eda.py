import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/online_retail_II.csv')

print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(df.head())

# Cleaning
df_clean = df.copy()

df_clean = df_clean.dropna(subset=['Customer ID'])
df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]

df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['Month'] = df_clean['InvoiceDate'].dt.to_period('M')

print(f"\nClean data: {df_clean.shape[0]:,} rows")
print(f"Total revenue: £{df_clean['TotalPrice'].sum():,.2f}")
print(f"Unique customers: {df_clean['Customer ID'].nunique():,}")
print(f"Date range: {df_clean['InvoiceDate'].min().date()} → {df_clean['InvoiceDate'].max().date()}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

monthly_revenue = df_clean.groupby('Month')['TotalPrice'].sum()
axes[0, 0].plot(monthly_revenue.index.astype(str), monthly_revenue.values, marker='o', color='#534AB7')
axes[0, 0].set_title('Monthly Revenue Trend')
axes[0, 0].tick_params(axis='x', rotation=45)

country_revenue = df_clean.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
axes[0, 1].barh(country_revenue.index[::-1], country_revenue.values[::-1], color='#1D9E75')
axes[0, 1].set_title('Top 10 Countries by Revenue')

order_sizes = df_clean.groupby('Invoice')['Quantity'].sum()
axes[1, 0].hist(order_sizes.clip(upper=200), bins=50, color='#D85A30', edgecolor='white')
axes[1, 0].set_title('Order Size Distribution')

customer_spend = df_clean.groupby('Customer ID')['TotalPrice'].sum()
axes[1, 1].hist(customer_spend.clip(upper=5000), bins=50, color='#378ADD', edgecolor='white')
axes[1, 1].set_title('Customer Spend Distribution')

plt.tight_layout()
plt.savefig('data/eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# Save clean data
df_clean.to_csv('data/clean_data.csv', index=False)