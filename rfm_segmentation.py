import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load clean data
df = pd.read_csv('data/clean_data.csv', parse_dates=['InvoiceDate'])

# RFM calculation
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('Customer ID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('Invoice', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

print(rfm.describe())

# Scoring (1-5, higher is better)
rfm['R_score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm['M_score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])

rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

# Segmentation
def segment(row):
    r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Lost'
    else:
        return 'Potential'

rfm['Segment'] = rfm.apply(segment, axis=1)

print("\nSegment distribution:")
print(rfm['Segment'].value_counts())
print(f"\nRevenue by segment:")
print(rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Segment counts
seg_counts = rfm['Segment'].value_counts()
colors = ['#534AB7', '#1D9E75', '#D85A30', '#378ADD', '#BA7517', '#D4537E']
axes[0].bar(seg_counts.index, seg_counts.values, color=colors)
axes[0].set_title('Customer Segments')
axes[0].tick_params(axis='x', rotation=30)

# Revenue by segment
seg_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
axes[1].bar(seg_revenue.index, seg_revenue.values, color=colors)
axes[1].set_title('Revenue by Segment')
axes[1].tick_params(axis='x', rotation=30)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x:,.0f}'))

# RFM scatter - Recency vs Monetary
segment_colors = {
    'Champions': '#534AB7', 'Loyal': '#1D9E75', 'New Customers': '#378ADD',
    'At Risk': '#D85A30', 'Lost': '#BA7517', 'Potential': '#D4537E'
}
for seg, group in rfm.groupby('Segment'):
    axes[2].scatter(group['Recency'], group['Monetary'].clip(upper=10000),
                    label=seg, alpha=0.5, s=20, color=segment_colors[seg])
axes[2].set_title('Recency vs Monetary')
axes[2].set_xlabel('Recency (days)')
axes[2].set_ylabel('Monetary (£)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('data/rfm_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# Save
rfm.to_csv('data/rfm_segments.csv', index=False)
print("\nSaved: data/rfm_segments.csv")