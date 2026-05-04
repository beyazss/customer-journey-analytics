import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load clean data
df = pd.read_csv('data/clean_data.csv', parse_dates=['InvoiceDate'])

# The dataset does not include channel/traffic source data.
# Channel touchpoints are simulated to demonstrate attribution logic.
# In a real setting, this data would come from GA4 or a CRM platform.
np.random.seed(42)

channels = ['Organic Search', 'Email', 'Paid Search', 'Direct', 'Social']
channel_weights = [0.35, 0.25, 0.20, 0.15, 0.05]

df['channel'] = np.random.choice(channels, size=len(df), p=channel_weights)

# Her müşterinin touchpoint journey'si
customer_journeys = df.groupby('Customer ID').agg(
    touchpoints=('channel', list),
    total_revenue=('TotalPrice', 'sum'),
    num_orders=('Invoice', 'nunique')
).reset_index()

# Last-click attribution
customer_journeys['last_click'] = customer_journeys['touchpoints'].apply(lambda x: x[-1])

last_click = customer_journeys.groupby('last_click')['total_revenue'].sum().sort_values(ascending=False)

# Linear attribution - her touchpoint'e eşit pay
def linear_attribution(journeys_df):
    attr = {ch: 0 for ch in channels}
    for _, row in journeys_df.iterrows():
        touches = row['touchpoints']
        revenue = row['total_revenue']
        share = revenue / len(touches)
        for t in touches:
            attr[t] += share
    return pd.Series(attr)

linear = linear_attribution(customer_journeys).sort_values(ascending=False)

# Time decay attribution - son touchpoint'e daha fazla ağırlık
def time_decay_attribution(journeys_df, decay=0.5):
    attr = {ch: 0 for ch in channels}
    for _, row in journeys_df.iterrows():
        touches = row['touchpoints']
        revenue = row['total_revenue']
        n = len(touches)
        weights = np.array([decay ** (n - i - 1) for i in range(n)])
        weights = weights / weights.sum()
        for t, w in zip(touches, weights):
            attr[t] += revenue * w
    return pd.Series(attr)

time_decay = time_decay_attribution(customer_journeys).sort_values(ascending=False)

print("Last-click attribution:")
print(last_click.apply(lambda x: f'£{x:,.0f}'))
print("\nLinear attribution:")
print(linear.apply(lambda x: f'£{x:,.0f}'))
print("\nTime decay attribution:")
print(time_decay.apply(lambda x: f'£{x:,.0f}'))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#534AB7', '#1D9E75', '#D85A30', '#378ADD', '#BA7517']

for ax, data, title in zip(axes,
                            [last_click, linear, time_decay],
                            ['Last-click', 'Linear', 'Time Decay']):
    sorted_data = data.sort_values(ascending=True)
    ax.barh(sorted_data.index, sorted_data.values, color=colors)
    ax.set_title(f'{title} Attribution')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('data/attribution_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# Save
attribution_df = pd.DataFrame({
    'last_click': last_click,
    'linear': linear,
    'time_decay': time_decay
})
attribution_df.to_csv('data/attribution_results.csv')
print("\nSaved: data/attribution_results.csv")