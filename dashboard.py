import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Customer Journey Analytics", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; max-width: 1400px; }
    .kpi-box {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #534AB7;
        border-radius: 6px;
        padding: 12px 16px;
    }
    .kpi-label { font-size: 10px; font-weight: 700; color: #94a3b8;
                 text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 4px; }
    .kpi-value { font-size: 20px; font-weight: 700; color: #1e293b; line-height: 1.2; }
    .kpi-sub   { font-size: 10px; color: #94a3b8; margin-top: 3px; }
    [data-testid="stSidebar"] { background: #0f172a !important; }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-size: 14px !important; }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #534AB7 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500;
                                    padding: 8px 20px; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background: #534AB7 !important; color: white !important; }
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

P = ["#534AB7","#1D9E75","#D85A30","#378ADD","#C97C1A","#D4537E"]

@st.cache_data
def load():
    rfm   = pd.read_csv('data/rfm_segments.csv')
    churn = pd.read_csv('data/churn_scores.csv')
    attr  = pd.read_csv('data/attribution_results.csv', index_col=0)
    df    = pd.read_csv('data/clean_data.csv', parse_dates=['InvoiceDate'])
    churn = churn.merge(rfm[['Customer ID','Segment']], on='Customer ID', how='left')
    return rfm, churn, attr, df

rfm, churn, attr, df = load()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Customer Journey\nAnalytics")
    st.markdown("---")
    st.markdown("**Filters**")

    all_seg = sorted(rfm['Segment'].unique())
    sel_seg = st.multiselect("Segment", all_seg, default=all_seg)

    all_ctr = sorted(df['Country'].value_counts().head(15).index)
    sel_ctr = st.multiselect("Country", all_ctr, default=all_ctr)

    min_d = df['InvoiceDate'].min().date()
    max_d = df['InvoiceDate'].max().date()
    dates = st.date_input("Date Range",
                          value=(min_d, max_d),
                          min_value=min_d,
                          max_value=max_d)

# ── APPLY FILTERS ─────────────────────────────────────────────────
dff = df.copy()
if sel_ctr:
    dff = dff[dff['Country'].isin(sel_ctr)]
if len(dates) == 2:
    dff = dff[(dff['InvoiceDate'].dt.date >= dates[0]) &
              (dff['InvoiceDate'].dt.date <= dates[1])]

rfm_f   = rfm[rfm['Segment'].isin(sel_seg)] if sel_seg else rfm
churn_f = churn[churn['Segment'].isin(sel_seg)] if sel_seg else churn

# ── HEADER ────────────────────────────────────────────────────────
st.markdown("## Customer Journey Analytics")
st.caption("RFM Segmentation · Multi-Touch Attribution · Churn Prediction — UCI Online Retail II")
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────
aov       = dff.groupby('Invoice')['TotalPrice'].sum().mean()
champ_shr = rfm[rfm['Segment']=='Champions']['Monetary'].sum() / rfm['Monetary'].sum()
churn_rate= churn_f['churned'].mean()

cols = st.columns(6)
kpis = [
    ("Total Revenue",       f"£{dff['TotalPrice'].sum():,.0f}",    f"{dff['Invoice'].nunique():,} orders"),
    ("Unique Customers",    f"{dff['Customer ID'].nunique():,}",     "in filtered view"),
    ("Avg Order Value",     f"£{aov:,.0f}",                         "per invoice"),
    ("Churn Rate",          f"{churn_rate:.1%}",                    "temporal split · AUC 0.76"),
    ("Champions Rev Share", f"{champ_shr:.1%}",                     "of total revenue"),
    ("Unique SKUs",         f"{dff['StockCode'].nunique():,}",       "distinct products sold"),
]
for col, (lbl, val, sub) in zip(cols, kpis):
    col.markdown(f'<div class="kpi-box"><div class="kpi-label">{lbl}</div>'
                 f'<div class="kpi-value">{val}</div>'
                 f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── TABS ──────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["📈 Overview", "👥 Segmentation", "📣 Attribution", "⚠️ Churn"])

def clean_fig(fig, h=300):
    fig.update_layout(height=h, margin=dict(t=40,b=10,l=10,r=10),
                      plot_bgcolor='white', paper_bgcolor='white',
                      font=dict(size=11))
    fig.update_xaxes(gridcolor='#f1f5f9', linecolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#f1f5f9', linecolor='#e2e8f0')
    return fig

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with t1:
    monthly = (dff.groupby(dff['InvoiceDate'].dt.to_period('M'))
               .agg(Revenue=('TotalPrice','sum'),
                    Customers=('Customer ID','nunique'))
               .reset_index())
    monthly['Period'] = monthly['InvoiceDate'].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly['Period'], y=monthly['Revenue'],
                         name='Revenue', marker_color='#534AB7', opacity=0.85),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly['Period'], y=monthly['Customers'],
                             name='Active Customers',
                             line=dict(color='#1D9E75', width=2),
                             mode='lines+markers', marker=dict(size=4)),
                  secondary_y=True)
    fig.update_layout(height=290, margin=dict(t=40,b=10,l=10,r=10),
                      plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(orientation='h', y=1.15),
                      title='Monthly Revenue & Active Customers')
    fig.update_yaxes(tickprefix='£', tickformat=',.0f',
                     title_text='Revenue (£)', gridcolor='#f1f5f9', secondary_y=False)
    fig.update_yaxes(title_text='Customers', secondary_y=True)
    fig.update_xaxes(tickangle=-30, gridcolor='#f1f5f9')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        cr = (dff.groupby('Country')['TotalPrice'].sum()
              .sort_values(ascending=False).head(10).reset_index())
        cr.columns = ['Country','Revenue']
        fig = px.bar(cr.sort_values('Revenue'), x='Revenue', y='Country',
                     orientation='h', title='Top 10 Countries by Revenue',
                     color='Revenue', color_continuous_scale='Purples')
        fig.update_xaxes(tickprefix='£', tickformat=',.0f', title='Revenue (£)')
        fig.update_yaxes(title='')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    with c2:
        dff2 = dff.copy()
        dff2['Hour'] = dff2['InvoiceDate'].dt.hour
        hr = dff2.groupby('Hour')['TotalPrice'].sum().reset_index()
        hr.columns = ['Hour of Day','Revenue']
        fig = px.bar(hr, x='Hour of Day', y='Revenue',
                     title='Revenue by Hour of Day',
                     color='Revenue', color_continuous_scale='Oranges')
        fig.update_yaxes(tickprefix='£', tickformat=',.0f', title='Revenue (£)')
        fig.update_xaxes(title='Hour of Day')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    with c3:
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dff3 = dff.copy()
        dff3['Day'] = dff3['InvoiceDate'].dt.day_name()
        dw = dff3.groupby('Day')['TotalPrice'].sum().reindex(dow_order).reset_index()
        dw.columns = ['Day','Revenue']
        fig = px.bar(dw, x='Day', y='Revenue', title='Revenue by Day of Week',
                     color='Revenue', color_continuous_scale='Greens')
        fig.update_yaxes(tickprefix='£', tickformat=',.0f', title='Revenue (£)')
        fig.update_xaxes(title='')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        tp = (dff.groupby('Description')['TotalPrice'].sum()
              .sort_values(ascending=False).head(10).reset_index())
        tp.columns = ['Product','Revenue']
        tp['Product'] = tp['Product'].str.title().str[:35]
        fig = px.bar(tp.sort_values('Revenue'), x='Revenue', y='Product',
                     orientation='h', title='Top 10 Products by Revenue',
                     color='Revenue', color_continuous_scale='Blues')
        fig.update_xaxes(tickprefix='£', tickformat=',.0f', title='Revenue (£)')
        fig.update_yaxes(title='')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    with c2:
        monthly['MoM'] = monthly['Revenue'].pct_change() * 100
        fig = go.Figure(go.Bar(
            x=monthly['Period'], y=monthly['MoM'],
            marker_color=['#1D9E75' if v >= 0 else '#D85A30'
                          for v in monthly['MoM'].fillna(0)]))
        fig.add_hline(y=0, line_color='#94a3b8', line_width=1)
        fig.update_layout(title='Month-over-Month Revenue Growth (%)')
        fig.update_yaxes(ticksuffix='%', title='Growth (%)')
        fig.update_xaxes(tickangle=-30, title='')
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════
with t2:
    c1, c2, c3 = st.columns([1.2, 1, 1])

    with c1:
        sa = (rfm_f.groupby('Segment')
              .agg(Count=('Customer ID','count'),
                   Revenue=('Monetary','sum'),
                   AvgRev=('Monetary','mean'))
              .reset_index().sort_values('Count', ascending=True))
        fig = px.bar(sa, x='Count', y='Segment', orientation='h',
                     color='AvgRev', color_continuous_scale='Purples',
                     title='Segment Size & Avg Revenue per Customer',
                     text='Count')
        fig.update_traces(textposition='outside')
        fig.update_xaxes(title='Number of Customers')
        fig.update_yaxes(title='')
        fig.update_layout(coloraxis_colorbar=dict(title='Avg £'))
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    with c2:
        sr = rfm_f.groupby('Segment')['Monetary'].sum().reset_index()
        sr.columns = ['Segment','Revenue']
        fig = px.pie(sr, names='Segment', values='Revenue',
                     title='Revenue Share by Segment',
                     color_discrete_sequence=P, hole=0.45)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    with c3:
        st_tbl = (rfm_f.groupby('Segment')
                  .agg(N=('Customer ID','count'),
                       Recency=('Recency','mean'),
                       Frequency=('Frequency','mean'),
                       Monetary=('Monetary','mean'))
                  .round(1).reset_index()
                  .sort_values('Monetary', ascending=False))
        fig = go.Figure(go.Table(
            columnwidth=[110,45,65,55,65],
            header=dict(values=['Segment','N','Recency','Freq','Avg £'],
                        fill_color='#534AB7',
                        font=dict(color='white', size=11),
                        align='left', height=26),
            cells=dict(
                values=[st_tbl['Segment'], st_tbl['N'],
                        st_tbl['Recency'], st_tbl['Frequency'],
                        st_tbl['Monetary'].apply(lambda x: f'£{x:,.0f}')],
                fill_color=[['#f8fafc','#fff']*10],
                align='left', font=dict(size=11), height=24)
        ))
        fig.update_layout(title='RFM Summary by Segment')
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(rfm_f, x='Recency', y='Monetary', color='Segment',
                         size='Frequency',
                         hover_data={'Customer ID': True,
                                     'Recency': True,
                                     'Frequency': True,
                                     'Monetary': ':.0f'},
                         color_discrete_sequence=P, opacity=0.6,
                         title='RFM Scatter — Recency vs Monetary (size = Frequency)')
        fig.update_yaxes(tickprefix='£', tickformat=',.0f', title='Monetary Value (£)')
        fig.update_xaxes(title='Recency (days since last purchase)')
        fig.update_layout(legend=dict(orientation='h', y=1.12))
        st.plotly_chart(clean_fig(fig, h=360), use_container_width=True)

    with c2:
        fig = px.box(rfm_f, x='Segment', y='Monetary', color='Segment',
                     color_discrete_sequence=P,
                     title='Monetary Distribution by Segment')
        fig.update_yaxes(tickprefix='£', tickformat=',.0f', title='Monetary Value (£)')
        fig.update_xaxes(title='', tickangle=-20)
        fig.update_layout(showlegend=False)
        st.plotly_chart(clean_fig(fig, h=360), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ATTRIBUTION
# ══════════════════════════════════════════════════════════════════
with t3:
    c1, c2 = st.columns([1.5, 1])

    with c1:
        ap = attr.reset_index()
        ap.columns = ['Channel','Last-click','Linear','Time Decay']
        am = ap.melt(id_vars='Channel', var_name='Model', value_name='Revenue')
        fig = px.bar(am, x='Channel', y='Revenue', color='Model',
                     barmode='group',
                     title='Attributed Revenue by Model & Channel',
                     color_discrete_sequence=['#534AB7','#1D9E75','#D85A30'])
        fig.update_yaxes(tickprefix='£', tickformat=',.0f', title='Revenue (£)')
        fig.update_xaxes(title='')
        fig.update_layout(legend=dict(orientation='h', y=1.12))
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    with c2:
        diff = ((attr['linear'] - attr['last_click']) /
                attr['last_click'] * 100).round(1)
        fig = go.Figure(go.Table(
            columnwidth=[100,80,80,85,70],
            header=dict(values=['Channel','Last-click','Linear','Time Decay','Δ% (LC→Lin)'],
                        fill_color='#534AB7',
                        font=dict(color='white', size=11),
                        align='left', height=26),
            cells=dict(values=[
                attr.index,
                attr['last_click'].apply(lambda x: f'£{x:,.0f}'),
                attr['linear'].apply(lambda x: f'£{x:,.0f}'),
                attr['time_decay'].apply(lambda x: f'£{x:,.0f}'),
                diff.apply(lambda x: f'+{x}%' if x > 0 else f'{x}%')
            ], fill_color=[['#f8fafc','#fff']*10],
               align='left', font=dict(size=11), height=24)
        ))
        fig.update_layout(title='Model Comparison Table')
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        rd = pd.DataFrame({'Channel': attr.index,
                           'Share': attr['linear'] / attr['linear'].sum()})
        fig = px.line_polar(rd, r='Share', theta='Channel', line_close=True,
                            title='Linear Attribution — Channel Share (Radar)',
                            color_discrete_sequence=['#534AB7'])
        fig.update_traces(fill='toself', opacity=0.55)
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    with c2:
        mt = attr[['last_click','linear','time_decay']].sum().reset_index()
        mt.columns = ['Model','Revenue']
        mt['Model'] = ['Last-click','Linear','Time Decay']
        fig = px.bar(mt, x='Model', y='Revenue',
                     title='Total Attributed Revenue by Model',
                     color='Model',
                     color_discrete_sequence=['#534AB7','#1D9E75','#D85A30'])
        fig.update_yaxes(tickprefix='£', tickformat=',.0f', title='Revenue (£)')
        fig.update_xaxes(title='')
        fig.update_layout(showlegend=False)
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — CHURN
# ══════════════════════════════════════════════════════════════════
with t4:
    churn_f = churn_f.copy()
    churn_f['Risk Tier'] = pd.cut(
        churn_f['churn_prob'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low (<40%)', 'Medium (40–70%)', 'High (>70%)'])

    risk_colors = {'Low (<40%)':'#1D9E75',
                   'Medium (40–70%)':'#C97C1A',
                   'High (>70%)':'#D85A30'}

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure()
        for label, color, mask in [('Active','#1D9E75', churn_f['churned']==0),
                                    ('Churned','#D85A30', churn_f['churned']==1)]:
            fig.add_trace(go.Histogram(x=churn_f[mask]['churn_prob'],
                                       name=label, opacity=0.75,
                                       marker_color=color, nbinsx=30))
        fig.update_layout(barmode='overlay', title='Churn Score Distribution',
                          legend=dict(orientation='h', y=1.12))
        fig.update_xaxes(title='Churn Probability')
        fig.update_yaxes(title='Number of Customers')
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    with c2:
        tc = churn_f['Risk Tier'].value_counts().reset_index()
        tc.columns = ['Risk Tier','Count']
        fig = px.pie(tc, names='Risk Tier', values='Count', hole=0.5,
                     title='Customer Risk Tier Distribution',
                     color='Risk Tier', color_discrete_map=risk_colors)
        fig.update_layout(legend=dict(orientation='h', y=-0.1))
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    with c3:
        sc = (churn_f.groupby('Segment')['churn_prob']
              .mean().sort_values().reset_index())
        sc.columns = ['Segment','Avg Churn Prob']
        colors = ['#1D9E75' if v < 0.4 else '#D85A30' if v > 0.6 else '#C97C1A'
                  for v in sc['Avg Churn Prob']]
        fig = px.bar(sc, x='Avg Churn Prob', y='Segment', orientation='h',
                     title='Avg Churn Risk by Segment')
        fig.update_traces(marker_color=colors)
        fig.add_vline(x=0.5, line_dash='dash', line_color='#94a3b8')
        fig.update_xaxes(title='Avg Churn Probability', tickformat='.0%')
        fig.update_yaxes(title='')
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(churn_f, x='Frequency', y='churn_prob',
                         color='Segment', opacity=0.5,
                         color_discrete_sequence=P,
                         title='Churn Probability vs Purchase Frequency',
                         hover_data=['Customer ID'])
        fig.add_hline(y=0.5, line_dash='dash', line_color='#94a3b8')
        fig.update_xaxes(title='Purchase Frequency (# orders)')
        fig.update_yaxes(title='Churn Probability', tickformat='.0%')
        fig.update_layout(legend=dict(orientation='h', y=1.12))
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

    with c2:
        fig = px.histogram(churn_f, x='Monetary', color='Risk Tier',
                           nbins=40, barmode='overlay', opacity=0.75,
                           title='Revenue Exposure by Risk Tier',
                           color_discrete_map=risk_colors)
        fig.update_xaxes(tickprefix='£', tickformat=',.0f',
                         title='Customer Lifetime Revenue (£)')
        fig.update_yaxes(title='Number of Customers')
        fig.update_layout(legend=dict(orientation='h', y=1.12))
        st.plotly_chart(clean_fig(fig, h=320), use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#94a3b8;font-size:11px;'>"
            "Customer Journey Analytics &nbsp;·&nbsp; UCI Online Retail II &nbsp;·&nbsp;"
            " Python · scikit-learn · Plotly · Streamlit</p>",
            unsafe_allow_html=True)




