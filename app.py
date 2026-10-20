import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="FinSight | Financial Risk Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 FinSight")
st.subheader("Financial Risk & Investment Analytics")
st.caption(
    "Historical market analysis using real price data "
    "processed with PySpark."
)

@st.cache_data
def load_data():
    daily = pd.read_csv("finsight_daily_risk_data.csv")
    summary = pd.read_csv("finsight_risk_summary.csv")

    daily["date"] = pd.to_datetime(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["drawdown"] = pd.to_numeric(
        daily["drawdown"], errors="coerce"
    )
    daily["daily_return"] = pd.to_numeric(
        daily["daily_return"], errors="coerce"
    )

    daily = daily.dropna(subset=["date", "ticker", "close"])
    return daily, summary

try:
    daily, summary = load_data()
except Exception as e:
    st.error(
        "Could not load the dashboard data. "
        "Ensure both CSV files are in the same folder as app.py."
    )
    st.exception(e)
    st.stop()

assets = sorted(daily["ticker"].unique())

st.sidebar.header("Dashboard Filters")
selected_assets = st.sidebar.multiselect(
    "Select stocks or indices",
    options=assets,
    default=assets[:min(3, len(assets))]
)

if not selected_assets:
    st.warning("Select at least one asset from the sidebar.")
    st.stop()

filtered = daily[daily["ticker"].isin(selected_assets)].copy()
filtered_summary = summary[
    summary["ticker"].isin(selected_assets)
].copy()

min_date = filtered["date"].min().date()
max_date = filtered["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        filtered["date"].dt.date.between(start_date, end_date)
    ]

if filtered.empty:
    st.warning("No data available for this date range.")
    st.stop()

# Calculate metrics for the selected date range
metrics = []

for ticker in selected_assets:
    asset = filtered[filtered["ticker"] == ticker].sort_values("date")
    if asset.empty:
        continue

    first_price = asset["close"].iloc[0]
    latest_price = asset["close"].iloc[-1]
    period_return = latest_price / first_price - 1

    daily_returns = asset["daily_return"].dropna()
    volatility = (
        daily_returns.std() * (252 ** 0.5)
        if len(daily_returns) > 1 else float("nan")
    )

    running_peak = asset["close"].cummax()
    drawdowns = asset["close"] / running_peak - 1
    max_drawdown = drawdowns.min()

    metrics.append({
        "ticker": ticker,
        "latest_close": latest_price,
        "period_return": period_return,
        "annualized_volatility": volatility,
        "max_drawdown": max_drawdown
    })

metrics_df = pd.DataFrame(metrics)

st.markdown("### Key Performance Indicators")

if not metrics_df.empty:
    best = metrics_df.loc[metrics_df["period_return"].idxmax()]
    worst = metrics_df.loc[metrics_df["period_return"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets selected", len(metrics_df))
    c2.metric(
        "Best period return",
        f"{best['ticker']}",
        f"{best['period_return']:.2%}"
    )
    c3.metric(
        "Lowest period return",
        f"{worst['ticker']}",
        f"{worst['period_return']:.2%}"
    )
    c4.metric(
        "Average annualized volatility",
        f"{metrics_df['annualized_volatility'].mean():.2%}"
    )

st.markdown("---")
st.markdown("### Historical Closing Prices")

price_fig = px.line(
    filtered,
    x="date",
    y="close",
    color="ticker",
    title="Closing price over time",
    labels={
        "date": "Date",
        "close": "Adjusted closing price",
        "ticker": "Asset"
    }
)
price_fig.update_layout(hovermode="x unified")
st.plotly_chart(price_fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.markdown("### Period Return Comparison")
    return_fig = px.bar(
        metrics_df,
        x="ticker",
        y="period_return",
        color="period_return",
        title="Price return over selected period",
        labels={"ticker": "Asset", "period_return": "Return"}
    )
    return_fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(return_fig, use_container_width=True)

with right:
    st.markdown("### Risk Comparison")
    risk_fig = px.bar(
        metrics_df,
        x="ticker",
        y="annualized_volatility",
        color="ticker",
        title="Annualized volatility",
        labels={
            "ticker": "Asset",
            "annualized_volatility": "Volatility"
        }
    )
    risk_fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(risk_fig, use_container_width=True)

st.markdown("### Drawdown Analysis")

drawdown_fig = px.line(
    filtered,
    x="date",
    y="drawdown",
    color="ticker",
    title="Drawdown from each asset's running peak",
    labels={
        "date": "Date",
        "drawdown": "Drawdown",
        "ticker": "Asset"
    }
)
drawdown_fig.update_yaxes(tickformat=".1%")
st.plotly_chart(drawdown_fig, use_container_width=True)

st.markdown("### Risk Metrics Table")

display_metrics = metrics_df.copy()
for col in [
    "period_return",
    "annualized_volatility",
    "max_drawdown"
]:
    display_metrics[col] = display_metrics[col].map(
        lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
    )

display_metrics = display_metrics.rename(columns={
    "ticker": "Asset",
    "latest_close": "Latest Close",
    "period_return": "Period Return",
    "annualized_volatility": "Annualized Volatility",
    "max_drawdown": "Maximum Drawdown"
})

st.dataframe(display_metrics, use_container_width=True)

st.download_button(
    "Download filtered market data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="finsight_filtered_market_data.csv",
    mime="text/csv"
)

st.caption(
    "Data is historical and may be delayed or revised by its provider. "
    "This dashboard is for educational analysis, not investment advice. "
    "Past performance does not guarantee future results."
)
