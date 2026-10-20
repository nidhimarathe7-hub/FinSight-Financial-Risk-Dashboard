
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="FinSight | Financial Risk Dashboard",
    page_icon="📈",
    layout="wide"
)

# -------------------- HEADER --------------------

st.title("📈 FinSight")
st.subheader("Financial Risk & Investment Analytics")
st.caption(
    "Advanced portfolio intelligence using real historical market data "
    "processed with PySpark."
)

# -------------------- LOAD DATA --------------------

@st.cache_data
def load_data():
    daily = pd.read_csv("finsight_daily_risk_data.csv")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")

    daily = daily.dropna(subset=["date", "ticker", "close"])
    daily = daily[daily["close"] > 0]
    daily = daily.sort_values(["ticker", "date"])
    daily = daily.drop_duplicates(["ticker", "date"])

    # Recalculate returns to ensure consistency
    daily["daily_return"] = (
        daily.groupby("ticker")["close"].pct_change()
    )

    return daily

try:
    daily = load_data()
except Exception as e:
    st.error(
        "Could not load finsight_daily_risk_data.csv. "
        "Ensure the CSV file is in the same GitHub folder as app.py."
    )
    st.exception(e)
    st.stop()

# -------------------- SIDEBAR FILTERS --------------------

assets = sorted(daily["ticker"].unique())

st.sidebar.header("Dashboard Controls")

selected_assets = st.sidebar.multiselect(
    "Select stocks or indices",
    options=assets,
    default=assets[:min(6, len(assets))]
)

if not selected_assets:
    st.warning("Select at least one asset from the sidebar.")
    st.stop()

filtered = daily[
    daily["ticker"].isin(selected_assets)
].copy()

min_date = filtered["date"].min().date()
max_date = filtered["date"].max().date()

date_range = st.sidebar.date_input(
    "Analysis date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("Start date must be before the end date.")
    st.stop()

filtered = filtered[
    filtered["date"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No data is available for this date range.")
    st.stop()

filtered = filtered.sort_values(["ticker", "date"])

# -------------------- PREPARE PRICE DATA --------------------

prices = filtered.pivot_table(
    index="date",
    columns="ticker",
    values="close",
    aggfunc="last"
).sort_index()

# Returns are calculated from consecutive available price observations.
returns = prices.pct_change(fill_method=None)

# -------------------- ASSET METRICS --------------------

metric_rows = []

for ticker in prices.columns:
    series = prices[ticker].dropna()
    asset_returns = series.pct_change().dropna()

    if len(series) < 2:
        continue

    total_return = series.iloc[-1] / series.iloc[0] - 1

    volatility = (
        asset_returns.std(ddof=1) * np.sqrt(252)
        if len(asset_returns) >= 2 else np.nan
    )

    drawdown = (series / series.cummax()) - 1

    metric_rows.append({
        "ticker": ticker,
        "latest_close": series.iloc[-1],
        "period_return": total_return,
        "annualized_volatility": volatility,
        "max_drawdown": drawdown.min()
    })

metrics_df = pd.DataFrame(metric_rows)

if metrics_df.empty:
    st.warning("Not enough price observations to calculate metrics.")
    st.stop()

# -------------------- NAVIGATION --------------------

overview_tab, portfolio_tab, correlation_tab, risk_tab = st.tabs([
    "📊 Market Overview",
    "💼 Portfolio Builder",
    "🔗 Correlation Analysis",
    "🛡️ Risk Intelligence"
])

# ======================================================
# TAB 1: MARKET OVERVIEW
# ======================================================

with overview_tab:

    st.markdown("### Key Performance Indicators")

    best = metrics_df.loc[metrics_df["period_return"].idxmax()]
    worst = metrics_df.loc[metrics_df["period_return"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Assets Analysed", len(metrics_df))

    c2.metric(
        "Best Period Return",
        best["ticker"],
        f'{best["period_return"]:.2%}'
    )

    c3.metric(
        "Lowest Period Return",
        worst["ticker"],
        f'{worst["period_return"]:.2%}'
    )

    average_vol = metrics_df["annualized_volatility"].mean()

    c4.metric(
        "Average Annualized Volatility",
        f"{average_vol:.2%}" if pd.notna(average_vol) else "N/A"
    )

    st.markdown("---")
    st.markdown("### Historical Closing Prices")

    price_fig = px.line(
        prices.reset_index(),
        x="date",
        y=prices.columns.tolist(),
        title="Historical Closing Price Trends",
        labels={
            "date": "Date",
            "value": "Adjusted Closing Price",
            "variable": "Asset"
        }
    )

    price_fig.update_layout(hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### Return Comparison")

        return_fig = px.bar(
            metrics_df,
            x="ticker",
            y="period_return",
            color="period_return",
            color_continuous_scale="RdYlGn",
            title="Historical Period Return"
        )

        return_fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(return_fig, use_container_width=True)

    with right:
        st.markdown("### Volatility Comparison")

        volatility_fig = px.bar(
            metrics_df,
            x="ticker",
            y="annualized_volatility",
            color="ticker",
            title="Annualized Volatility"
        )

        volatility_fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(volatility_fig, use_container_width=True)

    st.markdown("### Drawdown Analysis")

    drawdown_frames = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        drawdown = series / series.cummax() - 1

        drawdown_frames.append(pd.DataFrame({
            "date": series.index,
            "ticker": ticker,
            "drawdown": drawdown.values
        }))

    drawdown_data = pd.concat(drawdown_frames, ignore_index=True)

    drawdown_fig = px.line(
        drawdown_data,
        x="date",
        y="drawdown",
        color="ticker",
        title="Decline from Previous Price Peak"
    )

    drawdown_fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(drawdown_fig, use_container_width=True)

    st.markdown("### Asset Performance Table")

    display_metrics = metrics_df.rename(columns={
        "ticker": "Asset",
        "latest_close": "Latest Close",
        "period_return": "Period Return",
        "annualized_volatility": "Annualized Volatility",
        "max_drawdown": "Maximum Drawdown"
    })

    st.dataframe(
        display_metrics.style.format({
            "Latest Close": "{:,.2f}",
            "Period Return": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Maximum Drawdown": "{:.2%}"
        }),
        use_container_width=True
    )

# ======================================================
# TAB 2: PORTFOLIO BUILDER
# ======================================================

with portfolio_tab:

    st.markdown("### Build Your Hypothetical Portfolio")

    st.write(
        "Choose an investment amount and allocate it across the "
        "selected assets. Portfolio weights must total 100%."
    )

    investment = st.number_input(
        "Investment amount (₹)",
        min_value=1000.0,
        value=100000.0,
        step=5000.0
    )

    st.markdown("#### Asset Allocation")

    portfolio_assets = list(prices.columns)

    # Default equal allocation
    default_weight = 100.0 / len(portfolio_assets)

    weights_input = {}

    cols = st.columns(min(3, len(portfolio_assets)))

    for i, ticker in enumerate(portfolio_assets):
        with cols[i % len(cols)]:
            weights_input[ticker] = st.number_input(
                f"{ticker} weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=round(default_weight, 2),
                step=5.0,
                key=f"portfolio_weight_{ticker}"
            )

    weights = pd.Series(weights_input, dtype=float) / 100
    total_weight = weights.sum()

    st.metric("Total Portfolio Allocation", f"{total_weight:.2%}")

    if not np.isclose(total_weight, 1.0, atol=0.001):
        st.warning("Adjust the weights so that the total equals 100%.")

    else:
        allocation = pd.DataFrame({
            "Asset": weights.index,
            "Weight": weights.values,
            "Estimated Investment (₹)": weights.values * investment
        })

        left, right = st.columns(2)

        with left:
            allocation_fig = px.pie(
                allocation,
                names="Asset",
                values="Weight",
                hole=0.55,
                title="Portfolio Allocation"
            )

            st.plotly_chart(
                allocation_fig,
                use_container_width=True
            )

        with right:
            st.dataframe(
                allocation.style.format({
                    "Weight": "{:.2%}",
                    "Estimated Investment (₹)": "₹{:,.2f}"
                }),
                use_container_width=True
            )

        # Daily rebalanced weighted return approximation.
        # Exclude the first row and any rows with missing asset returns.
        portfolio_returns = returns[portfolio_assets].dropna(how="any")

        if len(portfolio_returns) < 2:
            st.info(
                "More overlapping daily return observations are needed "
                "to calculate portfolio performance."
            )

        else:
            portfolio_daily_return = portfolio_returns.mul(
                weights, axis=1
            ).sum(axis=1)

            growth = (1 + portfolio_daily_return).cumprod()

            portfolio_total_return = growth.iloc[-1] - 1

            portfolio_volatility = (
                portfolio_daily_return.std(ddof=1) * np.sqrt(252)
            )

            portfolio_drawdown = (
                growth / growth.cummax() - 1
            ).min()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Historical Portfolio Return",
                f"{portfolio_total_return:.2%}"
            )

            c2.metric(
                "Annualized Volatility",
                f"{portfolio_volatility:.2%}"
            )

            c3.metric(
                "Maximum Drawdown",
                f"{portfolio_drawdown:.2%}"
            )

            portfolio_fig = px.line(
                x=growth.index,
                y=growth.values,
                title="Hypothetical Portfolio Growth",
                labels={
                    "x": "Date",
                    "y": "Growth of ₹1"
                }
            )

            st.plotly_chart(
                portfolio_fig,
                use_container_width=True
            )

            st.caption(
                "This model assumes daily rebalancing to the chosen weights. "
                "It is not a buy-and-hold simulation and excludes costs, "
                "taxes, slippage and cash flows."
            )

# ======================================================
# TAB 3: CORRELATION ANALYSIS
# ======================================================

with correlation_tab:

    st.markdown("### Asset Return Correlation")

    st.write(
        "Correlation measures how asset returns have moved together "
        "historically. It does not establish causation or guarantee "
        "future diversification."
    )

    if len(prices.columns) < 2:
        st.info("Select at least two assets to view the correlation heatmap.")

    else:
        correlation = returns.corr(min_periods=30)

        correlation_fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Daily Return Correlation Heatmap"
        )

        st.plotly_chart(
            correlation_fig,
            use_container_width=True
        )

        pair_rows = []

        corr_columns = correlation.columns.tolist()

        for i in range(len(corr_columns)):
            for j in range(i + 1, len(corr_columns)):
                value = correlation.iloc[i, j]

                if pd.notna(value):
                    pair_rows.append({
                        "Asset A": corr_columns[i],
                        "Asset B": corr_columns[j],
                        "Correlation": value
                    })

        if pair_rows:
            pairs = pd.DataFrame(pair_rows)

            st.markdown("### Asset Pair Comparison")

            most_positive = pairs.loc[pairs["Correlation"].idxmax()]
            most_negative = pairs.loc[pairs["Correlation"].idxmin()]

            c1, c2 = st.columns(2)

            c1.metric(
                "Highest Pair Correlation",
                f'{most_positive["Asset A"]} / {most_positive["Asset B"]}',
                f'{most_positive["Correlation"]:.2f}'
            )

            c2.metric(
                "Lowest Pair Correlation",
                f'{most_negative["Asset A"]} / {most_negative["Asset B"]}',
                f'{most_negative["Correlation"]:.2f}'
            )

            st.dataframe(
                pairs.sort_values(
                    "Correlation", ascending=False
                ),
                use_container_width=True
            )

        else:
            st.info(
                "At least 30 overlapping daily return observations "
                "are needed for pairwise correlations."
            )

# ======================================================
# TAB 4: RISK INTELLIGENCE
# ======================================================

with risk_tab:

    st.markdown("### Sharpe Ratio & Risk Intelligence")

    risk_free_rate = st.number_input(
        "Assumed annual risk-free rate (%)",
        min_value=0.0,
        max_value=30.0,
        value=6.0,
        step=0.25
    ) / 100

    st.caption(
        "The risk-free rate is an adjustable assumption, not a live market quote."
    )

    risk_rows = []

    for ticker in prices.columns:

        series = prices[ticker].dropna()
        asset_returns = series.pct_change().dropna()

        if len(asset_returns) < 2:
            continue

        annualized_vol = asset_returns.std(ddof=1) * np.sqrt(252)

        annualized_return = (
            (1 + asset_returns).prod() ** (252 / len(asset_returns)) - 1
        )

        if annualized_vol > 0:
            sharpe = (
                annualized_return - risk_free_rate
            ) / annualized_vol
        else:
            sharpe = np.nan

        drawdowns = series / series.cummax() - 1
        max_drawdown = drawdowns.min()

        # Illustrative score from 0 to 100.
        # 60% weight on volatility, 40% on maximum drawdown.
        volatility_component = min(annualized_vol / 0.60, 1.0)
        drawdown_component = min(abs(max_drawdown) / 0.60, 1.0)

        risk_score = 100 * (
            0.60 * volatility_component
            + 0.40 * drawdown_component
        )

        if risk_score < 30:
            category = "Lower"
        elif risk_score < 60:
            category = "Moderate"
        else:
            category = "Higher"

        risk_rows.append({
            "Asset": ticker,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe,
            "Maximum Drawdown": max_drawdown,
            "Risk Score": risk_score,
            "Risk Category": category
        })

    risk_table = pd.DataFrame(risk_rows)

    if not risk_table.empty:

        st.markdown("### Historical Risk Metrics")

        st.dataframe(
            risk_table.style.format({
                "Annualized Return": "{:.2%}",
                "Annualized Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Maximum Drawdown": "{:.2%}",
                "Risk Score": "{:.1f}"
            }),
            use_container_width=True
        )

        st.markdown("### Illustrative Investment Risk Score")

        risk_fig = px.bar(
            risk_table,
            x="Asset",
            y="Risk Score",
            color="Risk Category",
            range_y=[0, 100],
            title="Historical Risk Score (0–100)"
        )

        st.plotly_chart(
            risk_fig,
            use_container_width=True
        )

        st.markdown("#### How the score works")

        st.write(
            "Risk Score = 100 × [0.60 × min(annualized volatility / 0.60, 1) "
            "+ 0.40 × min(abs(maximum drawdown) / 0.60, 1)]. "
            "Higher scores indicate greater historical volatility and/or "
            "drawdown under this illustrative formula."
        )

        st.download_button(
            "Download Advanced Risk Report",
            data=risk_table.to_csv(index=False).encode("utf-8"),
            file_name="finsight_advanced_risk_analysis.csv",
            mime="text/csv"
        )

        st.caption(
            "The Sharpe ratio is a historical estimate. The risk score is "
            "an illustrative, unvalidated indicator, not a certified rating "
            "or investment recommendation."
        )

# -------------------- DATA DOWNLOAD --------------------

st.markdown("---")

st.download_button(
    "Download Filtered Historical Market Data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="finsight_filtered_market_data.csv",
    mime="text/csv"
)

st.caption(
    "FinSight | Educational financial analytics. Historical results do not "
    "predict future performance. Metrics exclude fees, taxes and slippage."
)
