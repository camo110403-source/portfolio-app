import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.optimize import minimize
import datetime
import io
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Analytics", layout="wide")
st.title("Interactive Portfolio Analytics")

# ── Cached optimization helpers ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_portfolio_optimization(returns_csv, rf):
    """Cache GMV and tangency optimization so slider interactions don't re-run it."""
    rets = pd.read_csv(io.StringIO(returns_csv), index_col=0, parse_dates=True)
    mu = rets.mean()
    cov = rets.cov()
    n = len(rets.columns)
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    w0 = np.array([1 / n] * n)

    def _vol(w):
        return np.sqrt(w @ cov.values @ w) * np.sqrt(252)

    def _neg_sharpe(w):
        r = np.sum(mu * w) * 252
        v = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
        return -(r - rf) / v

    gmv_res = minimize(_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    tan_res = minimize(_neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return (
        gmv_res.x.tolist() if gmv_res.success else None,
        gmv_res.success,
        tan_res.x.tolist() if tan_res.success else None,
        tan_res.success,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_efficient_frontier(returns_csv, rf, n_points=80):
    """Cache the efficient frontier computation (80 minimize calls)."""
    rets = pd.read_csv(io.StringIO(returns_csv), index_col=0, parse_dates=True)
    mu = rets.mean()
    cov = rets.cov()
    n = len(rets.columns)
    bounds = [(0, 1)] * n
    w0 = np.array([1 / n] * n)

    def _vol(w):
        return np.sqrt(w @ cov.values @ w) * np.sqrt(252)

    target_returns = np.linspace(mu.min() * 252, mu.max() * 252, n_points)
    frontier_vols, frontier_rets = [], []
    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.sum(mu * w) * 252 - t},
        ]
        res = minimize(_vol, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            frontier_vols.append(res.fun)
            frontier_rets.append(target)
    return frontier_vols, frontier_rets


# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Portfolio Settings")

ticker_input = st.sidebar.text_area(
    "Enter 3–10 stock tickers (one per line or comma-separated)",
    value="AAPL\nMSFT\nGOOGL\nAMZN\nJPM"
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.date(2019, 1, 1),
    max_value=datetime.date.today() - datetime.timedelta(days=730)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.date.today(),
    max_value=datetime.date.today()
)

rf_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=0.0, max_value=20.0,
    value=2.0, step=0.1
) / 100

# ── Parse and validate tickers ─────────────────────────────────────────────────
raw = ticker_input.replace(",", "\n").upper().split()
tickers = [t.strip() for t in raw if t.strip()]

if len(tickers) < 3:
    st.error("Please enter at least 3 ticker symbols.")
    st.stop()
if len(tickers) > 10:
    st.error("Please enter no more than 10 ticker symbols.")
    st.stop()

# ── Validate date range ────────────────────────────────────────────────────────
if (end_date - start_date).days < 730:
    st.error("Please select a date range of at least 2 years.")
    st.stop()

# ── Data download ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Downloading data...")
def load_data(tickers, start, end):
    all_tickers = list(tickers) + ["^GSPC"]
    raw_data = {}
    failed = []

    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 100:
                failed.append(ticker)
            else:
                raw_data[ticker] = df["Close"]
        except Exception:
            failed.append(ticker)

    return raw_data, failed

with st.spinner("Downloading price data..."):
    raw_data, failed = load_data(tuple(tickers), str(start_date), str(end_date))

# ── Handle failures ────────────────────────────────────────────────────────────
user_failed = [t for t in failed if t != "^GSPC"]
if user_failed:
    st.error(f"Could not download data for: {', '.join(user_failed)}. Please check these tickers.")
    st.stop()

if "^GSPC" in failed:
    st.warning("Could not download S&P 500 benchmark data. Benchmark comparisons will be unavailable.")

# ── Build price DataFrame ──────────────────────────────────────────────────────
prices = pd.DataFrame({t: raw_data[t].squeeze() for t in tickers if t in raw_data})
benchmark = raw_data.get("^GSPC", None)

# ── Handle missing data ────────────────────────────────────────────────────────
missing_pct = prices.isnull().mean()
bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
if bad_tickers:
    st.warning(f"Dropping {bad_tickers} due to >5% missing data.")
    prices = prices.drop(columns=bad_tickers)
    tickers = [t for t in tickers if t not in bad_tickers]

if len(tickers) < 3:
    st.error(
        f"After removing tickers with insufficient data, fewer than 3 valid tickers remain "
        f"({', '.join(tickers) if tickers else 'none'}). Please select different tickers."
    )
    st.stop()

prices = prices.dropna()
if benchmark is not None:
    benchmark = benchmark.squeeze().reindex(prices.index).dropna()

# ── Compute returns ────────────────────────────────────────────────────────────
returns = prices.pct_change().dropna()
if benchmark is not None:
    bench_returns = benchmark.pct_change().dropna().reindex(returns.index).dropna()

st.success(f"Loaded data for: {', '.join(tickers)}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Exploratory Analysis",
    "Risk Analysis",
    "Correlation & Covariance",
    "Portfolio Optimization",
    "Estimation Window",
    "About"
])
# ── Tab 1: Exploratory Analysis ────────────────────────────────────────────────
with tab1:
    st.header("Exploratory Analysis")

    # Summary statistics table
    st.subheader("Summary Statistics")
    summary_data = {}
    for ticker in tickers:
        r = returns[ticker]
        summary_data[ticker] = {
            "Ann. Return": f"{r.mean() * 252:.2%}",
            "Ann. Volatility": f"{r.std() * np.sqrt(252):.2%}",
            "Skewness": f"{r.skew():.3f}",
            "Kurtosis": f"{r.kurtosis():.3f}",
            "Min Daily Return": f"{r.min():.2%}",
            "Max Daily Return": f"{r.max():.2%}",
        }
    if benchmark is not None:
        summary_data["S&P 500"] = {
            "Ann. Return": f"{bench_returns.mean() * 252:.2%}",
            "Ann. Volatility": f"{bench_returns.std() * np.sqrt(252):.2%}",
            "Skewness": f"{bench_returns.skew():.3f}",
            "Kurtosis": f"{bench_returns.kurtosis():.3f}",
            "Min Daily Return": f"{bench_returns.min():.2%}",
            "Max Daily Return": f"{bench_returns.max():.2%}",
        }
    st.dataframe(pd.DataFrame(summary_data).T)

    st.divider()

    # Cumulative wealth index chart
    st.subheader("Cumulative Wealth Index ($10,000 invested)")
    selected_stocks = st.multiselect(
        "Select stocks to display",
        options=tickers,
        default=tickers
    )
    fig_wealth = go.Figure()
    for ticker in selected_stocks:
        wealth = 10000 * (1 + returns[ticker]).cumprod()
        fig_wealth.add_trace(go.Scatter(
            x=wealth.index, y=wealth,
            mode="lines", name=ticker
        ))
    if benchmark is not None:
        bench_wealth = 10000 * (1 + bench_returns).cumprod()
        fig_wealth.add_trace(go.Scatter(
            x=bench_wealth.index, y=bench_wealth,
            mode="lines", name="S&P 500",
            line=dict(dash="dash", color="black")
        ))
    fig_wealth.update_layout(
        title="Cumulative Wealth Index",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    st.divider()

    # Distribution plot
    st.subheader("Return Distribution")
    selected_dist = st.selectbox("Select a stock", options=tickers)
    plot_type = st.radio("Plot type", ["Histogram + Normal Fit", "Q-Q Plot"], horizontal=True)

    r = returns[selected_dist].dropna()

    if plot_type == "Histogram + Normal Fit":
        mu, sigma = r.mean(), r.std()
        x_range = np.linspace(r.min(), r.max(), 200)
        normal_curve = stats.norm.pdf(x_range, mu, sigma)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=r, nbinsx=60, histnorm="probability density",
            name="Daily Returns", opacity=0.6
        ))
        fig_dist.add_trace(go.Scatter(
            x=x_range, y=normal_curve,
            mode="lines", name="Normal Fit",
            line=dict(color="red", width=2)
        ))
        fig_dist.update_layout(
            title=f"{selected_dist} Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_white"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    else:
        qq = stats.probplot(r, dist="norm")
        theoretical_q = qq[0][0]
        sample_q = qq[0][1]
        slope, intercept = qq[1][0], qq[1][1]
        line_y = slope * np.array([theoretical_q[0], theoretical_q[-1]]) + intercept
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_q, y=sample_q,
            mode="markers", name="Quantiles"
        ))
        fig_qq.add_trace(go.Scatter(
            x=[theoretical_q[0], theoretical_q[-1]], y=line_y,
            mode="lines", name="Normal Reference",
            line=dict(color="red", width=2)
        ))
        fig_qq.update_layout(
            title=f"{selected_dist} Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white"
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        # ── Tab 2: Risk Analysis ───────────────────────────────────────────────────────
with tab2:
    st.header("Risk Analysis")

    # Rolling volatility
    st.subheader("Rolling Annualized Volatility")
    vol_window = st.select_slider(
        "Rolling window (days)",
        options=[30, 60, 90, 120],
        value=60
    )
    fig_vol = go.Figure()
    for ticker in tickers:
        rolling_vol = returns[ticker].rolling(vol_window).std() * np.sqrt(252)
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol,
            mode="lines", name=ticker
        ))
    fig_vol.update_layout(
        title=f"Rolling {vol_window}-Day Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    # Drawdown analysis
    st.subheader("Drawdown Analysis")
    selected_dd = st.selectbox("Select a stock", options=tickers, key="dd_stock")
    wealth = (1 + returns[selected_dd]).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()

    st.metric("Maximum Drawdown", f"{max_dd:.2%}")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        mode="lines", name="Drawdown",
        fill="tozeroy", line=dict(color="red")
    ))
    fig_dd.update_layout(
        title=f"{selected_dd} Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # Risk-adjusted metrics
    st.subheader("Risk-Adjusted Metrics")
    rf_daily = rf_rate / 252
    metrics_data = {}

    for ticker in tickers:
        r = returns[ticker]
        excess = r - rf_daily
        downside = r[r < rf_daily] - rf_daily
        sharpe = (excess.mean() / r.std()) * np.sqrt(252)
        sortino = (excess.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 else np.nan
        metrics_data[ticker] = {
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino:.3f}"
        }

    if benchmark is not None:
        excess_b = bench_returns - rf_daily
        downside_b = bench_returns[bench_returns < rf_daily] - rf_daily
        sharpe_b = (excess_b.mean() / bench_returns.std()) * np.sqrt(252)
        sortino_b = (excess_b.mean() / downside_b.std()) * np.sqrt(252) if len(downside_b) > 0 else np.nan
        metrics_data["S&P 500"] = {
            "Sharpe Ratio": f"{sharpe_b:.3f}",
            "Sortino Ratio": f"{sortino_b:.3f}"
        }

    st.dataframe(pd.DataFrame(metrics_data).T)
    # ── Tab 3: Correlation & Covariance ───────────────────────────────────────────
with tab3:
    st.header("Correlation & Covariance Analysis")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = returns.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    fig_corr.update_layout(
        title="Pairwise Correlation Matrix",
        template="plotly_white"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # Rolling correlation
    st.subheader("Rolling Correlation")
    col_a, col_b = st.columns(2)
    with col_a:
        stock_a = st.selectbox("Stock A", options=tickers, index=0)
    with col_b:
        stock_b = st.selectbox("Stock B", options=tickers, index=1)
    roll_window = st.select_slider(
        "Rolling window (days)",
        options=[30, 60, 90, 120],
        value=60,
        key="corr_window"
    )

    if stock_a == stock_b:
        st.warning("Please select two different stocks.")
    else:
        rolling_corr = returns[stock_a].rolling(roll_window).corr(returns[stock_b])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            mode="lines", name=f"{stock_a} vs {stock_b}"
        ))
        fig_rc.update_layout(
            title=f"{roll_window}-Day Rolling Correlation: {stock_a} vs {stock_b}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            template="plotly_white"
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    st.divider()

    # Covariance matrix
    st.subheader("Covariance Matrix")
    with st.expander("Show Covariance Matrix"):
        cov_matrix = returns.cov()
        st.dataframe(cov_matrix.style.format("{:.6f}"))
        # ── Tab 4: Portfolio Optimization ─────────────────────────────────────────────
with tab4:
    st.header("Portfolio Construction & Optimization")

    n = len(tickers)
    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    rf_daily = rf_rate / 252

    # ── Helper functions ───────────────────────────────────────────────────────
    def portfolio_stats(weights):
        weights = np.array(weights)
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights) * np.sqrt(252)
        excess = returns @ weights - rf_daily
        downside = excess[excess < 0]
        sortino = (excess.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 else np.nan
        sharpe = (port_return - rf_rate) / port_vol
        wealth = (1 + returns @ weights).cumprod()
        peak = wealth.cummax()
        max_dd = ((wealth - peak) / peak).min()
        return port_return, port_vol, sharpe, sortino, max_dd

    def compute_prc(weights, cov):
        weights = np.array(weights)
        port_var = weights @ cov.values @ weights
        marginal = cov.values @ weights
        prc = (weights * marginal) / port_var
        return prc

    # ── Equal weight portfolio ─────────────────────────────────────────────────
    st.subheader("Equal-Weight Portfolio (1/N)")
    ew_weights = np.array([1 / n] * n)
    ew_ret, ew_vol, ew_sharpe, ew_sortino, ew_maxdd = portfolio_stats(ew_weights)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ann. Return", f"{ew_ret:.2%}")
    col2.metric("Ann. Volatility", f"{ew_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{ew_sharpe:.3f}")
    col4.metric("Sortino Ratio", f"{ew_sortino:.3f}")
    col5.metric("Max Drawdown", f"{ew_maxdd:.2%}")

    st.divider()

    # ── Optimization (cached) ──────────────────────────────────────────────────
    with st.spinner("Running portfolio optimization..."):
        returns_csv = returns.to_csv()
        gmv_w_list, gmv_ok, tan_w_list, tan_ok = run_portfolio_optimization(returns_csv, rf_rate)

    if not gmv_ok:
        st.error("GMV optimization did not converge. Try different tickers or a longer date range.")
    if not tan_ok:
        st.error("Tangency optimization did not converge. Try different tickers or a longer date range.")

    _opt_ok = gmv_ok and tan_ok
    gmv_weights = np.array(gmv_w_list) if gmv_ok else ew_weights
    tan_weights = np.array(tan_w_list) if tan_ok else ew_weights

    # ── Display GMV ────────────────────────────────────────────────────────────
    st.subheader("Global Minimum Variance (GMV) Portfolio")
    gmv_ret, gmv_vol, gmv_sharpe, gmv_sortino, gmv_maxdd = portfolio_stats(gmv_weights)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ann. Return", f"{gmv_ret:.2%}")
    col2.metric("Ann. Volatility", f"{gmv_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{gmv_sharpe:.3f}")
    col4.metric("Sortino Ratio", f"{gmv_sortino:.3f}")
    col5.metric("Max Drawdown", f"{gmv_maxdd:.2%}")

    fig_gmv_w = go.Figure(go.Bar(
        x=tickers, y=gmv_weights, name="GMV Weights"
    ))
    fig_gmv_w.update_layout(
        title="GMV Portfolio Weights",
        xaxis_title="Ticker", yaxis_title="Weight",
        template="plotly_white"
    )
    st.plotly_chart(fig_gmv_w, use_container_width=True)

    # GMV Risk Contribution
    gmv_prc = compute_prc(gmv_weights, cov_matrix)
    st.markdown("**Risk Contribution — GMV Portfolio**")
    st.caption("A stock with a higher risk contribution than its weight is a disproportionate source of portfolio volatility.")
    fig_gmv_prc = go.Figure(go.Bar(x=tickers, y=gmv_prc, name="Risk Contribution"))
    fig_gmv_prc.update_layout(
        title="GMV Percentage Risk Contribution",
        xaxis_title="Ticker", yaxis_title="Risk Contribution",
        template="plotly_white"
    )
    st.plotly_chart(fig_gmv_prc, use_container_width=True)

    st.divider()

    # ── Display Tangency ───────────────────────────────────────────────────────
    st.subheader("Maximum Sharpe Ratio (Tangency) Portfolio")
    tan_ret, tan_vol, tan_sharpe, tan_sortino, tan_maxdd = portfolio_stats(tan_weights)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ann. Return", f"{tan_ret:.2%}")
    col2.metric("Ann. Volatility", f"{tan_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{tan_sharpe:.3f}")
    col4.metric("Sortino Ratio", f"{tan_sortino:.3f}")
    col5.metric("Max Drawdown", f"{tan_maxdd:.2%}")

    fig_tan_w = go.Figure(go.Bar(
        x=tickers, y=tan_weights, name="Tangency Weights"
    ))
    fig_tan_w.update_layout(
        title="Tangency Portfolio Weights",
        xaxis_title="Ticker", yaxis_title="Weight",
        template="plotly_white"
    )
    st.plotly_chart(fig_tan_w, use_container_width=True)

    # Tangency Risk Contribution
    tan_prc = compute_prc(tan_weights, cov_matrix)
    st.markdown("**Risk Contribution — Tangency Portfolio**")
    st.caption("A stock with a higher risk contribution than its weight is a disproportionate source of portfolio volatility.")
    fig_tan_prc = go.Figure(go.Bar(x=tickers, y=tan_prc, name="Risk Contribution"))
    fig_tan_prc.update_layout(
        title="Tangency Percentage Risk Contribution",
        xaxis_title="Ticker", yaxis_title="Risk Contribution",
        template="plotly_white"
    )
    st.plotly_chart(fig_tan_prc, use_container_width=True)

    st.divider()

    # ── Custom Portfolio ───────────────────────────────────────────────────────
    st.subheader("Custom Portfolio")
    st.caption("Adjust the sliders to set your desired allocation. Weights are automatically normalized to sum to 1.")

    raw_weights = []
    for ticker in tickers:
        w = st.slider(f"{ticker} weight", 0.0, 1.0, 1/n, 0.01, key=f"slider_{ticker}")
        raw_weights.append(w)

    total = sum(raw_weights)
    if total == 0:
        st.error("At least one weight must be greater than zero.")
        custom_weights = ew_weights  # fallback so downstream code still runs
    else:
        custom_weights = np.array(raw_weights) / total
    st.markdown("**Normalized Weights:**")
    norm_df = pd.DataFrame({"Ticker": tickers, "Weight": [f"{w:.2%}" for w in custom_weights]})
    st.dataframe(norm_df.set_index("Ticker").T)

    cust_ret, cust_vol, cust_sharpe, cust_sortino, cust_maxdd = portfolio_stats(custom_weights)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ann. Return", f"{cust_ret:.2%}")
    col2.metric("Ann. Volatility", f"{cust_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{cust_sharpe:.3f}")
    col4.metric("Sortino Ratio", f"{cust_sortino:.3f}")
    col5.metric("Max Drawdown", f"{cust_maxdd:.2%}")

    st.divider()

    # ── Efficient Frontier ─────────────────────────────────────────────────────
    st.subheader("Efficient Frontier")
    st.caption("The efficient frontier shows the set of portfolios with the highest return for a given level of risk. The Capital Allocation Line (CAL) connects the risk-free rate to the tangency portfolio — any point on the CAL represents a mix of the risk-free asset and the tangency portfolio.")

    with st.spinner("Computing efficient frontier..."):
        frontier_vols, frontier_rets = compute_efficient_frontier(returns_csv, rf_rate)

    fig_ef = go.Figure()

    # Efficient frontier line
    fig_ef.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets,
        mode="lines", name="Efficient Frontier",
        line=dict(color="blue", width=2)
    ))

    # Individual stocks
    for ticker in tickers:
        s_vol = returns[ticker].std() * np.sqrt(252)
        s_ret = returns[ticker].mean() * 252
        fig_ef.add_trace(go.Scatter(
            x=[s_vol], y=[s_ret],
            mode="markers+text", name=ticker,
            text=[ticker], textposition="top center",
            marker=dict(size=10)
        ))

    # Portfolios
    fig_ef.add_trace(go.Scatter(
        x=[ew_vol], y=[ew_ret], mode="markers+text",
        name="Equal Weight", text=["EW"], textposition="top center",
        marker=dict(size=14, symbol="star", color="green")
    ))
    fig_ef.add_trace(go.Scatter(
        x=[gmv_vol], y=[gmv_ret], mode="markers+text",
        name="GMV", text=["GMV"], textposition="top center",
        marker=dict(size=14, symbol="diamond", color="purple")
    ))
    fig_ef.add_trace(go.Scatter(
        x=[tan_vol], y=[tan_ret], mode="markers+text",
        name="Tangency", text=["TAN"], textposition="top center",
        marker=dict(size=14, symbol="diamond", color="orange")
    ))
    fig_ef.add_trace(go.Scatter(
        x=[cust_vol], y=[cust_ret], mode="markers+text",
        name="Custom", text=["Custom"], textposition="top center",
        marker=dict(size=14, symbol="x", color="red")
    ))

    # S&P 500
    if benchmark is not None:
        sp_vol = bench_returns.std() * np.sqrt(252)
        sp_ret = bench_returns.mean() * 252
        fig_ef.add_trace(go.Scatter(
            x=[sp_vol], y=[sp_ret], mode="markers+text",
            name="S&P 500", text=["S&P 500"], textposition="top center",
            marker=dict(size=14, symbol="circle", color="black")
        ))

    # Capital Allocation Line
    if frontier_vols:
        cal_vols = np.linspace(0, max(frontier_vols) * 1.2, 100)
        cal_rets = rf_rate + tan_sharpe * cal_vols
        fig_ef.add_trace(go.Scatter(
            x=cal_vols, y=cal_rets,
            mode="lines", name="CAL",
            line=dict(dash="dash", color="orange", width=1.5)
        ))
    else:
        st.warning("Could not compute efficient frontier — optimizations did not converge.")

    fig_ef.update_layout(
        title="Efficient Frontier with CAL",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white"
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    st.divider()

    # ── Portfolio Comparison ───────────────────────────────────────────────────
    st.subheader("Portfolio Comparison")

    fig_comp = go.Figure()
    for weights, name in [
        (ew_weights, "Equal Weight"),
        (gmv_weights, "GMV"),
        (tan_weights, "Tangency"),
        (custom_weights, "Custom")
    ]:
        wealth = 10000 * (1 + returns @ weights).cumprod()
        fig_comp.add_trace(go.Scatter(
            x=wealth.index, y=wealth,
            mode="lines", name=name
        ))
    if benchmark is not None:
        bench_wealth = 10000 * (1 + bench_returns).cumprod()
        fig_comp.add_trace(go.Scatter(
            x=bench_wealth.index, y=bench_wealth,
            mode="lines", name="S&P 500",
            line=dict(dash="dash", color="black")
        ))
    fig_comp.update_layout(
        title="Cumulative Wealth Index — Portfolio Comparison",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Summary comparison table
    st.subheader("Portfolio Summary Table")
    summary = {
        "Equal Weight": [f"{ew_ret:.2%}", f"{ew_vol:.2%}", f"{ew_sharpe:.3f}", f"{ew_sortino:.3f}", f"{ew_maxdd:.2%}"],
        "GMV":          [f"{gmv_ret:.2%}", f"{gmv_vol:.2%}", f"{gmv_sharpe:.3f}", f"{gmv_sortino:.3f}", f"{gmv_maxdd:.2%}"],
        "Tangency":     [f"{tan_ret:.2%}", f"{tan_vol:.2%}", f"{tan_sharpe:.3f}", f"{tan_sortino:.3f}", f"{tan_maxdd:.2%}"],
        "Custom":       [f"{cust_ret:.2%}", f"{cust_vol:.2%}", f"{cust_sharpe:.3f}", f"{cust_sortino:.3f}", f"{cust_maxdd:.2%}"],
    }
    if benchmark is not None:
        sp_ret_ann = bench_returns.mean() * 252
        sp_vol_ann = bench_returns.std() * np.sqrt(252)
        sp_sharpe = (sp_ret_ann - rf_rate) / sp_vol_ann
        sp_excess = bench_returns - rf_daily
        sp_downside = sp_excess[sp_excess < 0]
        sp_sortino = (sp_excess.mean() / sp_downside.std()) * np.sqrt(252)
        sp_wealth = (1 + bench_returns).cumprod()
        sp_peak = sp_wealth.cummax()
        sp_maxdd = ((sp_wealth - sp_peak) / sp_peak).min()
        summary["S&P 500"] = [f"{sp_ret_ann:.2%}", f"{sp_vol_ann:.2%}", f"{sp_sharpe:.3f}", f"{sp_sortino:.3f}", f"{sp_maxdd:.2%}"]

    summary_df = pd.DataFrame(summary, index=["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"])
    st.dataframe(summary_df)
    # ── Tab 5: Estimation Window Sensitivity ──────────────────────────────────────
with tab5:
    st.header("Estimation Window Sensitivity")
    st.caption(
        "Mean-variance optimization is sensitive to its inputs. Small changes in estimated "
        "returns or covariances can produce dramatically different portfolio weights. "
        "This section lets you see that directly by comparing results across different lookback windows."
    )

    # Determine available windows based on date range
    total_days = (end_date - start_date).days
    available_windows = {"Full Sample": None}
    if total_days >= 365 * 5:
        available_windows["5 Years"] = 252 * 5
    if total_days >= 365 * 3:
        available_windows["3 Years"] = 252 * 3
    if total_days >= 365 * 1:
        available_windows["1 Year"] = 252 * 1

    if len(available_windows) < 2:
        st.warning("Select a longer date range to enable sensitivity analysis.")
    else:
        @st.cache_data(ttl=3600, show_spinner="Running sensitivity analysis...")
        def run_sensitivity(tickers_tuple, returns_json, rf, windows_json):
            import json
            import io
            rets = pd.read_csv(io.StringIO(returns_json), index_col=0, parse_dates=True)
            windows = json.loads(windows_json)
            results = {}
            bounds = [(0, 1)] * len(tickers_tuple)
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            w0 = np.array([1 / len(tickers_tuple)] * len(tickers_tuple))

            for label, n_days in windows.items():
                if n_days is None:
                    r_slice = rets
                else:
                    r_slice = rets.iloc[-n_days:]

                mu = r_slice.mean()
                cov = r_slice.cov()

                def pv(w):
                    return np.sqrt(w @ cov.values @ w) * np.sqrt(252)

                def ns(w):
                    r = np.sum(mu * w) * 252
                    v = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
                    return -(r - rf) / v

                gmv_res = minimize(pv, w0, method="SLSQP",
                                   bounds=bounds, constraints=constraints)
                tan_res = minimize(ns, w0, method="SLSQP",
                                   bounds=bounds, constraints=constraints)

                gmv_w = gmv_res.x if gmv_res.success else w0
                tan_w = tan_res.x if tan_res.success else w0

                gmv_ret = np.sum(mu * gmv_w) * 252
                gmv_vol = np.sqrt(gmv_w @ cov.values @ gmv_w) * np.sqrt(252)
                tan_ret = np.sum(mu * tan_w) * 252
                tan_vol = np.sqrt(tan_w @ cov.values @ tan_w) * np.sqrt(252)
                tan_sharpe = (tan_ret - rf) / tan_vol

                results[label] = {
                    "gmv_weights": gmv_w.tolist(),
                    "gmv_ret": gmv_ret,
                    "gmv_vol": gmv_vol,
                    "tan_weights": tan_w.tolist(),
                    "tan_ret": tan_ret,
                    "tan_vol": tan_vol,
                    "tan_sharpe": tan_sharpe,
                }
            return results

        import json
        import io
        with st.spinner("Running sensitivity analysis..."):
            returns_csv = returns.to_csv()
            sens_results = run_sensitivity(
                tuple(tickers),
                returns_csv,
                rf_rate,
                json.dumps({k: v for k, v in available_windows.items()})
            )
        # GMV comparison table
        st.subheader("GMV Portfolio — Across Lookback Windows")
        gmv_rows = {}
        for label, res in sens_results.items():
            row = {t: f"{w:.2%}" for t, w in zip(tickers, res["gmv_weights"])}
            row["Ann. Return"] = f"{res['gmv_ret']:.2%}"
            row["Ann. Volatility"] = f"{res['gmv_vol']:.2%}"
            gmv_rows[label] = row
        st.dataframe(pd.DataFrame(gmv_rows).T)

        # Tangency comparison table
        st.subheader("Tangency Portfolio — Across Lookback Windows")
        tan_rows = {}
        for label, res in sens_results.items():
            row = {t: f"{w:.2%}" for t, w in zip(tickers, res["tan_weights"])}
            row["Ann. Return"] = f"{res['tan_ret']:.2%}"
            row["Ann. Volatility"] = f"{res['tan_vol']:.2%}"
            row["Sharpe Ratio"] = f"{res['tan_sharpe']:.3f}"
            tan_rows[label] = row
        st.dataframe(pd.DataFrame(tan_rows).T)

        # Grouped bar chart of weights
        st.subheader("GMV Weights Across Windows")
        fig_sens = go.Figure()
        for label, res in sens_results.items():
            fig_sens.add_trace(go.Bar(
                name=label,
                x=tickers,
                y=res["gmv_weights"]
            ))
        fig_sens.update_layout(
            barmode="group",
            title="GMV Portfolio Weights by Estimation Window",
            xaxis_title="Ticker",
            yaxis_title="Weight",
            template="plotly_white"
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        # ── Tab 6: About ───────────────────────────────────────────────────────────────
with tab6:
    st.header("About & Methodology")

    st.subheader("Application Overview")
    st.write(
        "This application allows users to construct and analyze equity portfolios in real time. "
        "Users can enter between 3 and 10 stock tickers, select a date range, and explore "
        "return characteristics, risk metrics, correlations, and optimal portfolio allocations."
    )

    st.subheader("Data Source")
    st.write(
        "All price data is downloaded from Yahoo Finance using the yfinance library. "
        "Adjusted closing prices are used to account for dividends and stock splits. "
        "The S&P 500 (^GSPC) is used as the market benchmark for comparison purposes only "
        "and is not included in any portfolio optimization."
    )

    st.subheader("Key Assumptions")
    st.markdown(
        """
        - **Returns:** Simple (arithmetic) returns are used throughout: r = (P_t / P_{t-1}) - 1
        - **Annualization:** Mean daily return × 252 for annualized return; daily standard deviation × √252 for annualized volatility.
        - **Risk-free rate:** User-specified annualized rate (default 2%), converted to a daily rate by dividing by 252.
        - **Cumulative returns:** Computed as (1 + r).cumprod() on simple returns.
        - **Portfolio variance:** Computed using the full quadratic form w'Σw.
        """
    )

    st.subheader("Analytical Methods")
    st.markdown(
        """
        - **Sharpe Ratio:** (Annualized Return − Risk-Free Rate) / Annualized Volatility
        - **Sortino Ratio:** (Annualized Excess Return) / (Annualized Downside Deviation), where downside deviation uses only returns below the risk-free rate.
        - **Global Minimum Variance (GMV) Portfolio:** Minimizes portfolio volatility subject to no-short-selling constraints (weights between 0 and 1, sum to 1).
        - **Tangency Portfolio:** Maximizes the Sharpe ratio subject to the same constraints, implemented by minimizing the negative Sharpe ratio.
        - **Efficient Frontier:** Generated by solving a constrained optimization at each target return level.
        - **Risk Contribution:** For each asset i, PRC_i = w_i × (Σw)_i / σ²_p. Values sum to 1.
        - **Rolling calculations:** Rolling volatility and correlation produce NaN for the first (window − 1) observations — this is expected.
        - **Estimation Window Sensitivity:** The optimizer is re-run using only the most recent N trading days of data to illustrate how sensitive results are to the choice of lookback period.
        """
    )

    st.subheader("Libraries Used")
    st.markdown(
        """
        - **streamlit** — web application framework
        - **yfinance** — market data download
        - **pandas / numpy** — data manipulation and computation
        - **plotly** — interactive charts
        - **scipy** — statistical functions and portfolio optimization
        """
    )