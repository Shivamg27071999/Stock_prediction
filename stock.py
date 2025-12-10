# stock.py
import re
import math
import traceback
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Callable
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stock Predictor — Dark Visuals", layout="wide")

@st.cache_data(ttl=3600)
def raw_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _col_key_from_label(label):
    if isinstance(label, tuple) and len(label) > 0:
        return str(label[0]).lower()
    s = str(label).strip()
    m = re.match(r"^\(?['\"]?([^'\"\),]+)['\"]?", s)
    return m.group(1).lower() if m else s.lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    key_to_orig = {}
    for col in df.columns:
        key = _col_key_from_label(col)
        if key not in key_to_orig:
            key_to_orig[key] = col
    if "close" not in key_to_orig and "adj close" in key_to_orig:
        key_to_orig["close"] = key_to_orig["adj close"]
    def get(k):
        o = key_to_orig.get(k)
        if o is None:
            return None
        try:
            return df[o]
        except Exception:
            return None
    s_open = get("open"); s_high = get("high"); s_low = get("low"); s_close = get("close"); s_vol = get("volume")
    if s_close is None:
        return pd.DataFrame()
    if s_open is None: s_open = s_close.copy()
    if s_high is None: s_high = s_open.copy()
    if s_low is None: s_low = s_close.copy()
    if s_vol is None: s_vol = pd.Series(np.nan, index=s_close.index)
    out = pd.DataFrame({
        "Open": pd.to_numeric(s_open, errors="coerce"),
        "High": pd.to_numeric(s_high, errors="coerce"),
        "Low": pd.to_numeric(s_low, errors="coerce"),
        "Close": pd.to_numeric(s_close, errors="coerce"),
        "Volume": pd.to_numeric(s_vol, errors="coerce"),
    })
    return out.dropna()

def fetch_data_and_normalize(ticker: str, start: str, end: str) -> Tuple[pd.DataFrame, List[str]]:
    raw = raw_fetch(ticker, start, end)
    raw_cols = [str(c) for c in list(raw.columns)] if (raw is not None and not raw.empty) else []
    df_norm = normalize_columns(raw)
    return df_norm, raw_cols

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean(); roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100/(1+rs))
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UP"] = df["BB_MID"] + 2*df["BB_STD"]
    df["BB_LW"] = df["BB_MID"] - 2*df["BB_STD"]
    obv=[0]
    for i in range(1,len(df)):
        if df["Close"].iat[i] > df["Close"].iat[i-1]:
            obv.append(obv[-1] + (0 if pd.isna(df["Volume"].iat[i]) else df["Volume"].iat[i]))
        elif df["Close"].iat[i] < df["Close"].iat[i-1]:
            obv.append(obv[-1] - (0 if pd.isna(df["Volume"].iat[i]) else df["Volume"].iat[i]))
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df = df.drop(columns=["BB_STD"], errors="ignore")
    return df

def create_lag_features(df: pd.DataFrame, cols: List[str], lags: int) -> pd.DataFrame:
    frames=[]; names=[]
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        for i in range(1, lags+1):
            frames.append(s.shift(i)); names.append(f"{col}_lag_{i}")
    if not frames:
        return pd.DataFrame()
    lag_df = pd.concat(frames, axis=1)
    lag_df.columns = names
    return lag_df

def build_feature_matrix(df: pd.DataFrame, lags: int, include_indicators: bool) -> pd.DataFrame:
    if not {"Open","Close"}.issubset(df.columns):
        raise ValueError("Missing Open/Close")
    base = ["Open","Close"]
    inds=[]
    if include_indicators:
        inds = ["SMA_5","SMA_20","EMA_12","EMA_26","MACD","MACD_Signal","RSI_14","BB_MID","BB_UP","BB_LW","OBV"]
        missing = [c for c in inds if c not in df.columns]
        if missing:
            raise ValueError(f"Indicators missing: {missing}")
    cols = base + inds
    lag_df = create_lag_features(df, cols, lags)
    ma_o = df["Open"].rolling(lags).mean().rename("Open_MA")
    ma_c = df["Close"].rolling(lags).mean().rename("Close_MA")
    X = pd.concat([lag_df, ma_o, ma_c], axis=1).dropna()
    return X

def next_trading_days(start: datetime, n: int) -> List[datetime]:
    out=[]; d=start
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out

def iterative_forecast(model, df_recent: pd.DataFrame, predict_col: str, days: int, lags: int, feature_cols: List[str], fill_vals: Dict[str,float]) -> List[float]:
    series = df_recent[["Open","Close"]].copy().reset_index(drop=True)
    preds=[]
    for _ in range(days):
        if len(series) < lags:
            raise ValueError("Not enough rows for lags")
        last = series.iloc[-lags:]
        feat={}
        for i in range(1, lags+1):
            feat[f"Open_lag_{i}"] = last["Open"].iloc[-i]
            feat[f"Close_lag_{i}"] = last["Close"].iloc[-i]
        feat["Open_MA"] = last["Open"].mean(); feat["Close_MA"] = last["Close"].mean()
        row = pd.DataFrame([feat])
        for c in feature_cols:
            if c not in row.columns:
                row[c] = np.nan
        row = row[feature_cols].fillna(fill_vals).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        val = float(model.predict(row)[0])
        if predict_col == "Open":
            next_open = val; next_close = float(last["Close"].iloc[-1])
        else:
            next_open = float(last["Open"].iloc[-1]); next_close = val
        series = pd.concat([series, pd.DataFrame({"Open":[next_open],"Close":[next_close]})], ignore_index=True)
        preds.append(val)
    return preds

def expanding_backtest(model_factory: Callable[[], Any], X: pd.DataFrame, y: np.ndarray, initial_train_size: int, step: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    n = len(X)
    preds=[]; trues=[]; positions=[]
    i = initial_train_size
    if i >= n:
        return np.array(trues), np.array(preds), positions
    while i < n:
        model = model_factory()
        model.fit(X.iloc[:i], y[:i])
        yhat = model.predict(X.iloc[[i]])[0]
        preds.append(yhat); trues.append(y[i]); positions.append(i)
        i += step
    return np.array(trues), np.array(preds), positions

def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Any, Any]:
    if y_true is None or len(y_true)==0:
        return None, None
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(math.sqrt(mse))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return rmse, mape

def ridge_factory(alpha: float = 1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))

def rf_factory(params: dict = None):
    if not params:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RandomForestRegressor(n_estimators=200, random_state=42))
    n = int(params.get("n_estimators", 200))
    md = params.get("max_depth", None)
    mss = int(params.get("min_samples_split", 2))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RandomForestRegressor(n_estimators=n, max_depth=md, min_samples_split=mss, random_state=42))

st.title("Stock Predictor — Dark visuals")
with st.sidebar:
    ticker = st.text_input("Ticker (Yahoo)", value="YESBANK.NS")
    history = st.selectbox("History", ["1 year","3 years","5 years"], index=1)
    years = int(history.split()[0])
    include_ind = st.checkbox("Include indicators", value=True)
    lags = st.slider("Lag days", 3, 20, 5)
    predict_returns = st.checkbox("Predict returns (%) instead of price", value=False)
    model_mode = st.selectbox("Primary model", ["Ridge (Linear)","Random Forest"])
    ensemble_enable = st.checkbox("Enable ensemble", value=False)
    ensemble_members = st.multiselect("Ensemble members", ["Ridge (Linear)","Random Forest"], default=["Ridge (Linear)","Random Forest"])
    automl = st.checkbox("Enable AutoML (Ridge/RF)", value=False)
    automl_iters = st.number_input("AutoML iterations", min_value=10, max_value=100, value=20, step=5)
    test_step = st.slider("Backtest step", 1, 5, 1)
    init_train_frac = st.slider("Initial train fraction (%)", 10, 40, 20)
    run_btn = st.button("Run")

if not run_btn:
    st.info("Configure and click Run.")
    st.stop()

end_date = datetime.today()
start_date = end_date - timedelta(days=365*years + 30)
start_str = start_date.strftime("%Y-%m-%d"); end_str = end_date.strftime("%Y-%m-%d")
df_raw, raw_cols = fetch_data_and_normalize(ticker, start_str, end_str)
st.write("raw columns:", raw_cols)
if df_raw.empty:
    st.error("No data for ticker.")
    st.stop()

df_feat = add_indicators(df_raw) if include_ind else df_raw.copy()
X = build_feature_matrix(df_feat, lags=lags, include_indicators=include_ind)
if X.empty:
    st.error("Feature matrix empty; increase history or reduce lags.")
    st.stop()

if predict_returns:
    y_series = df_feat["Close"].shift(-1).loc[X.index] / df_feat["Close"].loc[X.index] - 1
    target_label = "Return"
else:
    y_series = df_feat["Close"].shift(-1).loc[X.index]
    target_label = "Price"

mask_valid = ~y_series.isna()
X = X.loc[mask_valid.index[mask_valid]]
y = y_series.loc[mask_valid.index[mask_valid]].values

if len(y) == 0:
    st.error("No valid targets after alignment.")
    st.stop()

st.write("features:", X.shape)
st.dataframe(X.head(5))

n = len(X)
initial_train_size = max(5, int(n * init_train_frac / 100))

def perform_automl(X_train: pd.DataFrame, y_train: np.ndarray, model_name: str, n_iter: int = 20):
    tscv = TimeSeriesSplit(n_splits=3)
    if model_name == "Ridge (Linear)":
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
        param_dist = {"ridge__alpha": np.logspace(-4, 2, 40)}
    else:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RandomForestRegressor(random_state=42))
        param_dist = {
            "randomforestregressor__n_estimators": [50,100,200],
            "randomforestregressor__max_depth": [None,5,10],
            "randomforestregressor__min_samples_split":[2,5,10]
        }
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=min(n_iter,40), cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42)
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_

models_to_run = []
models_to_run.append(model_mode)
if ensemble_enable:
    for m in ensemble_members:
        if m not in models_to_run:
            models_to_run.append(m)

model_results = {}
final_models = {}

def run_ridge_or_rf_with_optional_automl(name: str):
    if automl:
        tune_end = max(initial_train_size + 1, int(0.6 * len(X)))
        X_tune = X.iloc[:tune_end]; y_tune = y[:tune_end]
        try:
            best_est, best_params = perform_automl(X_tune, y_tune, "Ridge (Linear)" if name=="Ridge (Linear)" else "RF", n_iter=automl_iters)
            if name == "Ridge (Linear)":
                alpha = best_params.get("ridge__alpha", 1.0)
                factory = lambda: ridge_factory(alpha=alpha)
                final_model = ridge_factory(alpha=alpha)
            else:
                rf_p = {}
                if best_params:
                    for k,v in best_params.items():
                        if k.startswith("randomforestregressor__"):
                            rf_p[k.replace("randomforestregressor__","")] = v
                factory = lambda: rf_factory(rf_p)
                final_model = rf_factory(rf_p)
        except Exception:
            factory = ridge_factory if name=="Ridge (Linear)" else rf_factory
            final_model = factory()
    else:
        factory = ridge_factory if name=="Ridge (Linear)" else rf_factory
        final_model = factory()
    trues, preds, positions = expanding_backtest(factory, X, y, initial_train_size, test_step)
    try:
        final_model.fit(X, y)
    except Exception:
        final_model = factory(); final_model.fit(X, y)
    return {"y_true": trues, "y_pred": preds, "positions": positions, "final_model": final_model}

for mname in models_to_run:
    if mname == "Ridge (Linear)":
        res = run_ridge_or_rf_with_optional_automl("Ridge (Linear)")
        model_results["Ridge (Linear)"] = res
        final_models["Ridge (Linear)"] = res["final_model"]
    elif mname == "Random Forest":
        res = run_ridge_or_rf_with_optional_automl("Random Forest")
        model_results["Random Forest"] = res
        final_models["Random Forest"] = res["final_model"]

metrics = {}
for name, res in model_results.items():
    y_t = res.get("y_true", np.array([])); y_p = res.get("y_pred", np.array([]))
    rmse, mape = safe_metrics(y_t, y_p)
    metrics[name] = {"rmse": rmse, "mape": mape, "n": len(y_t)}

st.subheader("Backtest metrics")
colA, colB = st.columns(2)
with colA:
    for name, m in metrics.items():
        st.write(f"{name} — n={m['n']} — RMSE: {m['rmse'] if m['rmse'] is not None else 'N/A'}")
with colB:
    for name, m in metrics.items():
        st.write(f"{name} — MAPE: {m['mape'] if m['mape'] is not None else 'N/A'}")

fill_values = X.median().to_dict()
feature_cols = list(X.columns)
last_n = df_feat[-lags:].copy().reset_index(drop=True)
forecasts = {}
for name, model in final_models.items():
    try:
        preds = iterative_forecast(model, last_n, "Close", 5, lags, feature_cols, fill_values)
        forecasts[name] = preds
    except Exception as e:
        st.warning(f"Forecast failed for {name}: {e}")

if ensemble_enable and len(ensemble_members) > 0:
    member_preds = []
    for mem in ensemble_members:
        p = forecasts.get(mem)
        if p is not None:
            member_preds.append(p)
    if member_preds:
        forecasts["Ensemble"] = list(np.mean(np.array(member_preds), axis=0))

next_days = next_trading_days(datetime.today(), 5)
pred_df = pd.DataFrame({"Date":[d.strftime("%Y-%m-%d") for d in next_days]})
for k,v in forecasts.items():
    pred_df[k] = [round(float(x),6) for x in v]
pred_df = pred_df.set_index("Date")
st.subheader("Next-5 predictions")
st.dataframe(pred_df)
if not pred_df.empty:
    st.download_button("Download predictions CSV", pred_df.to_csv().encode(), "predictions_next5.csv")

chosen_for_plot = None
for choice in ["Ridge (Linear)","Random Forest"]:
    if choice in model_results and len(model_results[choice].get("y_true",[]))>0:
        chosen_for_plot = choice
        break

plot_template = "plotly_dark"

if chosen_for_plot:
    res = model_results[chosen_for_plot]
    y_t = res.get("y_true", np.array([])); y_p = res.get("y_pred", np.array([])); pos = res.get("positions",[])
    dates = [df_feat.index[p] for p in pos]
    dfplot = pd.DataFrame({"Date":dates, "Actual":y_t, "Pred":y_p}).set_index("Date")
    st.subheader(f"Backtest Predicted vs Actual — {chosen_for_plot}")
    fig = go.Figure(layout_template=plot_template)
    fig.add_trace(go.Scatter(x=dfplot.index, y=dfplot["Actual"], mode="lines+markers", name="Actual",
                             hovertemplate="Date: %{x}<br>Actual: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=dfplot.index, y=dfplot["Pred"], mode="lines+markers", name="Predicted",
                             hovertemplate="Date: %{x}<br>Pred: %{y:.4f}<extra></extra>"))
    fig.update_layout(template=plot_template, height=520, legend=dict(x=0.02,y=0.98))
    st.plotly_chart(fig, use_container_width=True)
    resid = dfplot["Actual"].values - dfplot["Pred"].values
    fig2 = px.histogram(resid, nbins=30, title="Residuals (Backtest)", template=plot_template)
    fig2.update_layout(height=360)
    fig2.update_traces(hovertemplate="Residual: %{x}<br>Count: %{y}<extra></extra>")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No backtest points to plot.")

recent = df_feat[["Close"]].reset_index().tail(200)
fig3 = go.Figure(layout_template=plot_template)
fig3.add_trace(go.Scatter(x=recent.iloc[:,0], y=recent["Close"], mode="lines", name="Recent Close",
                          hovertemplate="Date: %{x}<br>Close: %{y:.4f}<extra></extra>"))
for k,v in forecasts.items():
    fig3.add_trace(go.Scatter(x=[d for d in next_days], y=v, mode="markers+lines", name=f"Pred: {k}",
                              hovertemplate="Date: %{x}<br>Pred: %{y:.4f}<extra></extra>"))
fig3.update_layout(template=plot_template, height=520, legend=dict(x=0.02,y=0.98))
st.plotly_chart(fig3, use_container_width=True)

st.success("Done. Dark visuals applied.")
