import streamlit as st
import pandas as pd
import akshare as ak
import os
from datetime import datetime
import base64

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

st.set_page_config(page_title="SIXQUARE AI选股", layout="wide")

def save_uploaded_file(uploadedfile):
    with open(os.path.join(DATA_DIR, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def get_stock_data(symbol):
    try:
        df = ak.stock_us_daily(symbol=symbol)
        return df
    except Exception as e:
        return None

def calculate_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

# 策略与tradingview一致
def stock_signal_last_row(df, ema_len=5, threshold=3):
    df = df.copy()
    df['EMA'] = calculate_ema(df['close'], ema_len)
    df['below_EMA'] = df['close'] < df['EMA']
    df['below_count'] = df['below_EMA'].rolling(window=threshold, min_periods=1).sum()
    last = df.iloc[-1]
    # 连续threshold根收盘价低于EMA
    return last['below_count'] == threshold

def backtest_single(symbol, df, ema_len=5, threshold=3):
    df = df.copy()
    df['EMA'] = calculate_ema(df['close'], ema_len)
    df['below_EMA'] = df['close'] < df['EMA']
    df['below_count'] = df['below_EMA'].rolling(window=threshold, min_periods=1).sum()
    df['signal'] = df['below_count'] == threshold

    buy_idx = df.index[df['signal']]
    trades = []
    for idx in buy_idx:
        # 买入当天收盘，次日开盘卖出（示例，实际可调整为你的平仓逻辑）
        if idx + 1 < len(df):
            buy_price = df.loc[idx, 'close']
            sell_price = df.loc[idx + 1, 'open']
            ret = (sell_price - buy_price) / buy_price
            trades.append(ret)
    if not trades:
        win_rate = 0
        max_drawdown = 0
        trades_count = 0
        total_return = 0
    else:
        win_rate = sum([t > 0 for t in trades]) / len(trades)
        max_drawdown = min(trades) if trades else 0
        trades_count = len(trades)
        total_return = (1 + pd.Series(trades)).prod() - 1

    # 前端表格只显示5列
    return {
        "股票代码": symbol,
        "总盈亏率": f"{total_return*100:.2f}%",
        "最大回撤率": f"{max_drawdown*100:.2f}%",
        "胜率": f"{win_rate*100:.2f}%",
        "总交易次数": trades_count,
        # 详细字段仅CSV输出
        "每笔收益率": trades
    }

def load_all_data():
    stocks = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(DATA_DIR, fname), "r") as f:
                stocks += [line.strip().upper() for line in f.readlines() if line.strip()]
    return list(set(stocks))

# --- Streamlit界面 ---
tab1, tab2, tab3 = st.tabs(["股票池与数据下载", "今日选股信号", "批量回测"])

with tab1:
    st.markdown("### 1. 股票池管理 & 批量数据下载")
    uploadedfile = st.file_uploader("上传股票代码txt（每行一个代码）", type=["txt"])
    if uploadedfile:
        save_uploaded_file(uploadedfile)
        st.success(f"导入股票数量：{sum(1 for _ in open(os.path.join(DATA_DIR, uploadedfile.name)))}")
    stocks = load_all_data()
    if stocks:
        st.write(f"当前股票池：{'、'.join(stocks)}")

with tab2:
    st.markdown("### 2. 今日选股信号")
    stocks = load_all_data()
    if not stocks:
        st.warning("请先完成数据下载")
    else:
        picked = []
        for s in stocks:
            df = get_stock_data(s)
            if df is not None and len(df) > 10 and stock_signal_last_row(df):
                picked.append(s)
        if picked:
            st.success(f"今日可买入股票：{'、'.join(picked)}")
            csv = "\n".join(picked)
            st.download_button("下载今日选股txt", csv, file_name=f"今日选股_{datetime.now().date()}.txt")
        else:
            st.info("无符合买入信号的股票")

with tab3:
    st.markdown("### 3. 批量回测")
    stocks = load_all_data()
    if not stocks:
        st.warning("请先完成数据下载")
    else:
        st.write("点击表头可排序，导出CSV同表格排序一致。")
        results = []
        for s in stocks:
            df = get_stock_data(s)
            if df is not None and len(df) > 10:
                result = backtest_single(s, df)
                results.append(result)
        if results:
            df_result = pd.DataFrame(results)
            # 只显示5列
            show_cols = ["股票代码", "总盈亏率", "最大回撤率", "胜率", "总交易次数"]
            st.dataframe(df_result[show_cols].sort_values(by="总盈亏率", ascending=False), use_container_width=True)
            # 下载全部数据
            full_csv = df_result.to_csv(index=False, encoding="utf_8_sig")
            b64 = base64.b64encode(full_csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="回测详情_{datetime.now().date()}.csv">下载回测结果csv</a>', unsafe_allow_html=True)