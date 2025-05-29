import os
import pandas as pd
import streamlit as st
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
RESULT_DIR = "result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

st.set_page_config("SIXQUARE AI选股", layout="wide")

# 1. 读取最新日期
def check_latest_dates():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    date_dict = {}
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        if 'Date' in df.columns and not df.empty:
            date_dict[f.replace('.csv', '')] = df['Date'].iloc[-1]
    return date_dict

# 2. 批量数据下载（略，接口调用建议写到专用py或Notebook）

# 3. 信号与回测策略
def today_signal(symbols, ema_length=5, threshold=3):
    """完全对齐TradingView策略，仅当今日刚好首次达到阈值才入选"""
    buy_list = []
    for code in symbols:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{code}.csv"))
            if df.empty or len(df) < ema_length + threshold:
                continue
            df['EMA'] = df['Close'].ewm(span=ema_length, adjust=False).mean()
            below_count = 0
            for i in range(1, len(df)):
                if df.loc[i, 'Close'] < df.loc[i, 'EMA']:
                    below_count = below_count + 1 if below_count else 1
                else:
                    below_count = 0
                # 只在今天（最后一根）首次满足
                if below_count == threshold and i == len(df)-1:
                    buy_list.append(code)
        except Exception:
            continue
    return buy_list

def backtest_single(code, start_date, end_date, ema_length=5, threshold=3, initial_cash=10000):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{code}.csv"))
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)
        if df.empty or len(df) < ema_length + threshold:
            return None
        df['EMA'] = df['Close'].ewm(span=ema_length, adjust=False).mean()
        below_count = 0
        position = 0
        entry_price = 0
        cash = initial_cash
        equity = initial_cash
        max_drawdown = 0
        max_equity = initial_cash
        trades = []
        for i in range(1, len(df)):
            if df.loc[i, 'Close'] < df.loc[i, 'EMA']:
                below_count = below_count + 1 if below_count else 1
            else:
                below_count = 0
            # 开仓（与TV一致，只在首次连续n根当天开多）
            if position == 0 and below_count == threshold:
                position = cash / df.loc[i, 'Close']
                entry_price = df.loc[i, 'Close']
                cash = 0
                trades.append({'date': df.loc[i, 'Date'], 'type': 'buy', 'price': entry_price})
            # 平仓
            if position > 0 and df.loc[i, 'Close'] > df.loc[i-1, 'High']:
                exit_price = df.loc[i, 'Close']
                cash = position * exit_price
                trades.append({'date': df.loc[i, 'Date'], 'type': 'sell', 'price': exit_price})
                position = 0
                entry_price = 0
            # 每日动态权益
            equity = cash if position == 0 else position * df.loc[i, 'Close']
            if equity > max_equity:
                max_equity = equity
            dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
        # 收官强平
        if position > 0:
            cash = position * df.loc[len(df)-1, 'Close']
            trades.append({'date': df.loc[len(df)-1, 'Date'], 'type': 'sell_end', 'price': df.loc[len(df)-1, 'Close']})
        total_profit = cash - initial_cash
        total_profit_pct = total_profit / initial_cash * 100
        max_dd_amt = max_equity * max_drawdown
        max_dd_pct = max_drawdown * 100
        # 统计胜率
        profit_trades = [t for idx, t in enumerate(trades) if t['type'] == 'sell' or t['type'] == 'sell_end']
        total_trades = len(profit_trades)
        profit_count = 0
        loss_count = 0
        for idx, t in enumerate(profit_trades):
            if idx == 0:
                continue
            buy_px = trades[2*idx-2]['price'] if 2*idx-2 < len(trades) else None
            sell_px = t['price']
            if buy_px is not None:
                if sell_px > buy_px:
                    profit_count += 1
                else:
                    loss_count += 1
        win_rate = f"{(profit_count/total_trades*100):.2f}%" if total_trades > 0 else "0.00%"
        return {
            '股票代码': code,
            '总盈亏': f"{total_profit:.2f}",
            '总盈亏率': f"{total_profit_pct:.2f}%",
            '最大回撤': f"{max_dd_amt:.2f}",
            '最大回撤率': f"{max_dd_pct:.2f}%",
            '总交易数': total_trades,
            '盈利次数': profit_count,
            '亏损次数': loss_count,
            '胜率': win_rate,
            '初始资金': initial_cash,
        }
    except Exception:
        return None

def batch_backtest(symbols, start_date, end_date, ema_length=5, threshold=3, initial_cash=10000):
    results = []
    for code in symbols:
        res = backtest_single(code, start_date, end_date, ema_length, threshold, initial_cash)
        if res:
            results.append(res)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('总盈亏率', ascending=False)
    return df

# --- UI部分 ---
st.markdown("# SIXQUARE AI选股")
st.markdown("###### by SIXQUARE")

tabs = st.tabs(["📄股票池与数据下载", "📈今日选股信号", "📊批量回测"])

# --- 股票池与数据下载
with tabs[0]:
    st.header("1. 股票池管理 & 批量数据下载")
    st.info("上传股票代码txt（每行一个代码），可管理你的股票池并一键批量下载美股数据到服务器本地（txt格式, 只需一次，后续数据会自动增量更新）")
    uploaded = st.file_uploader("上传股票代码txt", type=['txt'])
    if uploaded:
        codes = [line.strip().upper() for line in uploaded.read().decode('utf-8').splitlines() if line.strip()]
        with open("stocks.txt", "w", encoding='utf-8') as f:
            for c in codes:
                f.write(f"{c}\n")
        st.success(f"已导入股票数量: {len(codes)}")
    else:
        if os.path.exists("stocks.txt"):
            with open("stocks.txt") as f:
                codes = [line.strip().upper() for line in f if line.strip()]
        else:
            codes = []
        st.write(f"已导入股票数量: {len(codes)}")
    if st.button("一键下载最新日K数据"):
        # 这里省略批量数据拉取逻辑。建议用yfinance/akshare批量下载
        st.success("（演示）日K数据已批量更新！")
    date_dict = check_latest_dates()
    if date_dict:
        maxdate = max(date_dict.values())
        st.markdown(f"**当前已下载股票及其数据最新日期：{maxdate}**")
        st.dataframe(pd.DataFrame({"代码": list(date_dict.keys()), "最新日期": list(date_dict.values())}))
    else:
        st.info("还没有任何股票数据，请先上传股票池并下载数据。")

# --- 今日选股信号
with tabs[1]:
    st.header("2. 今日选股信号")
    date_dict = check_latest_dates()
    maxdate = max(date_dict.values()) if date_dict else "-"
    st.markdown(f"**当前后台数据最新日期： {maxdate}**")
    st.write(f"当前股票池数量：{len(codes)}")
    if 'show_debug_signal' not in st.session_state:
        st.session_state['show_debug_signal'] = False
    if not st.session_state['show_debug_signal']:
        with st.form("signal_debug_form"):
            pwd = st.text_input("请输入调试密码", type='password', key='signal_pwd')
            debug_btn = st.form_submit_button("显示调试参数")
            if debug_btn:
                if pwd == "1118518":
                    st.session_state['show_debug_signal'] = True
                elif pwd != "":
                    st.error("密码错误")
        ema_length = 5
        threshold = 3
    else:
        ema_length = st.number_input("EMA长度", 1, 30, 5, key='ema_input1')
        threshold = st.number_input("连续低于EMA根数", 1, 10, 3, key='th_input1')
    if st.button("执行今日选股信号筛选"):
        buy_list = today_signal(codes, ema_length=ema_length, threshold=threshold)
        st.session_state['buy_list_today'] = buy_list
    buy_list_today = st.session_state.get('buy_list_today', [])
    if buy_list_today:
        st.success("今日可买入股票：" + "、".join(buy_list_today))
        buy_df = pd.DataFrame({"买入信号股票": buy_list_today})
        AgGrid(buy_df, fit_columns_on_grid_load=True)
        st.download_button("下载txt", "\n".join(buy_list_today), file_name=f"buy_signal_{datetime.now().strftime('%Y%m%d')}.txt")
    else:
        st.info("无可买入信号或未筛选。")

# --- 批量回测
with tabs[2]:
    st.header("3. 批量回测")
    date_dict = check_latest_dates()
    maxdate = max(date_dict.values()) if date_dict else "-"
    st.markdown(f"**当前后台数据最新日期： {maxdate}**")
    if 'show_debug_backtest' not in st.session_state:
        st.session_state['show_debug_backtest'] = False
    if not st.session_state['show_debug_backtest']:
        with st.form("backtest_debug_form"):
            pwd = st.text_input("请输入调试密码", type='password', key='backtest_pwd')
            debug_btn = st.form_submit_button("显示调试参数")
            if debug_btn:
                if pwd == "1118518":
                    st.session_state['show_debug_backtest'] = True
                elif pwd != "":
                    st.error("密码错误")
        ema_length = 5
        threshold = 3
    else:
        ema_length = st.number_input("EMA长度", 1, 30, 5, key='ema_input2')
        threshold = st.number_input("连续低于EMA根数", 1, 10, 3, key='th_input2')
    st.markdown("自定义回测区间，格式如2024-01-01（直接回车用默认）：")
    start_date = st.text_input("回测起始日期", value="2024-01-01", key='bt_start')
    end_date = st.text_input("回测结束日期", value="2025-05-01", key='bt_end')
    # 是否回测今日信号
    use_today = st.checkbox("仅回测今日选股信号", value=False)
    codes_for_bt = st.session_state.get('buy_list_today', []) if use_today else codes
    if st.button("批量回测"):
        bt_df = batch_backtest(codes_for_bt, start_date, end_date, ema_length, threshold)
        st.session_state['bt_df'] = bt_df
    bt_df = st.session_state.get('bt_df', None)
    if isinstance(bt_df, pd.DataFrame) and not bt_df.empty:
        gb = GridOptionsBuilder.from_dataframe(bt_df)
        gb.configure_pagination(paginationAutoPageSize=False)
        gb.configure_default_column(editable=False, groupable=True)
        gb.configure_side_bar()
        gb.configure_column('总盈亏率', type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
        grid_options = gb.build()
        AgGrid(bt_df, gridOptions=grid_options, enable_enterprise_modules=True, fit_columns_on_grid_load=True)
        st.download_button("下载回测结果csv", bt_df.to_csv(index=False, encoding='utf-8-sig'), file_name="回测结果.csv")
    else:
        st.info("无回测结果，点击按钮批量回测。")