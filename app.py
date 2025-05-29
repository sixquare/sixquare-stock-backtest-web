import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import akshare as ak
import time
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder

DATA_DIR = "data"
TODAY_SIGNAL_FILE = "today_buy_signal.txt"

def clear_data_dir():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

def batch_download(symbols, data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    status = []
    date_map = {}
    for code in symbols:
        try:
            df = ak.stock_us_daily(symbol=code)
            if not df.empty:
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close'
                })
                df = df[['Date', 'Open', 'High', 'Low', 'Close']]
                df.to_csv(f"{data_dir}/{code}.csv", index=False)
                latest_date = df['Date'].iloc[-1]
                status.append((code, "成功", latest_date))
                date_map[code] = latest_date
            else:
                status.append((code, "无数据", ""))
            time.sleep(0.2)
        except Exception as e:
            status.append((code, f"失败：{e}", ""))
    return status, date_map

def check_latest_dates(data_dir=DATA_DIR):
    code_dates = {}
    if not os.path.exists(data_dir):
        return code_dates
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            code = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, file))
            if not df.empty:
                code_dates[code] = df['Date'].iloc[-1]
    return code_dates

def today_signal(symbols, ema_length=5, threshold=3):
    buy_list = []
    for code in symbols:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{code}.csv"))
            if df.empty or len(df) < ema_length + threshold:
                continue
            df = df[-(ema_length + threshold + 2):].reset_index(drop=True)
            df['EMA'] = df['Close'].ewm(span=ema_length, adjust=False).mean()
            below_ema = df['Close'] < df['EMA']
            if all(below_ema.iloc[-threshold:]):
                buy_list.append(code)
        except Exception:
            continue
    return buy_list

def calc_max_drawdown(equity_curve):
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - cummax
    max_drawdown = drawdown.min()
    max_drawdown_rate = max_drawdown / cummax[np.argmin(drawdown)] if cummax[np.argmin(drawdown)] != 0 else 0
    return abs(max_drawdown), abs(max_drawdown_rate)

def batch_backtest(symbols, start_date, end_date, initial_capital=10000, ema_length=5, threshold=3, data_dir=DATA_DIR):
    results = []
    for code in symbols:
        try:
            fpath = os.path.join(data_dir, f"{code}.csv")
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
            if len(df) < ema_length + threshold:
                continue
            df['EMA'] = df['Close'].ewm(span=ema_length, adjust=False).mean()
            below_count = 0
            pos = None
            trades = []
            equity = initial_capital
            equity_curve = [equity]
            for i in range(1, len(df)):
                if df.iloc[i]['Close'] < df.iloc[i]['EMA']:
                    below_count += 1
                else:
                    below_count = 0
                if pos is None and below_count >= threshold:
                    size = equity / df.iloc[i]['Close']
                    entry_price = df.iloc[i]['Close']
                    pos = {'size': size, 'entry_price': entry_price}
                    equity = 0
                if pos is not None and df.iloc[i]['Close'] > df.iloc[i-1]['High']:
                    exit_price = df.iloc[i]['Close']
                    pnl = pos['size'] * (exit_price - pos['entry_price'])
                    equity = pos['size'] * exit_price
                    trades.append({'pnl': pnl})
                    pos = None
                equity_curve.append(equity if pos is None else pos['size'] * df.iloc[i]['Close'])
            if pos is not None:
                last_exit_price = df.iloc[-1]['Close']
                pnl = pos['size'] * (last_exit_price - pos['entry_price'])
                equity = pos['size'] * last_exit_price
                trades.append({'pnl': pnl})
                equity_curve[-1] = equity
            pnl_sum = sum([x['pnl'] for x in trades])
            win = sum([1 for x in trades if x['pnl'] > 0])
            lose = sum([1 for x in trades if x['pnl'] <= 0])
            total_trade = win + lose
            winrate = win / total_trade if total_trade > 0 else 0
            final_equity = equity_curve[-1]
            total_return_rate = (final_equity - initial_capital) / initial_capital if initial_capital else 0
            max_dd, max_dd_rate = calc_max_drawdown(np.array(equity_curve))
            results.append({
                "股票代码": code,
                "总盈亏": round(final_equity - initial_capital, 2),
                "总盈亏率": f"{total_return_rate*100:.2f}%",
                "最大回撤": round(max_dd, 2),
                "最大回撤率": f"{max_dd_rate*100:.2f}%",
                "总交易数": total_trade,
                "盈利次数": win,
                "亏损次数": lose,
                "胜率": f"{winrate*100:.2f}%",
                "初始资金": initial_capital,
            })
        except Exception as e:
            continue
    columns = ["股票代码", "总盈亏", "总盈亏率", "最大回撤", "最大回撤率", "总交易数", "盈利次数", "亏损次数", "胜率", "初始资金"]
    return pd.DataFrame(results)[columns]

def to_percent_float(series):
    return series.str.rstrip('%').astype(float)

def get_today_signal_symbols():
    if os.path.exists(TODAY_SIGNAL_FILE):
        with open(TODAY_SIGNAL_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []

st.set_page_config(page_title="SIXQUARE选股AI工具", layout="wide")
st.title("SIXQUARE选股AI工具")

tabs = st.tabs(["📥 股票池与数据下载", "📊 今日选股信号", "📈 批量回测"])

# TAB1 (不变)
with tabs[0]:
    st.header("1. 股票池管理 & 批量数据下载")
    stock_txt = st.file_uploader("上传股票代码txt（每行一个代码）", type=['txt'])
    if stock_txt:
        symbols = [line.decode('utf-8').strip().upper() for line in stock_txt if line.strip()]
        st.write(f"已导入股票数量: {len(symbols)}")
        if st.button("一键下载最新日K数据"):
            clear_data_dir()
            status, date_map = batch_download(symbols)
            st.success("下载完毕！")
            dfres = pd.DataFrame(status, columns=['代码', '状态', '最新日期'])
            st.write(dfres)
            st.info(f"已保存数据至 {DATA_DIR}/，后续操作会自动读取此目录。")
    st.subheader("当前已下载股票及其数据最新日期：")
    code_dates = check_latest_dates()
    if code_dates:
        st.write(pd.DataFrame(list(code_dates.items()), columns=['股票代码', '最新数据日期']))
    else:
        st.write("暂无已下载数据，请先上传股票池并下载。")

# TAB2 (不变)
with tabs[1]:
    st.header("2. 今日选股与卖出信号")
    code_dates = check_latest_dates()
    symbols = sorted(list(code_dates.keys()))
    st.write(f"当前股票池数量：{len(symbols)}")
    if code_dates:
        all_dates = list(code_dates.values())
        max_date = max([str(d) for d in all_dates if d]) if all_dates else "暂无数据"
        st.info(f"当前后台数据最新日期：{max_date}")
    else:
        st.info("当前暂无已下载数据，请先上传股票池并下载。")
    if 'show_debug_signal' not in st.session_state:
        st.session_state['show_debug_signal'] = False
    if not st.session_state['show_debug_signal']:
        with st.form("signal_debug_form"):
            pwd = st.text_input("请输入调试密码", type='password', key='signal_pwd')
            debug_btn = st.form_submit_button("显示调试参数")
            if debug_btn and pwd == "1118518":
                st.session_state['show_debug_signal'] = True
            elif debug_btn and pwd != "":
                st.error("密码错误")
        ema_length = 5
        threshold = 3
    else:
        ema_length = st.number_input("EMA长度", 1, 30, 5, key='ema_input1')
        threshold = st.number_input("连续低于EMA根数", 1, 10, 3, key='th_input1')

    if st.button("执行今日买卖信号筛选"):
        buy_list = []
        sell_list = []
        buy_dates = []
        sell_dates = []

        for code in symbols:
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, f"{code}.csv"))
                if df.empty or len(df) < ema_length + threshold + 2:
                    continue
                df['EMA'] = df['Close'].ewm(span=ema_length, adjust=False).mean()
                # --- 买入信号 ---
                below_ema = df['Close'] < df['EMA']
                for i in range(threshold, len(df)):
                    # 连续threshold根低于EMA，且此K线是最后一根K线（即今日信号）
                    if all(below_ema.iloc[i-threshold+1:i+1]) and i == len(df)-1:
                        buy_list.append(code)
                        buy_dates.append(df['Date'].iloc[i])
                        break
                # --- 卖出信号（符合平仓条件：收盘价 > 昨日high） ---
                for i in range(1, len(df)):
                    if df['Close'].iloc[i] > df['High'].iloc[i-1] and i == len(df)-1:
                        sell_list.append(code)
                        sell_dates.append(df['Date'].iloc[i])
                        break
            except Exception:
                continue

        # ---- 买入信号展示 ----
        if buy_list:
            st.success(f"今日出现买入信号的股票（应在【次一交易日开盘】市价买入）:")
            buy_df = pd.DataFrame({'股票代码': buy_list, '信号日期': buy_dates})
            st.dataframe(buy_df, use_container_width=True)
            st.write("👉 建议在第二天开盘（美股9:30AM）以市价买入上表股票")
            st.download_button('下载今日买入信号csv', buy_df.to_csv(index=False).encode('utf-8'), 'today_buy_signal.csv')
            st.download_button('下载今日买入信号txt', "\n".join(buy_list).encode('utf-8'), 'today_buy_signal.txt')
            with open(TODAY_SIGNAL_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(buy_list))
        else:
            st.info("今日无买入信号")

        # ---- 卖出信号展示 ----
        if sell_list:
            st.error(f"今日出现平仓信号的股票（应在【次一交易日开盘】市价卖出）:")
            sell_df = pd.DataFrame({'股票代码': sell_list, '信号日期': sell_dates})
            st.dataframe(sell_df, use_container_width=True)
            st.write("👉 建议在第二天开盘（美股9:30AM）以市价卖出上表股票")
            st.download_button('下载今日卖出信号csv', sell_df.to_csv(index=False).encode('utf-8'), 'today_sell_signal.csv')
            st.download_button('下载今日卖出信号txt', "\n".join(sell_list).encode('utf-8'), 'today_sell_signal.txt')
        else:
            st.info("今日无卖出信号")


# TAB3 - 回测表格仅显示简要5列，下载csv为全列
with tabs[2]:
    st.header("3. 批量回测")
    code_dates = check_latest_dates()
    symbols = sorted(list(code_dates.keys()))
    st.write(f"当前股票池数量：{len(symbols)}")
    if code_dates:
        all_dates = list(code_dates.values())
        max_date = max([str(d) for d in all_dates if d]) if all_dates else "暂无数据"
        st.info(f"当前后台数据最新日期：{max_date}")
    else:
        st.info("当前暂无已下载数据，请先上传股票池并下载。")
    today_signal_exists = os.path.exists(TODAY_SIGNAL_FILE)
    stock_list_option = "全部股票"
    if today_signal_exists:
        stock_list_option = st.radio("回测股票池来源", ["全部股票", "今日选股信号"], horizontal=True)
    else:
        st.info("如需回测今日选股信号，请先在【今日选股信号】执行一次选股。")
    if stock_list_option == "今日选股信号" and today_signal_exists:
        symbols_to_bt = get_today_signal_symbols()
    else:
        symbols_to_bt = symbols
    if 'backtest_df' not in st.session_state:
        st.session_state['backtest_df'] = None
    if 'show_debug_backtest' not in st.session_state:
        st.session_state['show_debug_backtest'] = False
    if not st.session_state['show_debug_backtest']:
        with st.form("backtest_debug_form"):
            pwd = st.text_input("请输入调试密码", type='password', key='backtest_pwd')
            debug_btn = st.form_submit_button("显示调试参数")
            if debug_btn and pwd == "1118518":
                st.session_state['show_debug_backtest'] = True
            elif debug_btn and pwd != "":
                st.error("密码错误")
        ema_length3 = 5
        threshold3 = 3
    else:
        ema_length3 = st.number_input("回测EMA长度", 1, 30, 5, key='ema_input2')
        threshold3 = st.number_input("回测连续低于EMA根数", 1, 10, 3, key='th_input2')
    start_date = st.date_input("回测起始日期", datetime(2024,1,1))
    end_date = st.date_input("回测结束日期", datetime(2025,5,1))
    if st.button("执行批量回测"):
        dfres = batch_backtest(symbols_to_bt, str(start_date), str(end_date), ema_length=ema_length3, threshold=threshold3)
        if not dfres.empty:
            dfres['总盈亏率数值'] = to_percent_float(dfres['总盈亏率'])
            dfres['最大回撤率数值'] = to_percent_float(dfres['最大回撤率'])
            dfres['胜率数值'] = to_percent_float(dfres['胜率'])
            st.session_state['backtest_df'] = dfres
    # 仅展示简要5列
    display_cols = ["股票代码", "总盈亏率", "最大回撤率", "胜率", "总交易数"]
    all_cols = ["股票代码", "总盈亏", "总盈亏率", "最大回撤", "最大回撤率", "总交易数", "盈利次数", "亏损次数", "胜率", "初始资金"]
    if st.session_state['backtest_df'] is not None and not st.session_state['backtest_df'].empty:
        gb = GridOptionsBuilder.from_dataframe(st.session_state['backtest_df'][display_cols])
        gb.configure_column("总盈亏率", type=["numericColumn"], valueGetter="Number(data.总盈亏率.replace('%',''))")
        gb.configure_column("最大回撤率", type=["numericColumn"], valueGetter="Number(data.最大回撤率.replace('%',''))")
        gb.configure_column("胜率", type=["numericColumn"], valueGetter="Number(data.胜率.replace('%',''))")
        gridOptions = gb.build()
        st.write("点击表头即可按数值排序，导出CSV同表格排序一致。")
        ag_ret = AgGrid(st.session_state['backtest_df'][display_cols], gridOptions=gridOptions, fit_columns_on_grid_load=True, height=500, return_mode='AS_INPUT')
        # 下载按钮用全字段
        st.download_button('下载回测结果csv', st.session_state['backtest_df'][all_cols].to_csv(index=False).encode('utf-8'), 'batch_backtest.csv')
    else:
        st.write("无回测结果")