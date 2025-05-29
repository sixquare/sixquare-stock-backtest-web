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

def clear_data_dir():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

def batch_download(symbols, data_dir=DATA_DIR):
    clear_data_dir()
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

def get_data_symbols():
    if not os.path.exists(DATA_DIR):
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    codes = [f.replace('.csv', '') for f in files]
    return codes

def check_latest_dates():
    code_dates = {}
    if not os.path.exists(DATA_DIR):
        return code_dates
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            code = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(DATA_DIR, file))
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

def batch_backtest(symbols, start_date, end_date, initial_capital=10000, ema_length=5, threshold=3):
    results = []
    for code in symbols:
        try:
            fpath = os.path.join(DATA_DIR, f"{code}.csv")
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

st.set_page_config(page_title="美股批量选股 & 回测 Web 工具", layout="wide")
st.title("SIXQUARE股市工具")

tabs = st.tabs(["📥 股票池与数据下载", "📊 今日选股信号", "📈 批量回测"])

with tabs[0]:
    st.header("1. 股票池管理 & 批量数据下载")
    stock_txt = st.file_uploader("上传股票代码txt（每行一个代码）", type=['txt'])
    if stock_txt:
        symbols = [line.decode('utf-8').strip().upper() for line in stock_txt if line.strip()]
        st.write(f"已导入股票数量: {len(symbols)}")
        if st.button("一键下载最新日K数据"):
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

with tabs[1]:
    st.header("2. 今日选股信号")
    code_dates = check_latest_dates()
    if code_dates:
        all_dates = list(code_dates.values())
        all_dates = [str(d) for d in all_dates if d]
        max_date = max(all_dates) if all_dates else "暂无数据"
        st.info(f"当前后台数据最新日期：{max_date}")
    else:
        st.info("当前暂无已下载数据，请先上传股票池并下载。")
    symbols = sorted(list(code_dates.keys()))
    st.write(f"当前股票池数量：{len(symbols)}")

    if 'debug1_unlock' not in st.session_state:
        st.session_state['debug1_unlock'] = False
    if 'debug1_ask' not in st.session_state:
        st.session_state['debug1_ask'] = False

    if not st.session_state['debug1_unlock']:
        if not st.session_state['debug1_ask']:
            if st.button("显示调试参数", key='show_debug1'):
                st.session_state['debug1_ask'] = True
        if st.session_state['debug1_ask']:
            pwd = st.text_input("请输入调试密码", type="password", key="pwd_input1")
            if st.button("确认密码", key="pwd_btn1"):
                if pwd == "1118518":
                    st.success("密码正确，已解锁参数设置！")
                    st.session_state['debug1_unlock'] = True
                elif pwd:
                    st.error("密码错误，请重试")
    if st.session_state['debug1_unlock']:
        ema_length = st.number_input("EMA长度", 1, 30, 5, key='ema_input1')
        threshold = st.number_input("连续低于EMA根数", 1, 10, 3, key='th_input1')
    else:
        ema_length = 5
        threshold = 3

    if st.button("执行今日选股信号筛选"):
        buy_list = today_signal(symbols, ema_length, threshold)
        st.success(f"今日可买入股票：{', '.join(buy_list) if buy_list else '无'}")
        if buy_list:
            # 排序按原symbols顺序
            ordered_buy_list = [code for code in symbols if code in buy_list]
            st.write(pd.DataFrame({'买入信号股票': ordered_buy_list}))
            # CSV下载按钮
            st.download_button('下载csv',
                               pd.DataFrame({'买入信号股票': ordered_buy_list}).to_csv(index=False).encode('utf-8'),
                               'today_buy_signal.csv')
            # TXT下载按钮（每行一个代码，顺序和stocks.txt一致）
            st.download_button(
                '下载txt(原顺序)',
                "\n".join(ordered_buy_list).encode('utf-8'),
                'today_buy_signal.txt'
            )

with tabs[2]:
    st.header("3. 批量回测")
    code_dates = check_latest_dates()
    if code_dates:
        all_dates = list(code_dates.values())
        all_dates = [str(d) for d in all_dates if d]
        max_date = max(all_dates) if all_dates else "暂无数据"
        st.info(f"当前后台数据最新日期：{max_date}")
    else:
        st.info("当前暂无已下载数据，请先上传股票池并下载。")
    symbols = sorted(list(code_dates.keys()))
    st.write(f"当前股票池数量：{len(symbols)}")

    if 'backtest_df' not in st.session_state:
        st.session_state['backtest_df'] = None

    if 'debug2_unlock' not in st.session_state:
        st.session_state['debug2_unlock'] = False
    if 'debug2_ask' not in st.session_state:
        st.session_state['debug2_ask'] = False

    if not st.session_state['debug2_unlock']:
        if not st.session_state['debug2_ask']:
            if st.button("显示调试参数", key='show_debug2'):
                st.session_state['debug2_ask'] = True
        if st.session_state['debug2_ask']:
            pwd2 = st.text_input("请输入调试密码", type="password", key="pwd_input2")
            if st.button("确认密码", key="pwd_btn2"):
                if pwd2 == "1118518":
                    st.success("密码正确，已解锁参数设置！")
                    st.session_state['debug2_unlock'] = True
                elif pwd2:
                    st.error("密码错误，请重试")
    if st.session_state['debug2_unlock']:
        ema_length3 = st.number_input("EMA长度", 1, 30, 5, key='ema_input2')
        threshold3 = st.number_input("连续低于EMA根数", 1, 10, 3, key='th_input2')
    else:
        ema_length3 = 5
        threshold3 = 3

    start_date = st.date_input("回测起始日期", datetime(2024,1,1))
    end_date = st.date_input("回测结束日期", datetime(2025,5,1))
    if st.button("执行批量回测"):
        dfres = batch_backtest(symbols, str(start_date), str(end_date), ema_length=ema_length3, threshold=threshold3)
        if not dfres.empty:
            dfres['总盈亏率数值'] = to_percent_float(dfres['总盈亏率'])
            dfres['最大回撤率数值'] = to_percent_float(dfres['最大回撤率'])
            dfres['胜率数值'] = to_percent_float(dfres['胜率'])
            columns = ["股票代码", "总盈亏", "总盈亏率", "最大回撤", "最大回撤率", "总交易数", "盈利次数", "亏损次数", "胜率", "初始资金"]
            st.session_state['backtest_df'] = dfres[columns + ['总盈亏率数值','最大回撤率数值','胜率数值']]

    if st.session_state['backtest_df'] is not None and not st.session_state['backtest_df'].empty:
        columns = ["股票代码", "总盈亏", "总盈亏率", "最大回撤", "最大回撤率", "总交易数", "盈利次数", "亏损次数", "胜率", "初始资金"]
        gb = GridOptionsBuilder.from_dataframe(st.session_state['backtest_df'])
        gb.configure_column("总盈亏率", type=["numericColumn"], valueGetter="Number(data.总盈亏率.replace('%',''))")
        gb.configure_column("最大回撤率", type=["numericColumn"], valueGetter="Number(data.最大回撤率.replace('%',''))")
        gb.configure_column("胜率", type=["numericColumn"], valueGetter="Number(data.胜率.replace('%',''))")
        gb.configure_column("总盈亏率数值", hide=True)
        gb.configure_column("最大回撤率数值", hide=True)
        gb.configure_column("胜率数值", hide=True)
        gridOptions = gb.build()
        st.write("点击表头即可按数值排序，导出CSV同表格排序一致。")
        ag_ret = AgGrid(st.session_state['backtest_df'], gridOptions=gridOptions, fit_columns_on_grid_load=True, height=500, return_mode='AS_INPUT')
        download_df = pd.DataFrame(ag_ret['data'])[columns]
        st.download_button('下载回测结果csv', download_df.to_csv(index=False).encode('utf-8'), 'batch_backtest.csv')
    else:
        st.write("无回测结果")