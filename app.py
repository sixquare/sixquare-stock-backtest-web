import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import akshare as ak

st.set_page_config(layout="wide")

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def is_china_stock(code):
    # 6位全数字视为A股
    return code.isdigit() and len(code) == 6

def get_stock_data(symbol, start_date, end_date):
    if is_china_stock(symbol):
        # A股
        try:
            df = ak.stock_zh_a_hist(symbol, period="daily", start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''), adjust="")
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df = df[["date", "close"]]
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except:
            return None
    else:
        # 美股
        try:
            df = ak.stock_us_daily(symbol)
            df = df.rename(columns={"date": "date", "close": "close"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
            df = df[df["date"] >= start_date]
            df = df[df["date"] <= end_date]
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except:
            return None

def calculate_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def get_signals(df, ema_len=5, threshold=3):
    # 用于生成买卖信号（和TradingView一致）
    df = df.copy()
    df['ema'] = calculate_ema(df['close'], ema_len)
    df['above_count'] = 0
    df['below_count'] = 0
    above = 0
    below = 0
    for i in range(len(df)):
        if df.loc[i, 'close'] > df.loc[i, 'ema']:
            above += 1
            below = 0
        elif df.loc[i, 'close'] < df.loc[i, 'ema']:
            below += 1
            above = 0
        else:
            above = 0
            below = 0
        df.at[i, 'above_count'] = above
        df.at[i, 'below_count'] = below

    df['buy_signal'] = ((df['below_count'] >= threshold))
    df['sell_signal'] = (df['close'] > df['close'].shift(1)) & (df['buy_signal'].shift(1))
    return df

def filter_signals(today, buy_df, sell_df):
    """当天有卖出信号就不出买入信号，优先保留卖出"""
    buy_df = buy_df.copy()
    sell_df = sell_df.copy()
    # 只保留今天的买卖信号
    buy_df = buy_df[buy_df['信号日期'] == today]
    sell_df = sell_df[sell_df['信号日期'] == today]
    # 剔除同一只股票当天既有买又有卖
    buy_codes = set(buy_df['股票代码'])
    sell_codes = set(sell_df['股票代码'])
    overlap = buy_codes & sell_codes
    buy_df = buy_df[~buy_df['股票代码'].isin(overlap)]
    # 卖出信号保留
    return buy_df, sell_df

st.title('SIXQUARE AI选股')

tab1, tab2, tab3 = st.tabs(['股票池与数据下载', '今日选股信号', '批量回测'])

with tab1:
    st.header('1. 股票池管理 & 批量数据下载')
    uploaded = st.file_uploader('上传股票代码txt（每行一个代码）', type='txt')
    if uploaded is not None:
        codes = [i.strip() for i in uploaded.read().decode('utf-8').split('\n') if i.strip()]
        st.success(f'导入股票数量：{len(codes)}')
        st.write('当前股票池:', ', '.join(codes))
        with st.spinner('正在下载数据...'):
            for code in codes:
                for market in ['us', 'cn']:
                    file_path = f'{DATA_DIR}/{code}.csv'
                    if os.path.exists(file_path):
                        continue
                    # 美股或A股自动判断
                    df = get_stock_data(code, '2020-01-01', datetime.now().strftime('%Y-%m-%d'))
                    if df is not None and not df.empty:
                        df.to_csv(file_path, index=False)
    st.write(" ")

with tab2:
    st.header('2. 今日选股信号')
    # 读取股票池
    pool = [f.split('.csv')[0] for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    st.write(f'当前股票池数量: {len(pool)}')
    today_str = datetime.now().strftime('%Y-%m-%d')
    st.info(f'当前后台数据最新日期：{today_str}')
    # 策略参数
    ema_len = 5
    threshold = 3
    today_buy_signals = []
    today_sell_signals = []
    for code in pool:
        file_path = f'{DATA_DIR}/{code}.csv'
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        if len(df) < ema_len + threshold + 2:
            continue
        df_sig = get_signals(df, ema_len=ema_len, threshold=threshold)
        # 买入信号（当天）
        buy_idx = df_sig.index[df_sig['buy_signal']].tolist()
        if buy_idx:
            for idx in buy_idx:
                if idx + 1 < len(df_sig):
                    date = df_sig.loc[idx, 'date']
                    if date == today_str:
                        today_buy_signals.append({'股票代码': code, '信号日期': date})
        # 卖出信号（当天）
        sell_idx = df_sig.index[df_sig['sell_signal']].tolist()
        if sell_idx:
            for idx in sell_idx:
                if idx + 1 < len(df_sig):
                    date = df_sig.loc[idx, 'date']
                    if date == today_str:
                        today_sell_signals.append({'股票代码': code, '信号日期': date})
    # 汇总
    buy_df = pd.DataFrame(today_buy_signals)
    sell_df = pd.DataFrame(today_sell_signals)
    buy_df, sell_df = filter_signals(today_str, buy_df, sell_df)

    # 展示买入信号
    st.success(f"今日可买入股票：{'，'.join(buy_df['股票代码'])}" if not buy_df.empty else "今日无买入信号")
    if not buy_df.empty:
        st.table(buy_df)
        # 展示买入建议
        buy_codes = buy_df['股票代码'].tolist()
        next_trading_date = (datetime.strptime(today_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y.%m.%d")
        if all(is_china_stock(code) for code in buy_codes):
            st.info(f"👉 建议{next_trading_date}开盘（A股9:30AM）市价买入")
        elif all(not is_china_stock(code) for code in buy_codes):
            st.info(f"👉 建议{next_trading_date}开盘（美股9:30AM）市价买入")
        else:
            st.info(f"👉 建议{next_trading_date}开盘市价买入，A股为9:30AM，美股为9:30AM")
    # 下载按钮
    if not buy_df.empty:
        st.download_button("下载今日买入信号csv", buy_df.to_csv(index=False), file_name=f'buy_signals_{today_str}.csv')
        st.download_button("下载今日买入信号txt", "\n".join(buy_df['股票代码'].tolist()), file_name=f'buy_signals_{today_str}.txt')

    # 卖出信号
    st.error(f"今日需卖出股票：{'，'.join(sell_df['股票代码'])}" if not sell_df.empty else "今日无卖出信号")
    if not sell_df.empty:
        st.table(sell_df)
        sell_codes = sell_df['股票代码'].tolist()
        next_trading_date = (datetime.strptime(today_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y.%m.%d")
        if all(is_china_stock(code) for code in sell_codes):
            st.info(f"👉 建议{next_trading_date}开盘（A股9:30AM）市价卖出")
        elif all(not is_china_stock(code) for code in sell_codes):
            st.info(f"👉 建议{next_trading_date}开盘（美股9:30AM）市价卖出")
        else:
            st.info(f"👉 建议{next_trading_date}开盘市价卖出，A股为9:30AM，美股为9:30AM")


# TAB3 - 回测表格只显示精简5列
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
        st.download_button('下载回测结果csv', st.session_state['backtest_df'][all_cols].to_csv(index=False).encode('utf-8'), 'batch_backtest.csv')
    else:
        st.write("无回测结果")