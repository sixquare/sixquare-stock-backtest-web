import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder

DATA_DIR = "data"
TODAY_SIGNAL_FILE = "today_buy_signal.txt"

def clear_data_dir():
    if os.path.exists(DATA_DIR):
        import shutil
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

def check_latest_dates(data_dir=DATA_DIR):
    code_dates = {}
    if not os.path.exists(data_dir):
        return code_dates
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            code = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, file))
            if not df.empty and 'Date' in df.columns:
                code_dates[code] = df['Date'].iloc[-1]
    return code_dates

def today_signal(symbols, ema_length=5, threshold=3):
    buy_list = []
    for code in symbols:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{code}.csv"))
            if df.empty or len(df) < ema_length + threshold:
                continue
            # 兼容列名
            colmap = {c.lower(): c for c in df.columns}
            close_col = colmap.get('close', None)
            if close_col is None:
                continue
            df['EMA'] = df[close_col].ewm(span=ema_length, adjust=False).mean()
            below_count = 0
            for i in range(1, len(df)):
                if df.loc[i, close_col] < df.loc[i, 'EMA']:
                    below_count = below_count + 1 if below_count else 1
                else:
                    below_count = 0
                if (below_count == threshold) and (i == len(df)-1):
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
            if 'Date' not in df.columns:
                continue
            colmap = {c.lower(): c for c in df.columns}
            close_col = colmap.get('close', None)
            high_col = colmap.get('high', None)
            if close_col is None or high_col is None:
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
            if len(df) < ema_length + threshold:
                continue
            df['EMA'] = df[close_col].ewm(span=ema_length, adjust=False).mean()
            below_count = 0
            pos = None
            trades = []
            equity = initial_capital
            equity_curve = [equity]
            for i in range(1, len(df)):
                if df.iloc[i][close_col] < df.iloc[i]['EMA']:
                    below_count += 1
                else:
                    below_count = 0
                if pos is None and below_count >= threshold:
                    size = equity / df.iloc[i][close_col]
                    entry_price = df.iloc[i][close_col]
                    pos = {'size': size, 'entry_price': entry_price}
                    equity = 0
                if pos is not None and df.iloc[i][close_col] > df.iloc[i-1][high_col]:
                    exit_price = df.iloc[i][close_col]
                    pnl = pos['size'] * (exit_price - pos['entry_price'])
                    equity = pos['size'] * exit_price
                    trades.append({'pnl': pnl})
                    pos = None
                equity_curve.append(equity if pos is None else pos['size'] * df.iloc[i][close_col])
            if pos is not None:
                last_exit_price = df.iloc[-1][close_col]
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
                "总盈亏率数值": total_return_rate*100,
                "最大回撤率数值": max_dd_rate*100,
                "胜率数值": winrate*100
            })
        except Exception as e:
            continue
    columns = ["股票代码", "总盈亏", "总盈亏率", "最大回撤", "最大回撤率", "总交易数", "盈利次数", "亏损次数", "胜率", "初始资金", "总盈亏率数值", "最大回撤率数值", "胜率数值"]
    return pd.DataFrame(results)[columns]

def to_percent_float(series):
    return series.str.rstrip('%').astype(float)

def get_today_signal_symbols():
    if os.path.exists(TODAY_SIGNAL_FILE):
        with open(TODAY_SIGNAL_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []

st.set_page_config(page_title="SIXQUARE AI选股", layout="wide")
st.title("SIXQUARE AI选股")

tabs = st.tabs(["📥 股票池与数据下载", "📊 今日选股信号", "📈 批量回测"])

# ---------------------------- TAB1 ----------------------------
with tabs[0]:
    st.header("1. 股票池管理 & 批量数据下载")
    stock_txt = st.file_uploader("上传股票代码txt（每行一个代码）", type=['txt'])
    if stock_txt:
        symbols = [line.decode('utf-8').strip().upper() for line in stock_txt if line.strip()]
        st.write(f"已导入股票数量: {len(symbols)}")
        if st.button("一键下载最新日K数据"):
            clear_data_dir()
            import akshare as ak
            status = []
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
                        df.to_csv(f"{DATA_DIR}/{code}.csv", index=False)
                        latest_date = df['Date'].iloc[-1]
                        status.append((code, "成功", latest_date))
                    else:
                        status.append((code, "无数据", ""))
                except Exception as e:
                    status.append((code, f"失败：{e}", ""))
            st.success("下载完毕！")
            dfres = pd.DataFrame(status, columns=['代码', '状态', '最新日期'])
            st.write(dfres)
            st.info(f"已保存数据至 {DATA_DIR}/，后续操作会自动读取此目录。")
    st.subheader("当前已下载股票及其数据最新日期：")
    code_dates = check_latest_dates()
    if code_dates:
        max_date = max([str(d) for d in code_dates.values() if d])
        st.markdown(f"已下载 <b>{len(code_dates)}</b> 只股票，数据最新日期：<b>{max_date}</b>", unsafe_allow_html=True)
        st.write(pd.DataFrame(list(code_dates.items()), columns=['股票代码', '最新数据日期']))
    else:
        st.write("暂无已下载数据，请先上传股票池并下载。")

# ---------------------------- TAB2 ----------------------------
with tabs[1]:
    st.header("2. 今日选股信号")
    code_dates = check_latest_dates()
    symbols = sorted(list(code_dates.keys()))
    st.write(f"当前股票池数量：{len(symbols)}")
    if code_dates:
        all_dates = list(code_dates.values())
        max_date = max([str(d) for d in all_dates if d]) if all_dates else "暂无数据"
        st.info(f"当前后台数据最新日期：{max_date}")
    else:
        st.info("当前暂无已下载数据，请先上传股票池并下载。")

    # --------- 调试参数隐藏与密码解锁 (form安全版) ----------
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

    # --------- 信号按钮 ----------
    if st.button("执行今日选股信号筛选"):
        buy_list = today_signal(symbols, ema_length, threshold)
        st.session_state['buy_list_today'] = buy_list
    buy_list_today = st.session_state.get('buy_list_today', [])
    if buy_list_today:
        st.success(f"今日可买入股票：{', '.join(buy_list_today) if buy_list_today else '无'}")
        ordered_buy_list = [code for code in symbols if code in buy_list_today]
        st.write(pd.DataFrame({'买入信号股票': ordered_buy_list}))
        st.download_button('下载csv', pd.DataFrame({'买入信号股票': ordered_buy_list}).to_csv(index=False).encode('utf-8'), 'today_buy_signal.csv')
        st.download_button('下载txt(原顺序)', "\n".join(ordered_buy_list).encode('utf-8'), 'today_buy_signal.txt')
        with open(TODAY_SIGNAL_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(ordered_buy_list))

# ---------------------------- TAB3 ----------------------------
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

    # --------- 回测调试参数隐藏与密码解锁 (form安全版) ----------
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

    # 日期选择器
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("回测起始日期", datetime(2024,1,1))
    with col2:
        end_date = st.date_input("回测结束日期", datetime(2025,5,1))

    # 回测按钮
    if st.button("执行批量回测"):
        dfres = batch_backtest(symbols_to_bt, str(start_date), str(end_date), ema_length=ema_length3, threshold=threshold3)
        if not dfres.empty:
            st.session_state['backtest_df'] = dfres

    # --------- 回测结果表格（只显示主要字段，更多/主表切换） ---------
    if st.session_state['backtest_df'] is not None and not st.session_state['backtest_df'].empty:
        all_columns = ["股票代码", "总盈亏", "总盈亏率", "最大回撤", "最大回撤率", "总交易数", "盈利次数", "亏损次数", "胜率", "初始资金"]
        main_columns = ["股票代码", "总盈亏率", "胜率"]

        if "show_all_cols" not in st.session_state:
            st.session_state.show_all_cols = False

        col_btn1, col_btn2 = st.columns([1,6])
        with col_btn1:
            if st.button("显示更多" if not st.session_state.show_all_cols else "只显示主要字段"):
                st.session_state.show_all_cols = not st.session_state.show_all_cols

        show_cols = all_columns if st.session_state.show_all_cols else main_columns

        # 排序并只显示想要的列
        display_df = st.session_state['backtest_df'].sort_values("总盈亏率数值", ascending=False)[show_cols].reset_index(drop=True)

        # aggrid 配置
        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_grid_options(sideBar=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=len(display_df))
        for col in display_df.columns:
            gb.configure_column(col, sortable=True)
        gridOptions = gb.build()

        st.write("点击表头即可排序，点击【显示更多】可展开所有字段。")
        ag_ret = AgGrid(display_df, gridOptions=gridOptions, fit_columns_on_grid_load=True, height=500, return_mode='AS_INPUT')
        st.download_button('下载回测结果csv', ag_ret['data'].to_csv(index=False).encode('utf-8'), 'batch_backtest.csv')
    else:
        st.write("无回测结果")