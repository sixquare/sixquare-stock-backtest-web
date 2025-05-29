import streamlit as st
import pandas as pd
import os
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
import io

# ========== 业务参数 ==========
DATA_DIR = 'data'
SIGNAL_FILE = 'today_signal.txt'

st.set_page_config(page_title='SIXQUARE AI选股', layout='wide')

# ========== 工具函数 ==========

def check_latest_dates():
    # 返回股票池所有最新数据日期
    if not os.path.exists(DATA_DIR):
        return {}
    dates = {}
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            if 'Date' in df.columns and len(df):
                dates[file.replace('.csv','')] = df['Date'].iloc[-1]
    return dates

def load_today_signal_codes():
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, 'r') as f:
            codes = [i.strip() for i in f if i.strip()]
        return codes
    return []

def save_today_signal_codes(codes):
    with open(SIGNAL_FILE, 'w') as f:
        f.write('\n'.join(codes))

def load_stock_data(code):
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'Date' not in df.columns or 'Open' not in df.columns:
        return None
    return df

def calculate_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def stock_signal_last_row(df, ema_len=5, threshold=3):
    # 策略：连续N根收盘低于EMA
    if len(df) < ema_len + threshold:
        return False
    ema = calculate_ema(df['Close'], ema_len)
    below = (df['Close'] < ema)
    count = 0
    for b in below.iloc[::-1]:
        if b:
            count += 1
        else:
            break
    return count >= threshold

def batch_backtest(codes, start_date, end_date, ema_len=5, threshold=3):
    results = []
    for code in codes:
        df = load_stock_data(code)
        if df is None or len(df) < ema_len+threshold:
            continue
        df = df.copy()
        df = df[df['Date'] >= start_date]
        df = df[df['Date'] <= end_date]
        if len(df) < ema_len+threshold:
            continue
        df['EMA'] = calculate_ema(df['Close'], ema_len)
        below_count = 0
        position = 0
        entry_price = 0
        cash = 10000
        max_drawdown = 0
        max_drawdown_pct = 0
        peak = cash
        win, lose, trades = 0, 0, 0
        trade_profits = []
        for i in range(1, len(df)):
            # 统计连续低于EMA根数
            if df.iloc[i]['Close'] < df.iloc[i]['EMA']:
                below_count += 1
            else:
                below_count = 0
            # 开仓
            if position == 0 and below_count >= threshold:
                position = cash / df.iloc[i]['Close']
                entry_price = df.iloc[i]['Close']
                cash = 0
            # 平仓
            if position > 0 and df.iloc[i]['Close'] > df.iloc[i-1]['High']:
                exit_price = df.iloc[i]['Close']
                profit = position * (exit_price - entry_price)
                cash = position * exit_price
                trades += 1
                trade_profits.append(profit)
                if profit >= 0:
                    win += 1
                else:
                    lose += 1
                position = 0
                entry_price = 0
            # 统计最大回撤
            equity = cash + (position * df.iloc[i]['Close'] if position > 0 else 0)
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak if peak != 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct
        # 持仓强制平仓
        if position > 0:
            exit_price = df.iloc[-1]['Close']
            profit = position * (exit_price - entry_price)
            cash = position * exit_price
            trades += 1
            trade_profits.append(profit)
            if profit >= 0:
                win += 1
            else:
                lose += 1
            position = 0
        total_profit = cash - 10000
        results.append({
            '股票代码': code,
            '总盈亏': round(total_profit, 2),
            '总盈亏率': f"{round(100*total_profit/10000, 2)}%",
            '最大回撤': round(max_drawdown, 2),
            '最大回撤率': f"{round(100*max_drawdown_pct, 2)}%",
            '总交易数': trades,
            '盈利次数': win,
            '亏损次数': lose,
            '胜率': f"{round(100*win/trades, 2)}%" if trades else '-',
            '初始资金': 10000
        })
    df_result = pd.DataFrame(results)
    return df_result

# ========== 页面 ==========

st.title("SIXQUARE AI选股")

tab1, tab2, tab3 = st.tabs(["股票池与数据下载", "今日选股信号", "批量回测"])

with tab1:
    st.subheader("1. 股票池管理 & 批量数据下载")
    latest_dates = check_latest_dates()
    st.markdown(f"当前已下载股票及其数据最新日期：{'，'.join(latest_dates.values()) if latest_dates else '暂无数据'}")
    uploaded_file = st.file_uploader("上传股票代码txt（每行一个代码）", type=['txt'])
    if uploaded_file:
        codes = [i.strip().upper() for i in uploaded_file.read().decode().splitlines() if i.strip()]
        st.write(f"已导入股票数量: {len(codes)}")
        with st.spinner('正在批量下载最新数据...'):
            import yfinance as yf
            from tqdm import tqdm
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            for code in tqdm(codes):
                try:
                    data = yf.download(code, progress=False)
                    if not data.empty:
                        data = data.reset_index()
                        data = data.rename(columns={'Date': 'Date', 'Open':'Open', 'High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
                        data[['Date','Open','High','Low','Close','Volume']].to_csv(os.path.join(DATA_DIR, f"{code}.csv"), index=False)
                except Exception as e:
                    st.warning(f"{code} 下载失败: {e}")
            st.success('数据下载完成！')

with tab2:
    st.subheader("2. 今日选股信号")
    st.markdown(f"当前后台数据最新日期：**{max(check_latest_dates().values()) if check_latest_dates() else '暂无'}**")
    # 调试参数
    if 'show_signal_debug' not in st.session_state:
        st.session_state['show_signal_debug'] = False
    if st.button("显示调试参数"):
        pwd = st.text_input("请输入调试密码", type="password", key="signal_pwd_input")
        if pwd == '1118518':
            st.session_state['show_signal_debug'] = True
            st.experimental_rerun()
        else:
            st.warning('密码错误！')
    if st.session_state['show_signal_debug']:
        ema_len = st.number_input("EMA长度", value=5, min_value=1)
        threshold = st.number_input("连续低于EMA根数", value=3, min_value=1)
    else:
        ema_len, threshold = 5, 3
    # 今日选股信号逻辑
    all_codes = [f.replace('.csv','') for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    today_codes = []
    latest_dates = check_latest_dates()
    for code in all_codes:
        df = load_stock_data(code)
        if df is not None and len(df):
            # 用最新一行作为“今日信号”
            if stock_signal_last_row(df, ema_len, threshold):
                today_codes.append(code)
    st.success('今日可买入股票：' + ', '.join(today_codes) if today_codes else "无可买入信号")
    save_today_signal_codes(today_codes)
    # 展示为表格+下载txt
    if today_codes:
        st.write("买入信号股票")
        st.dataframe(pd.DataFrame(today_codes, columns=['代码']))
        txt_bytes = io.BytesIO('\n'.join(today_codes).encode())
        st.download_button("下载txt", txt_bytes, file_name=f"今日选股信号_{datetime.now().strftime('%Y%m%d')}.txt")

with tab3:
    st.subheader("3. 批量回测")
    latest_dates = check_latest_dates()
    st.markdown(f"当前后台数据最新日期：**{max(latest_dates.values()) if latest_dates else '暂无'}**")
    # 日期选择器
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("回测起始日期", value=datetime(2024,1,1), format="YYYY-MM-DD")
    with col2:
        end_date = st.date_input("回测结束日期", value=datetime(2025,5,1), format="YYYY-MM-DD")
    only_today = st.checkbox("仅回测今日选股信号")
    if st.button("批量回测"):
        if only_today:
            codes = load_today_signal_codes()
        else:
            codes = [f.replace('.csv','') for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
        st.info(f"共回测{len(codes)}只股票。")
        df_result = batch_backtest(codes, str(start_date), str(end_date))
        if len(df_result):
            gb = GridOptionsBuilder.from_dataframe(df_result)
            gb.configure_pagination(enabled=False)   # 不分页
            gb.configure_default_column(editable=False, groupable=True)
            grid_options = gb.build()
            AgGrid(
                df_result,
                gridOptions=grid_options,
                height=800,
                theme='streamlit',
                fit_columns_on_grid_load=True,
                update_mode='NO_UPDATE',
                reload_data=True,
                allow_unsafe_jscode=True,
                enable_enterprise_modules=True
            )
            csv_bytes = df_result.to_csv(index=False, encoding='utf-8-sig').encode()
            st.download_button("下载回测结果csv", csv_bytes, file_name=f"回测结果_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
        else:
            st.warning("无回测结果，请检查股票池和数据。")