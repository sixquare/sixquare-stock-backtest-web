import streamlit as st
import pandas as pd
import os
from datetime import datetime

# 保证 session_state key 初始化，避免 KeyError
if 'backtest_df' not in st.session_state:
    st.session_state['backtest_df'] = None

# 其它初始化
if 'stocks_pool' not in st.session_state:
    st.session_state['stocks_pool'] = []

if 'latest_dates' not in st.session_state:
    st.session_state['latest_dates'] = []

st.set_page_config(page_title="SIXQUARE AI选股", layout="wide")

# ==== 页面结构 ====
tabs = st.tabs(["股票池与数据下载", "今日选股信号", "批量回测"])

# ============ 股票池与数据下载 =============
with tabs[0]:
    st.subheader("1. 股票池管理 & 批量数据下载")
    uploaded_file = st.file_uploader("上传股票代码txt（每行一个代码）", type=['txt'])
    if uploaded_file:
        codes = uploaded_file.read().decode('utf-8').strip().splitlines()
        st.session_state['stocks_pool'] = [c.strip().upper() for c in codes if c.strip()]
        st.success(f"导入股票数量：{len(st.session_state['stocks_pool'])}")

    if st.session_state['stocks_pool']:
        st.write("当前股票池：", ", ".join(st.session_state['stocks_pool']))

# ============ 今日选股信号 =============
with tabs[1]:
    st.subheader("2. 今日选股信号")
    st.write("请先完成数据下载")
    # 你原来的选股信号逻辑...

# ============ 批量回测 =============
with tabs[2]:
    st.header("3. 批量回测")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("回测起始日期", value=datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input("回测结束日期", value=datetime(2025, 5, 1))

    if st.button("执行批量回测"):
        # 这里举例生成假数据，请替换为你的真实回测逻辑
        data = []
        for code in st.session_state['stocks_pool']:
            # 假数据例子, 替换为你自己的回测代码
            data.append({
                '股票代码': code,
                '总盈亏率': round(100 * (0.5 - abs(hash(code)) % 100 / 200), 2),   # 随机演示
                '最大回撤率': round(abs(hash(code)) % 30 + 5, 2),
                '胜率': round(50 + abs(hash(code[::-1])) % 50, 2),
                '总交易数': abs(hash(code)) % 25 + 5,
                # 其它字段如需保留可加在这里
            })
        df = pd.DataFrame(data)
        st.session_state['backtest_df'] = df

    # 展示表格（只显示五项，自动用总盈亏率降序排序）
    show_cols = ['股票代码', '总盈亏率', '最大回撤率', '胜率', '总交易数']
    if st.session_state['backtest_df'] is not None and not st.session_state['backtest_df'].empty:
        df_show = st.session_state['backtest_df'].sort_values('总盈亏率', ascending=False)
        st.dataframe(df_show[show_cols], use_container_width=True)
        csv = st.session_state['backtest_df'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载回测结果csv", data=csv, file_name="回测结果.csv")

    else:
        st.write("无回测结果")