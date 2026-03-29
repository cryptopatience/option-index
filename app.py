import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import pathlib
from datetime import datetime, timedelta

st.set_page_config(
    page_title="옵션 저점매수 전략 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── 비밀번호 보호 ─────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 옵션 저점매수 전략 대시보드")
    pwd = st.text_input("비밀번호를 입력하세요", type="password")
    if st.button("로그인", type="primary"):
        if pwd == st.secrets.get("APP_PASSWORD", "1234"):
            st.session_state.authenticated = True
            st.session_state["run_ai_on_login"] = True   # 로그인 시 자동 분석 트리거
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    st.stop()

# ── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 옵션 저점매수 전략")
    st.divider()

    st.markdown("""
**SDEX/VOLI** — 옵션 테일 리스크 vs ATM 변동성

**SKEW/VIX** — 테일 리스크 프리미엄 vs 전반 변동성

**TDEX/COR1M** — 꼬리 헤지 비용 vs 상관관계 분산

**VVIX/VIX** — 변동성의 변동성 vs 전반 변동성

**TDEX & 200일 SMA** — 꼬리 위험 절대 수준

**VVIX & 7일 EMA** — 단기 변동성 스트레스

**VIX/VIX3M** — 변동성 기간 구조

**SPY 통합 신호** — 5개 지표 가중 평균
""")
    st.divider()

    st.markdown("### ⚙️ 설정")
    period = st.selectbox(
        "데이터 기간",
        ["6mo", "1y", "2y", "3y", "5y"],
        index=4,
        format_func=lambda x: {"6mo":"6개월","1y":"1년","2y":"2년","3y":"3년","5y":"5년"}[x],
    )
    st.divider()

    st.markdown("### ⚖️ SPY 신호 가중치")
    w_vvix_vix  = st.slider("④ VVIX/VIX",   0.0, 3.0, 1.0, 0.1)
    w_vvix_ema  = st.slider("⑥ VVIX/EMA7",  0.0, 3.0, 1.0, 0.1)
    w_skew_vix  = st.slider("② SKEW/VIX",   0.0, 3.0, 1.0, 0.1)
    w_tdex_cor  = st.slider("③ TDEX/COR1M", 0.0, 3.0, 1.0, 0.1)
    w_sdex_voli = st.slider("① SDEX/VOLI",  0.0, 3.0, 1.0, 0.1)
    st.divider()

    st.markdown("### 🤖 AI 분석")
    ai_model = st.selectbox(
        "모델 선택",
        ["둘 다", "Gemini 2.5 Pro", "GPT-4o"],
        key="ai_model_sel",
    )
    ai_mode = st.radio(
        "분석 유형",
        ["Deep Dive", "기본 분석"],
        key="ai_mode_sel",
        horizontal=True,
    )

# ── 컬러 팔레트 ──────────────────────────────────────────────────────────────
C = dict(
    bg="#131722", grid="#1e2130", text="#b2b5be",
    bull="#26a69a", bear="#ef5350",
    blue="#2196F3", yellow="#FFD700", orange="#FF9800",
    purple="#9C27B0", white="#E0E0E0", cyan="#00BCD4",
    green2="#00E676", red2="#FF1744",
)

# ── 데이터 로딩 ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cboe(name: str, period: str) -> pd.Series | None:
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
    try:
        df = pd.read_csv(url, parse_dates=[0])
        df.columns = [c.strip().upper() for c in df.columns]
        date_col = df.columns[0]
        close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        s = df.set_index(date_col)[close_col].dropna()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index().astype(float)
        cutoff = pd.Timestamp.now() - pd.tseries.frequencies.to_offset(period)
        return s[s.index >= cutoff]
    except Exception:
        return None

def close(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None:
        return None
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.Series(s, dtype=float).dropna()

def ratio(a: pd.Series | None, b: pd.Series | None) -> pd.Series | None:
    if a is None or b is None:
        return None
    m = pd.concat([a, b], axis=1).dropna()
    if m.empty:
        return None
    m.columns = ["a", "b"]
    return (m["a"] / m["b"]).replace([np.inf, -np.inf], np.nan).dropna()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()

def last(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, pd.Series) and not s.empty:
        return float(s.iloc[-1])
    return None

with st.spinner("데이터 로딩 중..."):
    spx_df   = fetch("^GSPC",  period)
    vix_df   = fetch("^VIX",   period)
    vvix_df  = fetch("^VVIX",  period)
    skew_df  = fetch("^SKEW",  period)
    vix3m_df = fetch("^VIX3M", period)
    sdex_df  = fetch("^SDEX",  period)
    voli_df  = fetch("^VOLI",  period)
    tdex_df  = fetch("^TDEX",  period)
    cor1m_df = fetch("^COR1M", period)

# COR1M fallback: CBOE CDN
spx_c   = close(spx_df)
vix_c   = close(vix_df)
vvix_c  = close(vvix_df)
skew_c  = close(skew_df)
vix3m_c = close(vix3m_df)
sdex_c  = close(sdex_df)
voli_c  = close(voli_df)
tdex_c  = close(tdex_df)
cor1m_c = close(cor1m_df)
if cor1m_c is None:
    cor1m_c = fetch_cboe("COR1M", period)

r_vix_vix3m  = ratio(vix_c,  vix3m_c)
r_vvix_vix   = ratio(vvix_c, vix_c)
r_skew_vix   = ratio(skew_c, vix_c)
r_sdex_voli  = ratio(sdex_c, voli_c)
r_tdex_cor1m = ratio(tdex_c, cor1m_c)

# ── 타이틀 ──────────────────────────────────────────────────────────────────
st.title("📊 옵션 저점매수 전략 대시보드")
st.caption("SDEX/VOLI · SKEW/VIX · TDEX/COR1M · VVIX/VIX · TDEX·SMA · VVIX·EMA · VIX/VIX3M · SPY 통합신호")

# ── 현재값 메트릭 ────────────────────────────────────────────────────────────
v_spx        = last(spx_c)
v_vix        = last(vix_c)
v_vvix       = last(vvix_c)
v_skew       = last(skew_c)
v_vix3m      = last(vix3m_c)
v_tdex       = last(tdex_c)
v_vix_vix3m  = last(r_vix_vix3m)
v_vvix_vix   = last(r_vvix_vix)
v_skew_vix   = last(r_skew_vix)
v_sdex_voli  = last(r_sdex_voli)
v_tdex_cor1m = last(r_tdex_cor1m)

row1 = st.columns(6)
for col, (label, v, fmt) in zip(row1, [
    ("SPX",       v_spx,       "{:.0f}"),
    ("VIX",       v_vix,       "{:.2f}"),
    ("VVIX",      v_vvix,      "{:.2f}"),
    ("SKEW",      v_skew,      "{:.2f}"),
    ("VIX3M",     v_vix3m,     "{:.2f}"),
    ("TDEX",      v_tdex,      "{:.2f}"),
]):
    col.metric(label, fmt.format(v) if v is not None else "N/A")

row2 = st.columns(5)
for col, (label, v, fmt) in zip(row2, [
    ("VIX/VIX3M",  v_vix_vix3m,  "{:.3f}"),
    ("VVIX/VIX",   v_vvix_vix,   "{:.2f}"),
    ("SKEW/VIX",   v_skew_vix,   "{:.2f}"),
    ("SDEX/VOLI",  v_sdex_voli,  "{:.2f}"),
    ("TDEX/COR1M", v_tdex_cor1m, "{:.2f}"),
]):
    col.metric(label, fmt.format(v) if v is not None else "N/A")

st.divider()

# ── 차트 헬퍼 ───────────────────────────────────────────────────────────────
def base_layout(title: str, height: int = 300) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=C["text"], size=13)),
        height=height,
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font=dict(color=C["text"]),
        xaxis=dict(gridcolor=C["grid"], showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=C["grid"], showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=55, r=90, t=40, b=30),
        hovermode="x unified",
    )

def hl(fig, y, color, dash="dash", label="", width=1):
    fig.add_hline(y=y, line_dash=dash, line_color=color, line_width=width,
                  annotation_text=f"  {label}", annotation_font_color=color,
                  annotation_position="right")

def scatter(x, y, name, color, width=1.5, dash="solid", fill=None, fillcolor=None):
    return go.Scatter(x=x, y=y, name=name,
                      line=dict(color=color, width=width, dash=dash),
                      fill=fill, fillcolor=fillcolor)

# ── 1. SPX 캔들 차트 ─────────────────────────────────────────────────────────
st.subheader("📈 S&P 500 (SPX) — EMA 10 / 50 / 200")
if spx_df is not None:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=spx_df.index,
        open=spx_df["Open"].squeeze(), high=spx_df["High"].squeeze(),
        low=spx_df["Low"].squeeze(),   close=spx_c,
        name="SPX",
        increasing_line_color=C["bull"], decreasing_line_color=C["bear"],
        increasing_fillcolor=C["bull"], decreasing_fillcolor=C["bear"],
    ))
    for span, color, name in [(10, C["yellow"],"EMA 10"),(50, C["orange"],"EMA 50"),(200, C["blue"],"EMA 200")]:
        fig.add_trace(scatter(spx_c.index, ema(spx_c, span), name, color))
    fig.update_layout(**base_layout("S&P 500 Index", height=420))
    st.plotly_chart(fig, use_container_width=True)

# ── 2. VIX + VVIX ────────────────────────────────────────────────────────────
st.subheader("😰 VIX — 변동성 지수")
if vix_c is not None:
    fig = go.Figure()
    fig.add_trace(scatter(vix_c.index, vix_c, "VIX", C["bear"],
                          fill="tozeroy", fillcolor="rgba(239,83,80,0.12)"))
    if vvix_c is not None:
        fig.add_trace(go.Scatter(x=vvix_c.index, y=vvix_c, name="VVIX",
                                  line=dict(color=C["purple"], width=1, dash="dot"), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="VVIX", overlaying="y", side="right",
                                       gridcolor=C["grid"], showgrid=False,
                                       tickfont=dict(color=C["purple"])))
    for y, color, label in [(15,C["bull"],"15 평온"),(20,C["yellow"],"20 주의"),
                             (30,C["orange"],"30 공포"),(40,C["bear"],"40 극단 공포")]:
        hl(fig, y, color, label=label)
    fig.update_layout(**base_layout("VIX 변동성 지수", height=300))
    st.plotly_chart(fig, use_container_width=True)

# ── 3. 비율 차트 2×2 ────────────────────────────────────────────────────────
st.subheader("📐 비율 지표 차트")

# ① SDEX/VOLI   ② SKEW/VIX
c1, c2 = st.columns(2)

with c1:  # ① SDEX/VOLI
    st.markdown("**① SDEX/VOLI** — 옵션 테일 리스크 vs ATM 변동성")
    if r_sdex_voli is not None:
        fig = go.Figure()
        fig.add_trace(scatter(r_sdex_voli.index, r_sdex_voli, "SDEX/VOLI", C["yellow"],
                              fill="tozeroy", fillcolor="rgba(255,215,0,0.08)"))
        hl(fig, 1.0, C["bull"],   label="1.0 Must Buy Secular Lows (+2)")
        hl(fig, 2.0, C["orange"], label="2.0 Must Buy Cyclical Lows (+1)")
        fig.update_layout(**base_layout("SDEX/VOLI", height=280))
        st.plotly_chart(fig, use_container_width=True)
        v = v_sdex_voli
        if v is not None:
            if v <= 1.0:   st.success(f"🟢 현재 {v:.2f} — Must Buy Secular Lows (+2)")
            elif v <= 2.0: st.success(f"🟢 현재 {v:.2f} — Must Buy Cyclical Lows (+1)")
            else:          st.info(f"⚪ 현재 {v:.2f} — 일반 시장 상태 (0)")
    else:
        st.warning("SDEX/VOLI 데이터 없음 (CBOE 전용)")

with c2:  # ② SKEW/VIX
    st.markdown("**② SKEW/VIX** — 테일 리스크 프리미엄 vs 전반 변동성")
    if r_skew_vix is not None:
        fig = go.Figure()
        fig.add_trace(scatter(r_skew_vix.index, r_skew_vix, "SKEW/VIX", C["orange"]))
        hl(fig, 2.0,  C["bull"],   label="2.0 Approaching Must Buy (+2)")
        hl(fig, 5.0,  C["yellow"], label="5.0 Correction Lows (+1)")
        hl(fig, 11.0, C["bear"],   label="11.0 Reversal Risk (−2)")
        fig.update_layout(**base_layout("SKEW/VIX", height=280))
        st.plotly_chart(fig, use_container_width=True)
        v = v_skew_vix
        if v is not None:
            if v <= 2.0:   st.success(f"🟢 현재 {v:.2f} — Approaching Must Buy Levels (+2)")
            elif v <= 5.0: st.success(f"🟢 현재 {v:.2f} — Correction / Cyclical Bear Lows (+1)")
            elif v < 11.0: st.info(f"⚪ 현재 {v:.2f} — 일반 시장 상태 (0)")
            else:          st.error(f"🔴 현재 {v:.2f} — Approaching Reversal Risk Levels (−2)")
    else:
        st.warning("SKEW 데이터 없음")

# ③ TDEX/COR1M   ④ VVIX/VIX
c3, c4 = st.columns(2)

with c3:  # ③ TDEX/COR1M
    st.markdown("**③ TDEX/COR1M** — 꼬리 헤지 비용 vs 상관관계 분산")
    if r_tdex_cor1m is not None:
        fig = go.Figure()
        fig.add_trace(scatter(r_tdex_cor1m.index, r_tdex_cor1m, "TDEX/COR1M", C["cyan"]))
        hl(fig, 0.5, C["bull"],   label="0.5 Cheap Tail Hedge (+1)")
        hl(fig, 1.0, C["bear"],   label="1.0 3σ OTM 비용 과다 (−1)")
        fig.update_layout(**base_layout("TDEX/COR1M", height=280))
        st.plotly_chart(fig, use_container_width=True)
        v = v_tdex_cor1m
        if v is not None:
            if v <= 0.5:  st.success(f"🟢 현재 {v:.2f} — Cheap Tail Hedge (+1)")
            elif v <= 1.0:st.info(f"⚪ 현재 {v:.2f} — 중립적 가격대 (0)")
            else:         st.error(f"🔴 현재 {v:.2f} — 3σ OTM 보호 비용 과다 (−1)")
    else:
        st.warning("TDEX/COR1M 데이터 없음")

with c4:  # ④ VVIX/VIX
    st.markdown("**④ VVIX/VIX** — 변동성의 변동성 vs 전반 변동성 (50일 SMA)")
    if r_vvix_vix is not None:
        sma50 = sma(r_vvix_vix, 50)
        fig = go.Figure()
        fig.add_trace(scatter(r_vvix_vix.index, r_vvix_vix, "VVIX/VIX", C["purple"]))
        fig.add_trace(scatter(sma50.index, sma50, "50일 SMA", C["yellow"], width=1.2, dash="dot"))
        hl(fig, 3.50, C["bull"],   label="3.50 Getting Overdone (+2)")
        hl(fig, 4.75, C["yellow"], label="4.75 BTFD Potential (+1)")
        hl(fig, 6.50, C["bear"],   label="6.50 Getting Extended (−1)")
        fig.update_layout(**base_layout("VVIX/VIX", height=280))
        st.plotly_chart(fig, use_container_width=True)
        v = v_vvix_vix
        v_sma50 = last(sma50)
        if v is not None:
            if v <= 3.5:   st.success(f"🟢 현재 {v:.2f} — Getting Overdone: 역발상 강한 롱 (+2)")
            elif v <= 4.75:st.success(f"🟢 현재 {v:.2f} — Correction BTFD Potential (+1)")
            elif v < 6.5:  st.info(f"⚪ 현재 {v:.2f} — 일반 시장 상태 (0)")
            else:          st.error(f"🔴 현재 {v:.2f} — Getting Extended: 과도한 공포 (−1)")
            if v_sma50:
                arrow = "↑ SMA50 상회 → 공포 증가" if v > v_sma50 else "↓ SMA50 하회 → 공포 완화"
                st.caption(f"50일 SMA: {v_sma50:.2f}  |  {arrow}")
    else:
        st.warning("VVIX 데이터 없음")

# ── 4. TDEX & 200일 SMA   +   VVIX & 7일 EMA ────────────────────────────────
c5, c6 = st.columns(2)

with c5:  # ⑤ TDEX & 200일 SMA
    st.markdown("**⑤ TDEX & 200일 SMA** — 꼬리 위험 절대 수준 + 추세")
    if tdex_c is not None:
        sma200 = sma(tdex_c, 200)
        v_sma200 = last(sma200)
        fig = go.Figure()
        fig.add_trace(scatter(tdex_c.index, tdex_c, "TDEX", C["cyan"],
                              fill="tozeroy", fillcolor="rgba(0,188,212,0.10)"))
        fig.add_trace(scatter(sma200.index, sma200, "200일 SMA", C["blue"], width=1.5, dash="dot"))
        hl(fig, 30,  C["bear"],   label="30 MUST BUY ZONE (+2)")
        hl(fig, 25,  C["orange"], label="25 Big Crashes End (+2)")
        hl(fig, 20,  C["yellow"], label="20 Corrections End (+1)")
        hl(fig, 15,  C["bull"],   label="15 Harder to Extend (0)", dash="dot")
        hl(fig, 7.5, C["white"],  label="7.5 OTM Puts @ Premium (0)", dash="dot")
        if v_sma200:
            hl(fig, v_sma200, C["blue"], dash="solid", label=f"SMA200 {v_sma200:.1f}")
        fig.update_layout(**base_layout("TDEX & 200일 SMA — 꼬리 위험 수준", height=280))
        st.plotly_chart(fig, use_container_width=True)
        v = v_tdex
        if v is not None:
            if v > 30:         st.success(f"🟢 현재 {v:.1f} — MUST BUY ZONE (+2)")
            elif v > 25:       st.success(f"🟢 현재 {v:.1f} — Big Crashes End Above 25 (+2)")
            elif v > 20:       st.success(f"🟢 현재 {v:.1f} — Corrections End Above 20 (+1)")
            elif v > 15:       st.info(f"⚪ 현재 {v:.1f} — Harder for Corrections to Extend (0)")
            elif v > 7.5:      st.info(f"⚪ 현재 {v:.1f} — OTM Puts @ Premium (0)")
            else:              st.warning(f"❓ 현재 {v:.1f} — Correction Ending or Crash Beginning")
            if v_sma200:
                arrow = "↑ SMA200 상회 → 꼬리 위험 증가" if v > v_sma200 else "↓ SMA200 하회 → 꼬리 위험 감소"
                st.caption(f"200일 SMA: {v_sma200:.1f}  |  {arrow}")
    else:
        st.warning("TDEX 데이터 없음")

with c6:  # ⑥ VVIX & 7일 EMA
    st.markdown("**⑥ VVIX & 7일 EMA** — 단기 변동성 스트레스 + EMA 방향")
    if vvix_c is not None:
        ema7 = ema(vvix_c, 7)
        v_vvix = last(vvix_c)
        v_ema7 = last(ema7)
        fig = go.Figure()
        fig.add_trace(scatter(vvix_c.index, vvix_c, "VVIX", C["purple"],
                              fill="tozeroy", fillcolor="rgba(156,39,176,0.12)"))
        fig.add_trace(scatter(ema7.index, ema7, "7일 EMA", C["yellow"], width=1.5))
        hl(fig, 100, C["bear"],   label="100 스트레스 레벨 (위험)")
        hl(fig, 80,  C["bull"],   label="80 저레벨 (낙용)")
        if v_vvix:
            hl(fig, v_vvix, C["white"], dash="solid", label=f"현재 {v_vvix:.1f}")
        fig.update_layout(**base_layout("VVIX — 스트레스 레벨 & 7일 EMA", height=280))
        st.plotly_chart(fig, use_container_width=True)
        if v_vvix is not None:
            if v_vvix > 100:   st.error(f"🔴 현재 {v_vvix:.1f} — 스트레스 레벨 (패닉 상태)")
            elif v_vvix >= 80: st.info(f"⚪ 현재 {v_vvix:.1f} — 정상 레벨 (중립)")
            else:              st.success(f"🟢 현재 {v_vvix:.1f} — 저레벨 (Low Stress)")
            if v_ema7:
                arrow = "↑ EMA7 상회 → VIX 숏 주의" if v_vvix > v_ema7 else "↓ EMA7 하회 → VIX 숏 유리"
                st.caption(f"7일 EMA: {v_ema7:.1f}  |  {arrow}")
    else:
        st.warning("VVIX 데이터 없음")

# ── 5. VIX/VIX3M ────────────────────────────────────────────────────────────
st.markdown("**⑦ VIX/VIX3M** — 단기 변동성 기간 구조 (컨탱고 vs 백워데이션)")
if r_vix_vix3m is not None:
    fig = go.Figure()
    fig.add_trace(scatter(r_vix_vix3m.index, r_vix_vix3m, "VIX/VIX3M", C["blue"],
                          fill="tozeroy", fillcolor="rgba(33,150,243,0.10)"))
    hl(fig, 0.8, C["yellow"], label="0.8 강한 컨탱고 (역설적 주의)")
    hl(fig, 1.0, C["bear"],   label="1.0 백워데이션 (시장 충격 신호)")
    fig.update_layout(**base_layout("VIX/VIX3M — 변동성 기간 구조", height=280))
    st.plotly_chart(fig, use_container_width=True)
    v = v_vix_vix3m
    if v is not None:
        if v <= 0.8:  st.warning(f"🟡 현재 {v:.3f} — 강한 컨탱고: 안정적이나 과거 VIX 급등 사례 多 (역설적 주의)")
        elif v < 1.0: st.info(f"⚪ 현재 {v:.3f} — 정상 기간 구조 (중립)")
        else:         st.error(f"🔴 현재 {v:.3f} — 백워데이션: 시장 스트레스 급증 (위험)")
else:
    st.warning("VIX/VIX3M 데이터 없음")

def build_spy_signal_prompt():
    """SPY 통합 신호 전용 분석 프롬프트"""
    def fv(v, fmt="{:.2f}"): return fmt.format(v) if v is not None else "N/A"

    cur_signal = "N/A"
    signal_name = "N/A"
    if sig_df is not None and not sig_df.empty:
        cur_raw_v  = float(sig_df["raw_signal"].iloc[-1])
        cur_final  = int(sig_df["final_signal"].iloc[-1])
        cur_signal = f"{cur_final:+d} (raw {cur_raw_v:.2f})"
        signal_name = {2:"강한 롱",1:"롱",0:"중립",-1:"숏",-2:"강한 숏"}.get(cur_final, "N/A")

    contrib_lines = []
    if sig_df is not None and not sig_df.empty:
        ind_cols   = [c for c in sig_df.columns if c not in ("raw_signal","final_signal")]
        total_w    = sum([w_vvix_vix, w_vvix_ema, w_skew_vix, w_tdex_cor, w_sdex_voli])
        weight_map = {
            "④ VVIX/VIX": w_vvix_vix, "⑥ VVIX/EMA7": w_vvix_ema,
            "② SKEW/VIX": w_skew_vix, "③ TDEX/COR1M": w_tdex_cor,
            "① SDEX/VOLI": w_sdex_voli,
        }
        for col in ind_cols:
            sc = sig_df[col].iloc[-1]
            w  = weight_map.get(col, 1.0)
            contrib = sc * w / total_w if total_w > 0 else 0
            contrib_lines.append(f"  - {col}: 신호 {int(sc):+d}, 가중치 {w:.1f}, 기여도 {contrib:+.3f}")

    history_lines = []
    if sig_df is not None and not sig_df.empty:
        chg     = sig_df["final_signal"].diff().fillna(0) != 0
        changed = sig_df[chg].tail(7)
        for dt, row in changed.iterrows():
            history_lines.append(
                f"  - {dt.strftime('%Y-%m-%d')}: {int(row['final_signal']):+d} (raw {row['raw_signal']:.2f})"
            )

    contrib_text = "\n".join(contrib_lines) if contrib_lines else "  데이터 없음"
    history_text = "\n".join(history_lines) if history_lines else "  이력 없음"
    date_str = datetime.now().strftime("%Y-%m-%d")

    return f"""당신은 전문 옵션 트레이더이자 퀀트 애널리스트입니다.
아래 SPY 통합 신호 데이터를 바탕으로 한국어로 상세 분석을 작성하세요.

## 📊 SPY 통합 신호 현황 ({date_str})

- **현재 최종 신호**: {cur_signal}  →  {signal_name}
- **신호 범위**: −2(강한 숏) ~ +2(강한 롱), 5개 지표 가중 평균

### 지표별 기여도 (최신 봉)
{contrib_text}

### 최근 신호 변화 이력
{history_text}

### 참고 지표값
- SPX: {fv(v_spx, '{:.0f}')}  |  VIX: {fv(v_vix)}  |  VIX3M: {fv(v_vix3m)}
- VVIX/VIX: {fv(v_vvix_vix)}  |  SKEW/VIX: {fv(v_skew_vix)}
- SDEX/VOLI: {fv(v_sdex_voli)}  |  TDEX/COR1M: {fv(v_tdex_cor1m)}

---

## 분석 요청

### 1. 현재 신호 해석
현재 통합 신호({cur_signal})의 의미를 각 지표별 기여도와 함께 구체적으로 설명하세요.
어떤 지표가 가장 큰 영향을 미쳤으며, 신호의 신뢰도는 어느 정도인지 평가하세요.

### 2. 신호 히스토리 맥락
최근 신호 변화 이력을 바탕으로 현재 신호가 추세 전환점인지, 추세 지속인지 판단하세요.
신호가 빈번히 바뀌고 있다면 그 의미(불확실성 증가 등)도 분석하세요.

### 3. 신호 신뢰도 평가
5개 지표가 동일한 방향을 가리키는지, 혼재된 신호인지 평가하세요.
신호 간 충돌이 있다면 어떤 지표를 우선시해야 하는지 이유와 함께 제시하세요.

### 4. 구체적 매매 전략
현재 신호({signal_name})에 근거한 구체적인 SPY/SPX 옵션 트레이드를 제안하세요.
- 전략 구조 (예: 콜 스프레드, 풋 매수, 스트래들 등)
- 진입 시점 및 조건
- 목표가와 손절 기준
- 포지션 사이징 권고 (포트폴리오 대비 %)

### 5. 신호 무효화 조건
현재 신호가 틀릴 수 있는 시나리오와, 신호를 무효화할 지표 변화 조건을 명시하세요.
"""

def run_spy_ai_analysis():
    """SPY 통합 신호 전용 AI 분석 실행"""
    model_sel  = st.session_state.get("ai_model_sel", "둘 다")
    prompt     = build_spy_signal_prompt()
    use_gemini = model_sel in ("Gemini 2.5 Pro", "둘 다")
    use_openai = model_sel in ("GPT-4o", "둘 다")

    if use_gemini:
        try:
            import google.generativeai as genai
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            for model_name in ["gemini-2.5-pro", "gemini-2.5-pro-preview-05-06",
                                "gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp  = model.generate_content(prompt)
                    st.session_state["spy_gemini_result"] = f"*모델: {model_name}*\n\n{resp.text}"
                    break
                except Exception:
                    continue
            else:
                st.session_state["spy_gemini_result"] = "Gemini: 사용 가능한 모델을 찾지 못했습니다."
        except Exception as e:
            st.session_state["spy_gemini_result"] = f"Gemini 오류: {e}"
    else:
        st.session_state.pop("spy_gemini_result", None)

    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            st.session_state["spy_openai_result"] = resp.choices[0].message.content
        except Exception as e:
            st.session_state["spy_openai_result"] = f"ChatGPT 오류: {e}"
    else:
        st.session_state.pop("spy_openai_result", None)

    st.session_state["rerun_spy_ai"] = False

def build_indicator_prompt():
    """8개 지표 개별 분석 프롬프트"""
    def fv(v, fmt="{:.2f}"): return fmt.format(v) if v is not None else "N/A"

    _sma50_vvix_vix = last(sma(r_vvix_vix, 50)) if r_vvix_vix is not None else None
    _sma200_tdex    = last(sma(tdex_c, 200))     if tdex_c    is not None else None
    _ema7_vvix      = last(ema(vvix_c, 7))        if vvix_c    is not None else None
    _sma200_spx     = last(sma(spx_c, 200))      if spx_c    is not None else None

    date_str = datetime.now().strftime("%Y-%m-%d")

    return f"""당신은 CBOE 변동성 지표 전문 애널리스트입니다.
아래 8개 지표의 현재값을 바탕으로 각 지표별 심층 분석을 한국어로 작성하세요.
각 지표마다 반드시 ①현재값 해석 ②역사적 맥락 ③단기 시사점 ④주의사항을 포함하세요.

날짜: {date_str}
SPX: {fv(v_spx, '{:.0f}')} (200일 SMA: {fv(_sma200_spx, '{:.0f}')})

---

### ① SDEX/VOLI — 옵션 테일 리스크 vs ATM 변동성
현재값: **{fv(v_sdex_voli)}**
기준: ≤1.0(+2 Must Buy Secular) / 1.0~2.0(+1 Must Buy Cyclical) / >2.0(0 중립)
SDEX는 S&P500 하방 분산 지수, VOLI는 ATM 변동성 지수입니다.
이 비율이 낮을수록 테일 리스크가 ATM 변동성 대비 저렴하게 가격화된 것입니다.

### ② SKEW/VIX — 테일 리스크 프리미엄 vs 전반 변동성
현재값: **{fv(v_skew_vix)}**  (SKEW: {fv(v_skew, '{:.1f}')}, VIX: {fv(v_vix)})
기준: ≤2.0(+2 Must Buy) / 2.0~5.0(+1 Correction Lows) / 5.0~11.0(0) / ≥11.0(−2 Reversal Risk)
SKEW가 낮고 VIX가 높을 때 이 비율이 낮아져 역발상 매수 기회입니다.

### ③ TDEX/COR1M — 꼬리 헤지 비용 vs 상관관계 분산
현재값: **{fv(v_tdex_cor1m)}**
기준: ≤0.5(+1 Cheap Tail Hedge) / 0.5~1.0(0 중립) / >1.0(−1 비용 과다)
TDEX=꼬리 위험 지수, COR1M=1개월 내재 상관관계 지수입니다.

### ④ VVIX/VIX — 변동성의 변동성 vs 전반 변동성
현재값: **{fv(v_vvix_vix)}** (50일 SMA: {fv(_sma50_vvix_vix)})
현재 SMA50 {'상회' if v_vvix_vix is not None and _sma50_vvix_vix is not None and v_vvix_vix > _sma50_vvix_vix else '하회'} — {'공포 증가 추세' if v_vvix_vix is not None and _sma50_vvix_vix is not None and v_vvix_vix > _sma50_vvix_vix else '공포 완화 추세'}
기준: ≤3.5(+2 Getting Overdone) / 3.5~4.75(+1 BTFD) / 4.75~6.5(0) / ≥6.5(−1 Getting Extended)

### ⑤ TDEX & 200일 SMA — 꼬리 위험 절대 수준
현재값: **{fv(v_tdex, '{:.1f}')}** (200일 SMA: {fv(_sma200_tdex, '{:.1f}')})
TDEX {'>' if v_tdex is not None and _sma200_tdex is not None and v_tdex > _sma200_tdex else '<'} SMA200 → {'꼬리 위험 증가 추세' if v_tdex is not None and _sma200_tdex is not None and v_tdex > _sma200_tdex else '꼬리 위험 감소 추세'}
기준: >30(+2 MUST BUY) / >25(+2 Big Crashes End) / >20(+1 Corrections End) / 15~20(0) / 7.5~15(0)

### ⑥ VVIX & 7일 EMA — 단기 변동성 스트레스
현재값: **{fv(v_vvix, '{:.1f}')}** (7일 EMA: {fv(_ema7_vvix, '{:.1f}')})
VVIX {'>' if v_vvix is not None and _ema7_vvix is not None and v_vvix > _ema7_vvix else '<'} EMA7 → {'공포 상승 중 (VIX 숏 주의)' if v_vvix is not None and _ema7_vvix is not None and v_vvix > _ema7_vvix else '공포 완화 중 (VIX 숏 유리)'}
기준: >100(위험/패닉) / 80~100(정상) / <80(저레벨)

### ⑦ VIX/VIX3M — 변동성 기간 구조
현재값: **{fv(v_vix_vix3m, '{:.3f}')}**  (VIX: {fv(v_vix)}, VIX3M: {fv(v_vix3m)})
기준: ≤0.8(컨탱고/역설적 주의) / 0.8~1.0(정상) / ≥1.0(백워데이션/위험)

### ⑧ SPY 통합 신호 — 5개 지표 가중 평균
현재 최종 신호: **{f'{int(sig_df["final_signal"].iloc[-1]):+d} (raw {float(sig_df["raw_signal"].iloc[-1]):.2f})' if sig_df is not None and not sig_df.empty else 'N/A'}**
구성: SDEX/VOLI + SKEW/VIX + TDEX/COR1M + VVIX/VIX + VVIX/EMA7 (각 가중치 동일)

---

## 작성 지침
- 각 지표별로 독립된 섹션으로 작성 (마크다운 ### 헤더 유지)
- 현재값이 역사적으로 어느 백분위에 해당하는지 추정하여 설명
- 각 지표가 현재 시장의 어떤 측면을 반영하는지 명확히 설명
- 지표 간 연계성이 있는 경우 상호 참조하여 분석
- 마지막에 "종합: 8대 지표가 공통적으로 시사하는 한 가지 핵심 메시지" 섹션 추가
"""

def run_indicator_ai_analysis():
    """8개 지표 개별 AI 분析 실행"""
    model_sel  = st.session_state.get("ai_model_sel", "둘 다")
    prompt     = build_indicator_prompt()
    use_gemini = model_sel in ("Gemini 2.5 Pro", "둘 다")
    use_openai = model_sel in ("GPT-4o", "둘 다")

    if use_gemini:
        try:
            import google.generativeai as genai
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            for model_name in ["gemini-2.5-pro", "gemini-2.5-pro-preview-05-06",
                                "gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp  = model.generate_content(prompt)
                    st.session_state["ind_gemini_result"] = f"*모델: {model_name}*\n\n{resp.text}"
                    break
                except Exception:
                    continue
            else:
                st.session_state["ind_gemini_result"] = "Gemini: 사용 가능한 모델을 찾지 못했습니다."
        except Exception as e:
            st.session_state["ind_gemini_result"] = f"Gemini 오류: {e}"
    else:
        st.session_state.pop("ind_gemini_result", None)

    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            st.session_state["ind_openai_result"] = resp.choices[0].message.content
        except Exception as e:
            st.session_state["ind_openai_result"] = f"ChatGPT 오류: {e}"
    else:
        st.session_state.pop("ind_openai_result", None)

    st.session_state["rerun_ind_ai"] = False

# ── 6. SPY 통합 신호 ─────────────────────────────────────────────────────────
st.divider()
st.subheader("⑧ SPY 통합 신호 — 5개 지표 가중 평균 (−2 ~ +2)")

# ── 신호 시계열 계산 (Pine Script 로직 동일) ──────────────────────────────────
def build_spy_signal(r_vvix_vix, r_skew_vix, r_tdex_cor1m, r_sdex_voli, vvix_c,
                     w1, w2, w3, w4, w5):
    """각 지표의 역사적 신호 시계열 → 가중 평균 → 최종 신호"""

    parts = {}  # name -> (signal_series, weight)

    if r_vvix_vix is not None and w1 > 0:
        sig = pd.Series(0.0, index=r_vvix_vix.index)
        sig[r_vvix_vix <= 3.50] = 2.0
        sig[(r_vvix_vix > 3.50) & (r_vvix_vix <= 4.75)] = 1.0
        sig[r_vvix_vix >= 6.50] = -1.0
        parts["④ VVIX/VIX"] = (sig, w1)

    if vvix_c is not None and w2 > 0:
        e7 = vvix_c.ewm(span=7, adjust=False).mean()
        m = pd.concat([vvix_c, e7], axis=1).dropna()
        m.columns = ["v", "e"]
        sig = pd.Series(0.0, index=m.index)
        sig[m["v"] < m["e"]] =  1.0   # 공포 완화
        sig[m["v"] > m["e"]] = -1.0   # 위험 상승
        parts["⑥ VVIX/EMA7"] = (sig, w2)

    if r_skew_vix is not None and w3 > 0:
        sig = pd.Series(0.0, index=r_skew_vix.index)
        sig[r_skew_vix <= 2.0] = 2.0
        sig[(r_skew_vix > 2.0) & (r_skew_vix <= 5.0)] = 1.0
        sig[r_skew_vix >= 11.0] = -2.0
        parts["② SKEW/VIX"] = (sig, w3)

    if r_tdex_cor1m is not None and w4 > 0:
        sig = pd.Series(0.0, index=r_tdex_cor1m.index)
        sig[r_tdex_cor1m <= 0.5] =  1.0
        sig[r_tdex_cor1m >= 1.0] = -1.0
        parts["③ TDEX/COR1M"] = (sig, w4)

    if r_sdex_voli is not None and w5 > 0:
        sig = pd.Series(0.0, index=r_sdex_voli.index)
        sig[r_sdex_voli <= 1.0] = 2.0
        sig[(r_sdex_voli > 1.0) & (r_sdex_voli <= 2.0)] = 1.0
        parts["① SDEX/VOLI"] = (sig, w5)

    if not parts:
        return None

    # 가중 평균 (NaN 무시 — 데이터 없는 지표 제외)
    df  = pd.concat([s for s, _ in parts.values()], axis=1)
    df.columns = list(parts.keys())
    wts = np.array([w for _, w in parts.values()])

    vals    = df.values.astype(float)
    w_mat   = np.where(~np.isnan(vals), wts, 0.0)
    row_w   = w_mat.sum(axis=1)
    row_w[row_w == 0] = np.nan
    raw     = np.nansum(vals * w_mat, axis=1) / row_w
    final   = np.clip(np.round(raw), -2, 2)

    result = pd.DataFrame({"raw_signal": raw, "final_signal": final}, index=df.index)
    for col in df.columns:
        result[col] = df[col].values
    return result.dropna(subset=["raw_signal"])

sig_df = build_spy_signal(
    r_vvix_vix, r_skew_vix, r_tdex_cor1m, r_sdex_voli, vvix_c,
    w_vvix_vix, w_vvix_ema, w_skew_vix, w_tdex_cor, w_sdex_voli,
)

SIGNAL_LABELS = {
    2: "🟢 강한 롱 (Strong Long)",  1: "🟢 롱 (Long)",
    0: "⚪ 중립 (Neutral)",        -1: "🔴 숏 (Short)",
   -2: "🔴 강한 숏 (Strong Short)",
}

if sig_df is not None and not sig_df.empty:
    # ── 차트 ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # 배경 컬러 존 (TradingView hrect 재현)
    for y0, y1, fc in [
        ( 1.5,  2.5, "rgba(0,230,118,0.10)"),
        ( 0.5,  1.5, "rgba(0,230,118,0.05)"),
        (-0.5,  0.5, "rgba(150,150,150,0.04)"),
        (-1.5, -0.5, "rgba(255,23,68,0.05)"),
        (-2.5, -1.5, "rgba(255,23,68,0.10)"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fc, line_width=0)

    # 기준선 (Pine Script hline 재현)
    for y, color, label in [
        ( 1.5, "rgba(0,230,118,0.6)",  "1.5 강한 롱 기준"),
        ( 0.5, "rgba(0,230,118,0.4)",  "0.5 롱 기준"),
        ( 0.0, "rgba(180,180,180,0.5)","0 중립선"),
        (-0.5, "rgba(255,23,68,0.4)",  "-0.5 숏 기준"),
        (-1.5, "rgba(255,23,68,0.6)",  "-1.5 강한 숏 기준"),
    ]:
        hl(fig, y, color, dash="dot", label=label)

    # Raw 신호 — 파란 실선 (TradingView plot style_line)
    fig.add_trace(go.Scatter(
        x=sig_df.index, y=sig_df["raw_signal"],
        name="통합 신호 (raw)", line=dict(color=C["blue"], width=2),
    ))

    # Final 신호 — 흰색 계단선 (TradingView plot style_stepline)
    fig.add_trace(go.Scatter(
        x=sig_df.index, y=sig_df["final_signal"],
        name="최종 신호", line=dict(color=C["white"], width=3, shape="hv"),
        mode="lines",
    ))

    # 삼각형 마커 — 신호 변화 지점만 (TradingView plotshape 재현)
    chg = sig_df["final_signal"].diff().fillna(0) != 0
    changed = sig_df[chg]

    for sig_val, sym, color, name in [
        ( 2, "triangle-up",   "#00E676", "강한 롱 ▲▲"),
        ( 1, "triangle-up",   "#69F0AE", "롱 ▲"),
        (-1, "triangle-down", "#FF5252", "숏 ▼"),
        (-2, "triangle-down", "#B71C1C", "강한 숏 ▼▼"),
    ]:
        pts = changed[changed["final_signal"] == sig_val]
        if not pts.empty:
            sz = 16 if abs(sig_val) == 2 else 11
            fig.add_trace(go.Scatter(
                x=pts.index, y=pts["final_signal"],
                mode="markers", name=name,
                marker=dict(symbol=sym, size=sz, color=color,
                            line=dict(color="white", width=1)),
            ))

    fig.update_layout(**base_layout("⑧ SPY 통합 매매 신호 — 가중 평균 (Pine Script 동일 로직)", height=380))
    fig.update_yaxes(range=[-2.6, 2.6], tickvals=[-2, -1, 0, 1, 2])
    st.plotly_chart(fig, use_container_width=True)

    # ── 현재 신호 강조 ───────────────────────────────────────────────────────
    cur_raw   = float(sig_df["raw_signal"].iloc[-1])
    cur_final = int(sig_df["final_signal"].iloc[-1])
    cur_label = SIGNAL_LABELS.get(cur_final, "")

    mcols = st.columns([1, 2])
    mcols[0].metric("현재 최종 신호", f"{cur_final:+d}", f"raw {cur_raw:.2f}")
    with mcols[1]:
        if   cur_final >= 2:  st.success(f"✅ {cur_label}")
        elif cur_final == 1:  st.success(f"✅ {cur_label}")
        elif cur_final == 0:  st.info(f"➡️ {cur_label}")
        elif cur_final == -1: st.error(f"⚠️ {cur_label}")
        else:                 st.error(f"🚨 {cur_label}")

    # ── 지표별 기여도 테이블 (TradingView infoTable 재현) ────────────────────
    st.markdown("**지표별 기여도 (최신 봉)**")
    ind_cols   = [c for c in sig_df.columns if c not in ("raw_signal","final_signal")]
    total_w    = sum([w_vvix_vix, w_vvix_ema, w_skew_vix, w_tdex_cor, w_sdex_voli])
    weight_map = {
        "④ VVIX/VIX":   w_vvix_vix,
        "⑥ VVIX/EMA7":  w_vvix_ema,
        "② SKEW/VIX":   w_skew_vix,
        "③ TDEX/COR1M": w_tdex_cor,
        "① SDEX/VOLI":  w_sdex_voli,
    }
    tbl_rows = []
    for col in ind_cols:
        sc   = sig_df[col].iloc[-1]
        w    = weight_map.get(col, 1.0)
        contrib = sc * w / total_w if total_w > 0 else 0
        tbl_rows.append({
            "지표": col,
            "현재 신호": int(sc),
            "신호 해석": SIGNAL_LABELS.get(int(sc), "N/A"),
            "가중치": w,
            "기여도": round(contrib, 3),
        })
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

else:
    st.warning("SPY 통합 신호 계산에 필요한 데이터가 부족합니다.")

# ── 9. 8대 지표 현재값 요약 테이블 ───────────────────────────────────────────
st.divider()
st.subheader("📋 ⑨ 8대 지표 현재값 요약")

def _sig_label(sc):
    return {2:"🟢 +2 강한 롱", 1:"🟢 +1 롱", 0:"⚪ 0 중립",
            -1:"🔴 −1 숏", -2:"🔴 −2 강한 숏"}.get(int(sc) if sc is not None else 0, "N/A")

def _na(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None else "N/A"

_sma50_vvix_vix = last(sma(r_vvix_vix, 50)) if r_vvix_vix is not None else None
_sma200_tdex    = last(sma(tdex_c, 200))     if tdex_c    is not None else None
_ema7_vvix      = last(ema(vvix_c, 7))        if vvix_c    is not None else None

# 지표별 신호 점수
def _score_sdex_voli(v):
    if v is None: return None
    if v <= 1.0: return 2
    if v <= 2.0: return 1
    return 0

def _score_skew_vix(v):
    if v is None: return None
    if v <= 2.0: return 2
    if v <= 5.0: return 1
    if v < 11.0: return 0
    return -2

def _score_tdex_cor1m(v):
    if v is None: return None
    if v <= 0.5: return 1
    if v < 1.0:  return 0
    return -1

def _score_vvix_vix(v):
    if v is None: return None
    if v <= 3.5:  return 2
    if v <= 4.75: return 1
    if v < 6.5:   return 0
    return -1

def _score_tdex(v):
    if v is None: return None
    if v > 30: return 2
    if v > 25: return 2
    if v > 20: return 1
    return 0

def _score_vvix_ema(v, e):
    if v is None or e is None: return None
    return 1 if v < e else -1

def _score_vix_vix3m(v):
    if v is None: return None
    if v <= 0.8: return 0
    if v < 1.0:  return 0
    return -1

_cur_spy_score = None
if sig_df is not None and not sig_df.empty:
    _cur_spy_score = int(sig_df["final_signal"].iloc[-1])

_tbl_data = [
    {
        "지표": "① SDEX/VOLI",
        "현재값": _na(v_sdex_voli),
        "기준": "≤1.0(+2) / ≤2.0(+1) / >2.0(0)",
        "신호": _sig_label(_score_sdex_voli(v_sdex_voli)),
        "해석": ("Must Buy Secular" if v_sdex_voli is not None and v_sdex_voli <= 1.0
                 else "Must Buy Cyclical" if v_sdex_voli is not None and v_sdex_voli <= 2.0
                 else "일반 시장 상태"),
    },
    {
        "지표": "② SKEW/VIX",
        "현재값": _na(v_skew_vix),
        "기준": "≤2.0(+2) / ≤5.0(+1) / ≥11.0(−2)",
        "신호": _sig_label(_score_skew_vix(v_skew_vix)),
        "해석": ("Approaching Must Buy" if v_skew_vix is not None and v_skew_vix <= 2.0
                 else "Correction Lows" if v_skew_vix is not None and v_skew_vix <= 5.0
                 else "Reversal Risk" if v_skew_vix is not None and v_skew_vix >= 11.0
                 else "일반 시장 상태"),
    },
    {
        "지표": "③ TDEX/COR1M",
        "현재값": _na(v_tdex_cor1m),
        "기준": "≤0.5(+1) / ≤1.0(0) / >1.0(−1)",
        "신호": _sig_label(_score_tdex_cor1m(v_tdex_cor1m)),
        "해석": ("Cheap Tail Hedge" if v_tdex_cor1m is not None and v_tdex_cor1m <= 0.5
                 else "OTM 비용 과다" if v_tdex_cor1m is not None and v_tdex_cor1m > 1.0
                 else "중립적 가격대"),
    },
    {
        "지표": "④ VVIX/VIX",
        "현재값": f"{_na(v_vvix_vix)} (SMA50: {_na(_sma50_vvix_vix)})",
        "기준": "≤3.5(+2) / ≤4.75(+1) / ≥6.5(−1)",
        "신호": _sig_label(_score_vvix_vix(v_vvix_vix)),
        "해석": ("Getting Overdone" if v_vvix_vix is not None and v_vvix_vix <= 3.5
                 else "BTFD Potential" if v_vvix_vix is not None and v_vvix_vix <= 4.75
                 else "Getting Extended" if v_vvix_vix is not None and v_vvix_vix >= 6.5
                 else "일반 시장 상태"),
    },
    {
        "지표": "⑤ TDEX & 200SMA",
        "현재값": f"{_na(v_tdex, '{:.1f}')} (SMA200: {_na(_sma200_tdex, '{:.1f}')})",
        "기준": ">30(+2) / >25(+2) / >20(+1)",
        "신호": _sig_label(_score_tdex(v_tdex)),
        "해석": ("MUST BUY ZONE" if v_tdex is not None and v_tdex > 30
                 else "Big Crashes End" if v_tdex is not None and v_tdex > 25
                 else "Corrections End" if v_tdex is not None and v_tdex > 20
                 else "OTM Puts @ Premium" if v_tdex is not None and v_tdex > 7.5
                 else "N/A"),
    },
    {
        "지표": "⑥ VVIX & EMA7",
        "현재값": f"{_na(v_vvix, '{:.1f}')} (EMA7: {_na(_ema7_vvix, '{:.1f}')})",
        "기준": ">100(위험) / 80~100(중립) / <80(낙관)",
        "신호": _sig_label(_score_vvix_ema(v_vvix, _ema7_vvix)),
        "해석": ("스트레스 레벨" if v_vvix is not None and v_vvix > 100
                 else "정상 레벨" if v_vvix is not None and v_vvix >= 80
                 else "저레벨"),
    },
    {
        "지표": "⑦ VIX/VIX3M",
        "현재값": _na(v_vix_vix3m, "{:.3f}"),
        "기준": "≤0.8(주의) / <1.0(중립) / ≥1.0(위험)",
        "신호": _sig_label(_score_vix_vix3m(v_vix_vix3m)),
        "해석": ("강한 컨탱고 (역설적 주의)" if v_vix_vix3m is not None and v_vix_vix3m <= 0.8
                 else "백워데이션 (시장 스트레스)" if v_vix_vix3m is not None and v_vix_vix3m >= 1.0
                 else "정상 기간 구조"),
    },
    {
        "지표": "⑧ SPY 통합 신호",
        "현재값": (f"{_cur_spy_score:+d} (raw {float(sig_df['raw_signal'].iloc[-1]):.2f})"
                   if sig_df is not None and not sig_df.empty else "N/A"),
        "기준": "+2(강한 롱) ~ −2(강한 숏)",
        "신호": _sig_label(_cur_spy_score),
        "해석": {2:"강한 롱 — 강한 매수",1:"롱 — 매수",0:"중립 — 관망",
                 -1:"숏 — 매도 주의",-2:"강한 숏 — 강한 매도 주의"}.get(
                    _cur_spy_score if _cur_spy_score is not None else 0, "N/A"),
    },
]
st.dataframe(pd.DataFrame(_tbl_data), use_container_width=True, hide_index=True)

# ── AI 프롬프트 빌더 (지표값 확정 후 정의) ───────────────────────────────────
def build_prompt():
    def fv(v, fmt="{:.2f}"): return fmt.format(v) if v is not None else "N/A"
    lines = [
        "당신은 전문 옵션 트레이더이자 퀀트 애널리스트입니다.",
        "다음은 현재 시장의 주요 변동성 지표 현황입니다. 이를 바탕으로 SPY/S&P500 시장에 대한 종합 분석을 한국어로 작성해 주세요.",
        "",
        "## 현재 지표 현황",
        f"- SPX: {fv(v_spx, '{:.0f}')}",
        f"- VIX: {fv(v_vix)}  |  VIX3M: {fv(v_vix3m)}",
        f"- VVIX: {fv(v_vvix)}  |  SKEW: {fv(v_skew)}",
        f"- TDEX: {fv(v_tdex)}",
        f"- VIX/VIX3M: {fv(v_vix_vix3m, '{:.3f}')}",
        f"- VVIX/VIX: {fv(v_vvix_vix)}",
        f"- SKEW/VIX: {fv(v_skew_vix)}",
        f"- SDEX/VOLI: {fv(v_sdex_voli)}",
        f"- TDEX/COR1M: {fv(v_tdex_cor1m)}",
    ]
    if sig_df is not None and not sig_df.empty:
        cur_raw   = float(sig_df["raw_signal"].iloc[-1])
        cur_final = int(sig_df["final_signal"].iloc[-1])
        lines.append(f"- SPY 통합 신호: {cur_final:+d} (raw {cur_raw:.2f})")
    lines += [
        "",
        "## 분석 요청",
        "1. 각 지표 현황이 시사하는 시장 상황을 간결히 설명하세요.",
        "2. 매수/중립/매도 관점의 종합 판단을 내려주세요.",
        "3. 단기(1~2주)·중기(1~3개월) 리스크와 기회를 알려주세요.",
        "4. 현재 환경에 적합한 옵션 전략을 제안해 주세요.",
    ]
    return "\n".join(lines)

def build_deep_dive_prompt():
    """Deep Dive 7-섹션 전문가 분석 프롬프트"""
    def fv(v, fmt="{:.2f}"): return fmt.format(v) if v is not None else "N/A"

    # 추가 계산값
    sma50_vvix_vix = last(sma(r_vvix_vix, 50)) if r_vvix_vix is not None else None
    sma200_tdex    = last(sma(tdex_c, 200))     if tdex_c    is not None else None
    sma200_spx     = last(sma(spx_c, 200))      if spx_c     is not None else None
    ema7_vvix      = last(ema(vvix_c, 7))        if vvix_c    is not None else None

    def rel(val, ref, above_msg, below_msg):
        if val is None or ref is None: return "N/A"
        return above_msg if val > ref else below_msg

    vvix_vix_vs_sma50 = rel(v_vvix_vix, sma50_vvix_vix,
                             "SMA50 상회 — 공포 증가 추세", "SMA50 하회 — 공포 완화 추세")
    tdex_vs_sma200    = rel(v_tdex, sma200_tdex,
                             "SMA200 상회 — 꼬리 위험 증가", "SMA200 하회 — 꼬리 위험 감소")
    spx_vs_sma200     = rel(v_spx, sma200_spx,
                             "SMA200 상회 — 상승 추세 유지", "SMA200 하회 — 하락 추세 진입")
    vvix_vs_ema7      = rel(v_vvix, ema7_vvix,
                             "EMA7 상회 — 단기 공포 상승 중", "EMA7 하회 — 단기 공포 완화 중")

    cur_signal = "N/A"
    if sig_df is not None and not sig_df.empty:
        cur_raw   = float(sig_df["raw_signal"].iloc[-1])
        cur_final = int(sig_df["final_signal"].iloc[-1])
        cur_signal = f"{cur_final:+d} (raw {cur_raw:.2f})"

    date_str = datetime.now().strftime("%Y-%m-%d")

    return f"""당신은 10년 이상 경력의 전문 옵션 트레이더이자 CBOE 변동성 지표 전문 리서처입니다.
아래 8대 변동성 지표 데이터를 바탕으로 심층 전문 분석을 한국어로 작성하세요.
반드시 아래 7개 섹션을 모두 포함하고, 각 섹션당 최소 150단어 이상 작성하세요.

## 📊 현재 지표 데이터 ({date_str} 기준)

| 지표 | 현재값 | 상태 |
|------|--------|------|
| SPX | {fv(v_spx, '{:.0f}')} | {spx_vs_sma200} |
| VIX | {fv(v_vix)} | — |
| VIX3M | {fv(v_vix3m)} | — |
| VVIX | {fv(v_vvix)} | {vvix_vs_ema7} |
| VVIX 7일 EMA | {fv(ema7_vvix)} | — |
| SKEW | {fv(v_skew)} | — |
| TDEX | {fv(v_tdex)} | {tdex_vs_sma200} |
| TDEX 200일 SMA | {fv(sma200_tdex)} | — |
| VIX/VIX3M | {fv(v_vix_vix3m, '{:.3f}')} | — |
| VVIX/VIX | {fv(v_vvix_vix)} | {vvix_vix_vs_sma50} |
| VVIX/VIX 50일 SMA | {fv(sma50_vvix_vix)} | — |
| SKEW/VIX | {fv(v_skew_vix)} | — |
| SDEX/VOLI | {fv(v_sdex_voli)} | — |
| TDEX/COR1M | {fv(v_tdex_cor1m)} | — |
| SPY 통합 신호 | {cur_signal} | — |

---

## 🔍 심층 분석 요청 — 7개 섹션 (각 섹션 최소 150단어)

### 섹션 1. REGIME CLASSIFICATION — 현재 시장 레짐 분류
현재 지표들이 시사하는 시장 레짐(Risk-On / Risk-Off / Transition / Panic)을 분류하고, 각 지표가 해당 레짐 판단에 어떻게 기여하는지 설명하세요. 역사적으로 유사한 시장 환경(GFC 2008, COVID 2020, 2022 약세장 등)과 비교하여 현재 상황의 특이점과 공통점을 분석하세요.

### 섹션 2. CROSS-INDICATOR CONSISTENCY CHECK — 지표 간 일관성 검증
8개 지표 신호들이 서로 일관된 방향을 가리키는지, 혹은 충돌하는 신호가 있는지 구체적으로 분석하세요. 불일치하는 지표가 있다면 그 원인을 추론하고, 어떤 지표에 더 높은 신뢰도를 부여해야 하는지 판단 근거를 제시하세요.

### 섹션 3. TAIL RISK PRICING ANALYSIS — 테일 리스크 가격 분석
SDEX/VOLI, SKEW/VIX, TDEX/COR1M 세 지표를 종합하여 현재 시장에서 테일 리스크가 어떻게 가격화되고 있는지 분석하세요. OTM 풋 옵션의 현재 가격 수준이 역사적 분포에서 어느 위치에 있는지, 테일 헤지 전략의 비용 대비 효익을 구체적으로 평가하세요.

### 섹션 4. VOL TERM STRUCTURE SIGNAL — 변동성 기간 구조 신호
VIX/VIX3M 비율 및 VVIX/VIX와 그 50일 SMA를 통해 변동성 기간 구조가 시사하는 바를 분석하세요. 컨탱고/백워데이션 상태, 단기 변동성 스트레스 수준, VVIX 절대 레벨과 EMA7 대비 위치, 향후 변동성 방향성에 대한 구체적인 견해를 제시하세요.

### 섹션 5. POSITIONING & FLOW INFERENCE — 포지셔닝 및 플로우 추론
현재 지표 조합에서 기관 투자자, 헤지 펀드, 리테일 투자자의 포지셔닝을 추론하세요. 옵션 시장의 공급/수요 불균형, 딜러 감마 노출, 풋/콜 비대칭, 마켓 마이크로스트럭처 관점의 통찰을 제공하세요.

### 섹션 6. ACTIONABLE TRADE SCENARIOS — 실행 가능한 트레이드 시나리오
현재 시장 환경에 최적화된 구체적인 옵션 전략을 3가지 시나리오로 제시하세요.
- **베이스 케이스**: 진입 조건, 옵션 구조(스트라이크·만기·구조), 목표 수익률, 손절 기준
- **불 케이스**: 동일 항목
- **베어 케이스**: 동일 항목

### 섹션 7. SINGLE BOTTOM LINE — 핵심 결론 및 행동 지침
위 6개 섹션을 종합하여 단 하나의 핵심 판단을 내리세요. 향후 2~4주 내 가장 가능성 높은 시나리오, 주요 모니터링 지표 3개, 포지션 사이징 권고를 포함하세요. 결론은 "지금 당장 무엇을 해야 하는가"에 대한 명확한 행동 지침으로 마무리하세요.
"""

def run_ai_analysis():
    """선택된 모델 및 분석 유형으로 AI 분석 실행"""
    model_sel = st.session_state.get("ai_model_sel", "둘 다")
    mode_sel  = st.session_state.get("ai_mode_sel",  "기본 분석")

    prompt     = build_deep_dive_prompt() if mode_sel == "Deep Dive" else build_prompt()
    max_tokens = 4000 if mode_sel == "Deep Dive" else 1500

    use_gemini = model_sel in ("Gemini 2.5 Pro", "둘 다")
    use_openai = model_sel in ("GPT-4o", "둘 다")

    if use_gemini:
        try:
            import google.generativeai as genai
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            for model_name in ["gemini-2.5-pro", "gemini-2.5-pro-preview-05-06",
                                "gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp  = model.generate_content(prompt)
                    st.session_state["gemini_result"] = f"*모델: {model_name}*\n\n{resp.text}"
                    break
                except Exception:
                    continue
            else:
                st.session_state["gemini_result"] = "Gemini: 사용 가능한 모델을 찾지 못했습니다."
        except Exception as e:
            st.session_state["gemini_result"] = f"Gemini 오류: {e}"
    else:
        st.session_state.pop("gemini_result", None)

    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            st.session_state["openai_result"] = resp.choices[0].message.content
        except Exception as e:
            st.session_state["openai_result"] = f"ChatGPT 오류: {e}"
    else:
        st.session_state.pop("openai_result", None)

    st.session_state["run_ai_on_login"] = False
    st.session_state["rerun_ai"] = False

# ── AI 분析 트리거 ────────────────────────────────────────────────────────────
if st.session_state.get("run_ai_on_login") or st.session_state.get("rerun_ai"):
    with st.spinner("AI 종합분析 중..."):
        run_ai_analysis()

if st.session_state.get("rerun_spy_ai"):
    with st.spinner("SPY 신호 AI 분析 중..."):
        run_spy_ai_analysis()

if st.session_state.get("rerun_ind_ai"):
    with st.spinner("개별 지표 AI 분析 중..."):
        run_indicator_ai_analysis()

# ── 사이드바: AI 버튼 + Discord 상태 + 로그아웃 ─────────────────────────
with st.sidebar:
    st.divider()
    if st.button("🔄 AI 재분석", use_container_width=True):
        st.session_state["rerun_ai"] = True
        st.rerun()
    if st.button("🔍 SPY 신호 분析", use_container_width=True):
        st.session_state["rerun_spy_ai"] = True
        st.rerun()
    if st.button("📊 개별 지표 분析", use_container_width=True):
        st.session_state["rerun_ind_ai"] = True
        st.rerun()

    # ── Discord 전송 상태 ────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📨 Discord 알림")
    _status_path = pathlib.Path(__file__).parent / "discord_status.json"
    if _status_path.exists():
        try:
            _s    = json.loads(_status_path.read_text(encoding="utf-8"))
            _ok   = _s.get("overall_ok", False)
            _icon = "✅" if _ok else "❌"
            _last = _s.get("last_sent", "—")
            _next = _s.get("next_send", "—")
            _det  = _s.get("detail", "")
            st.markdown(f"{_icon} **마지막 전송**  \n`{_last}`  \n_{_det}_")
            st.markdown(f"🕗 **다음 예정**  \n`{_next}`")
        except Exception:
            st.warning("상태 파일 읽기 실패")
    else:
        _now  = datetime.now()
        _next = _now.replace(hour=8, minute=0, second=0, microsecond=0)
        if _next <= _now:
            _next += timedelta(days=1)
        st.markdown(f"⏳ **아직 전송 전**  \n다음 예정: `{_next.strftime('%Y-%m-%d 08:00')}`")

    st.markdown("<br>" * 3, unsafe_allow_html=True)
    st.divider()
    if st.button("🚪 로그아웃", use_container_width=True):
        for k in ["authenticated","gemini_result","openai_result",
                  "spy_gemini_result","spy_openai_result",
                  "ind_gemini_result","ind_openai_result",
                  "run_ai_on_login","rerun_ai","rerun_spy_ai","rerun_ind_ai"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── AI 분析 결과 출력 ─────────────────────────────────────────────────────────
def _parse_section(text: str, header_key: str) -> str:
    """AI 응답에서 특정 헤더 섹션만 추출"""
    idx = text.find(header_key)
    if idx == -1:
        return ""
    line_start = text.rfind("\n", 0, idx)
    line_start = line_start + 1 if line_start != -1 else 0
    next_h = text.find("\n###", idx + len(header_key))
    if next_h == -1:
        next_h = len(text)
    return text[line_start:next_h].strip()

has_main   = "gemini_result"     in st.session_state or "openai_result"     in st.session_state
has_spy    = "spy_gemini_result" in st.session_state or "spy_openai_result" in st.session_state
has_ind    = "ind_gemini_result" in st.session_state or "ind_openai_result" in st.session_state

if has_main or has_spy or has_ind:
    st.divider()
    st.subheader("🤖 AI 분析")

    # ── 종합 분析 ─────────────────────────────────────────────────────────
    if has_main or has_spy:
        st.markdown("#### 📊 종합분析")
        tab_labels = []
        if st.session_state.get("gemini_result"):     tab_labels.append("Gemini 종합")
        if st.session_state.get("openai_result"):     tab_labels.append("ChatGPT 종합")
        if st.session_state.get("spy_gemini_result"): tab_labels.append("Gemini SPY")
        if st.session_state.get("spy_openai_result"): tab_labels.append("ChatGPT SPY")

        if tab_labels:
            tabs = st.tabs(tab_labels)
            ti = 0
            if st.session_state.get("gemini_result"):
                with tabs[ti]: st.markdown(st.session_state["gemini_result"])
                ti += 1
            if st.session_state.get("openai_result"):
                with tabs[ti]: st.markdown(st.session_state["openai_result"])
                ti += 1
            if st.session_state.get("spy_gemini_result"):
                with tabs[ti]: st.markdown(st.session_state["spy_gemini_result"])
                ti += 1
            if st.session_state.get("spy_openai_result"):
                with tabs[ti]: st.markdown(st.session_state["spy_openai_result"])

    # ── 개별 지표 分析 ────────────────────────────────────────────────────
    if has_ind:
        with st.expander("🔬 개별 지표 分析", expanded=False):
            ind_names = [
                "① SDEX/VOLI", "② SKEW/VIX", "③ TDEX/COR1M", "④ VVIX/VIX",
                "⑤ TDEX & 200일 SMA", "⑥ VVIX & 7일 EMA", "⑦ VIX/VIX3M", "⑧ SPY 통합 신호",
            ]
            selected = st.selectbox("지표 선택", ind_names, key="ind_select")

            g_text = st.session_state.get("ind_gemini_result", "")
            o_text = st.session_state.get("ind_openai_result", "")

            tab_list = []
            if g_text: tab_list.append("Gemini")
            if o_text: tab_list.append("ChatGPT")

            if tab_list:
                ind_tabs = st.tabs(tab_list)
                ti = 0
                if g_text:
                    with ind_tabs[ti]:
                        section = _parse_section(g_text, selected)
                        st.markdown(section if section else g_text)
                    ti += 1
                if o_text:
                    with ind_tabs[ti]:
                        section = _parse_section(o_text, selected)
                        st.markdown(section if section else o_text)
