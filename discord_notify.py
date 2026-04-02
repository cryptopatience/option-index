"""
discord_notify.py - 8대 지표 현재값 요약 + AI 종합분析을 매일 Discord 로 전송
실행: python discord_notify.py
"""

import sys
import os
import json
import pathlib
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

WEBHOOK_URL = (
    "https://discord.com/api/webhooks/1487700119137816657/"
    "rG52A7W_J8oTlsITvWaJgAuukFiOUueoICHRPW7bMEoDIcmrmMkSDBeNC8e6z4N66WMC"
)
PERIOD     = "1y"
SCRIPT_DIR = pathlib.Path(__file__).parent
STATUS_FILE = SCRIPT_DIR / "discord_status.json"


def save_status(ok_table: bool, ok_ai: bool, detail: str = "") -> None:
    now      = datetime.now()
    payload = {
        "last_sent":   now.strftime("%Y-%m-%d %H:%M:%S"),
        "table_ok":    ok_table,
        "ai_ok":       ok_ai,
        "overall_ok":  ok_table and ok_ai,
        "detail":      detail,
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── secrets.toml 로더 ────────────────────────────────────────────────────────

def load_secrets() -> dict:
    """Load API keys from .streamlit/secrets.toml"""
    secrets_path = SCRIPT_DIR / ".streamlit" / "secrets.toml"
    result = {}
    if not secrets_path.exists():
        return result
    try:
        import tomllib
        with open(secrets_path, "rb") as f:
            result = tomllib.load(f)
    except ImportError:
        # Python < 3.11 fallback: simple line parser
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ── 데이터 수집 ──────────────────────────────────────────────────────────────

def fetch(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=PERIOD, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception:
        return None


def fetch_cboe(name: str) -> pd.Series | None:
    url = (f"https://cdn.cboe.com/api/global/us_indices/daily_prices/"
           f"{name}_History.csv")
    try:
        df = pd.read_csv(url, parse_dates=[0])
        df.columns = [c.strip().upper() for c in df.columns]
        date_col  = df.columns[0]
        close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        s = df.set_index(date_col)[close_col].dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index().astype(float)
    except Exception:
        return None


def close(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None:
        return None
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.Series(s, dtype=float).dropna()


def last(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, pd.Series) and not s.empty:
        return float(s.iloc[-1])
    return None


def ratio(a, b):
    if a is None or b is None:
        return None
    m = pd.concat([a, b], axis=1).dropna()
    if m.empty:
        return None
    m.columns = ["a", "b"]
    return (m["a"] / m["b"]).replace([np.inf, -np.inf], np.nan).dropna()


def sma(s, w):
    return s.rolling(w).mean() if s is not None else None


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean() if s is not None else None


def na(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None else "N/A"


# ── 신호 점수 ────────────────────────────────────────────────────────────────

def score_sdex_voli(v):
    if v is None: return None
    if v <= 1.0: return 2
    if v <= 2.0: return 1
    return 0

def score_skew_vix(v):
    if v is None: return None
    if v <= 2.0: return 2
    if v <= 5.0: return 1
    if v < 11.0: return 0
    return -2

def score_tdex_cor1m(v):
    if v is None: return None
    if v <= 0.5: return 1
    if v < 1.0:  return 0
    return -1

def score_vvix_vix(v):
    if v is None: return None
    if v <= 3.5:  return 2
    if v <= 4.75: return 1
    if v < 6.5:   return 0
    return -1

def score_tdex(v):
    if v is None: return None
    if v > 30: return 2
    if v > 25: return 2
    if v > 20: return 1
    return 0

def score_vvix_ema(v, e):
    if v is None or e is None: return None
    return 1 if v < e else -1

def score_vix_vix3m(v):
    if v is None: return None
    if v < 1.0: return 0
    return -1

SIGNAL_EMOJI = {2: "🟢", 1: "🟢", 0: "⚪", -1: "🔴", -2: "🔴"}
SIGNAL_NAME  = {2: "+2 강한롱", 1: "+1 롱", 0: "0 중립", -1: "−1 숏", -2: "−2 강한숏"}

def fmt_signal(sc):
    if sc is None:
        return "⚫ N/A"
    return f"{SIGNAL_EMOJI.get(sc, '⚫')} {SIGNAL_NAME.get(sc, str(sc))}"


# ── SPY 통합 신호 ────────────────────────────────────────────────────────────

def build_spy_signal(r_vvix_vix, r_skew_vix, r_tdex_cor1m, r_sdex_voli, vvix_c):
    parts = {}
    w = 1.0
    if r_vvix_vix is not None:
        sig = pd.Series(0.0, index=r_vvix_vix.index)
        sig[r_vvix_vix <= 3.50] = 2.0
        sig[(r_vvix_vix > 3.50) & (r_vvix_vix <= 4.75)] = 1.0
        sig[r_vvix_vix >= 6.50] = -1.0
        parts["vvix_vix"] = (sig, w)
    if vvix_c is not None:
        e7 = vvix_c.ewm(span=7, adjust=False).mean()
        m  = pd.concat([vvix_c, e7], axis=1).dropna()
        m.columns = ["v", "e"]
        sig = pd.Series(0.0, index=m.index)
        sig[m["v"] < m["e"]] =  1.0
        sig[m["v"] > m["e"]] = -1.0
        parts["vvix_ema"] = (sig, w)
    if r_skew_vix is not None:
        sig = pd.Series(0.0, index=r_skew_vix.index)
        sig[r_skew_vix <= 2.0] = 2.0
        sig[(r_skew_vix > 2.0) & (r_skew_vix <= 5.0)] = 1.0
        sig[r_skew_vix >= 11.0] = -2.0
        parts["skew_vix"] = (sig, w)
    if r_tdex_cor1m is not None:
        sig = pd.Series(0.0, index=r_tdex_cor1m.index)
        sig[r_tdex_cor1m <= 0.5] =  1.0
        sig[r_tdex_cor1m >= 1.0] = -1.0
        parts["tdex_cor1m"] = (sig, w)
    if r_sdex_voli is not None:
        sig = pd.Series(0.0, index=r_sdex_voli.index)
        sig[r_sdex_voli <= 1.0] = 2.0
        sig[(r_sdex_voli > 1.0) & (r_sdex_voli <= 2.0)] = 1.0
        parts["sdex_voli"] = (sig, w)
    if not parts:
        return None, None
    df    = pd.concat([s for s, _ in parts.values()], axis=1).dropna()
    raw   = df.values.mean(axis=1)
    final = int(np.clip(round(float(raw[-1])), -2, 2))
    return final, float(raw[-1])


# ── 테이블 빌더 (section ⑨ 동일 구조) ───────────────────────────────────────

def build_rows(v_sdex_voli, v_skew_vix, v_tdex_cor1m, v_vvix_vix, sma50_vvix_vix,
               v_tdex, sma200_tdex, v_vvix, ema7_vvix, v_vix_vix3m,
               spy_final, spy_raw):
    return [
        {
            "지표":  "① SDEX/VOLI",
            "현재값": na(v_sdex_voli),
            "기준":  "≤1.0(+2) / ≤2.0(+1) / >2.0(0)",
            "신호":  fmt_signal(score_sdex_voli(v_sdex_voli)),
            "해석":  ("Must Buy Secular" if v_sdex_voli is not None and v_sdex_voli <= 1.0
                      else "Must Buy Cyclical" if v_sdex_voli is not None and v_sdex_voli <= 2.0
                      else "일반 시장 상태"),
        },
        {
            "지표":  "② SKEW/VIX",
            "현재값": na(v_skew_vix),
            "기준":  "≤2.0(+2) / ≤5.0(+1) / ≥11.0(−2)",
            "신호":  fmt_signal(score_skew_vix(v_skew_vix)),
            "해석":  ("Approaching Must Buy" if v_skew_vix is not None and v_skew_vix <= 2.0
                      else "Correction Lows" if v_skew_vix is not None and v_skew_vix <= 5.0
                      else "Reversal Risk" if v_skew_vix is not None and v_skew_vix >= 11.0
                      else "일반 시장 상태"),
        },
        {
            "지표":  "③ TDEX/COR1M",
            "현재값": na(v_tdex_cor1m),
            "기준":  "≤0.5(+1) / ≤1.0(0) / >1.0(−1)",
            "신호":  fmt_signal(score_tdex_cor1m(v_tdex_cor1m)),
            "해석":  ("Cheap Tail Hedge" if v_tdex_cor1m is not None and v_tdex_cor1m <= 0.5
                      else "OTM 비용 과다" if v_tdex_cor1m is not None and v_tdex_cor1m > 1.0
                      else "중립적 가격대"),
        },
        {
            "지표":  "④ VVIX/VIX",
            "현재값": f"{na(v_vvix_vix)} (SMA50:{na(sma50_vvix_vix)})",
            "기준":  "≤3.5(+2) / ≤4.75(+1) / ≥6.5(−1)",
            "신호":  fmt_signal(score_vvix_vix(v_vvix_vix)),
            "해석":  ("Getting Overdone" if v_vvix_vix is not None and v_vvix_vix <= 3.5
                      else "BTFD Potential" if v_vvix_vix is not None and v_vvix_vix <= 4.75
                      else "Getting Extended" if v_vvix_vix is not None and v_vvix_vix >= 6.5
                      else "일반 시장 상태"),
        },
        {
            "지표":  "⑤ TDEX & 200SMA",
            "현재값": f"{na(v_tdex, '{:.1f}')} (SMA:{na(sma200_tdex, '{:.1f}')})",
            "기준":  ">30(+2) / >25(+2) / >20(+1)",
            "신호":  fmt_signal(score_tdex(v_tdex)),
            "해석":  ("MUST BUY ZONE" if v_tdex is not None and v_tdex > 30
                      else "Big Crashes End" if v_tdex is not None and v_tdex > 25
                      else "Corrections End" if v_tdex is not None and v_tdex > 20
                      else "OTM Puts @ Premium"),
        },
        {
            "지표":  "⑥ VVIX & EMA7",
            "현재값": f"{na(v_vvix, '{:.1f}')} (EMA7:{na(ema7_vvix, '{:.1f}')})",
            "기준":  ">100(위험) / ≥80(중립) / <80(낙관)",
            "신호":  fmt_signal(score_vvix_ema(v_vvix, ema7_vvix)),
            "해석":  ("스트레스 레벨" if v_vvix is not None and v_vvix > 100
                      else "정상 레벨" if v_vvix is not None and v_vvix >= 80
                      else "저레벨"),
        },
        {
            "지표":  "⑦ VIX/VIX3M",
            "현재값": na(v_vix_vix3m, "{:.3f}"),
            "기준":  "≤0.8(주의) / <1.0(중립) / ≥1.0(위험)",
            "신호":  fmt_signal(score_vix_vix3m(v_vix_vix3m)),
            "해석":  ("강한 컨탱고" if v_vix_vix3m is not None and v_vix_vix3m <= 0.8
                      else "백워데이션" if v_vix_vix3m is not None and v_vix_vix3m >= 1.0
                      else "정상 기간 구조"),
        },
        {
            "지표":  "⑧ SPY 통합 신호",
            "현재값": (f"{spy_final:+d} (raw {spy_raw:.2f})"
                       if spy_final is not None else "N/A"),
            "기준":  "+2(강한롱) ~ −2(강한숏)",
            "신호":  fmt_signal(spy_final),
            "해석":  {2:"강한 롱 — 강한 매수", 1:"롱 — 매수", 0:"중립 — 관망",
                      -1:"숏 — 매도 주의", -2:"강한 숏 — 강한 매도 주의"}.get(
                         spy_final if spy_final is not None else 0, "N/A"),
        },
    ]


# ── Discord 페이로드 빌더 ─────────────────────────────────────────────────────

def build_payload(rows, spy_final, spy_raw, v_spx, v_vix, v_vvix, v_skew):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    color = (0x26a69a if spy_final is not None and spy_final > 0
             else 0xef5350 if spy_final is not None and spy_final < 0
             else 0x808080)

    spy_line = (f"{fmt_signal(spy_final)}  (raw {spy_raw:.2f})"
                if spy_final is not None else "⚫ N/A")

    # ── 헤더 embed ──────────────────────────────────────────────────────────
    header_embed = {
        "title": "📊 옵션 저점매수 전략 — ⑨ 8대 지표 현재값 요약",
        "description": (
            f"**{date_str}**\n"
            f"SPX `{na(v_spx, '{:.0f}')}` │ "
            f"VIX `{na(v_vix)}` │ "
            f"VVIX `{na(v_vvix, '{:.1f}')}` │ "
            f"SKEW `{na(v_skew, '{:.1f}')}`\n"
            f"**⑧ SPY 통합 신호**: {spy_line}"
        ),
        "color": color,
        "footer": {"text": "데이터: Yahoo Finance / CBOE"},
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }

    # ── 지표 테이블 embed (코드블록 — 모노스페이스 정렬) ────────────────────
    lines = [
        f"{'지표':<16} {'현재값':<22} {'신호':<12} {'해석'}",
        "─" * 72,
    ]
    for row in rows:
        ind   = row["지표"]
        val   = row["현재값"]
        sig   = row["신호"].replace("🟢","[매수]").replace("⚪","[중립]").replace("🔴","[주의]").replace("⚫","[N/A]")
        interp = row["해석"]
        lines.append(f"{ind:<14} {val:<22} {sig:<12} {interp}")

    table_text = "```\n" + "\n".join(lines) + "\n```"

    # ── 개별 embed fields (지표별 3열 인라인) ───────────────────────────────
    fields = []
    for row in rows:
        fields.append({
            "name": row["지표"],
            "value": (
                f"`{row['현재값']}`\n"
                f"{row['신호']}\n"
                f"_{row['해석']}_\n"
                f"```{row['기준']}```"
            ),
            "inline": True,
        })
    # 3열 정렬을 위한 빈 필드
    while len(fields) % 3 != 0:
        fields.append({"name": "\u200b", "value": "\u200b", "inline": True})

    table_embed = {
        "description": table_text,
        "color": color,
        "fields": fields,
    }

    return {"embeds": [header_embed, table_embed]}


# ── AI 종합분析 ──────────────────────────────────────────────────────────────

def build_analysis_prompt(rows, spy_final, spy_raw,
                          v_spx, v_vix, v_vvix, v_skew, v_vix3m) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    spy_line = (f"{spy_final:+d} (raw {spy_raw:.2f})"
                if spy_final is not None else "N/A")

    indicator_lines = "\n".join(
        f"- {r['지표']}: {r['현재값']}  →  {r['신호']}  ({r['해석']})"
        for r in rows
    )

    return f"""당신은 전문 옵션 트레이더이자 퀀트 애널리스트입니다.
아래는 {date_str} 기준 8대 CBOE 변동성 지표 현황입니다.
이를 바탕으로 SPY/S&P500 시장에 대한 종합 분析을 한국어로 작성하세요.

## 지표 현황
- SPX: {na(v_spx, '{:.0f}')}  |  VIX: {na(v_vix)}  |  VIX3M: {na(v_vix3m)}
- VVIX: {na(v_vvix, '{:.1f}')}  |  SKEW: {na(v_skew, '{:.1f}')}
{indicator_lines}
- SPY 통합 신호: {spy_line}

## 분析 요청 (간결하게, 총 500자 내외)
1. 현재 지표들이 시사하는 시장 환경 요약 (2~3문장)
2. 매수/중립/매도 종합 판단과 근거
3. 단기(1~2주) 핵심 리스크 1가지 + 기회 1가지
4. 지금 당장 취할 수 있는 옵션 전략 1가지 (구체적으로)
"""


def _gemini_analyze(prompt: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        for model_name in ["gemini-2.5-pro", "gemini-2.5-pro-preview-05-06",
                           "gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"]:
            try:
                model = genai.GenerativeModel(model_name)
                resp  = model.generate_content(prompt)
                return f"*{model_name}*\n\n{resp.text}"
            except Exception:
                continue
        return "Gemini: 사용 가능한 모델 없음"
    except Exception as e:
        return f"Gemini 오류: {e}"


def _openai_analyze(prompt: str, api_key: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ChatGPT 오류: {e}"


def _chunk(text: str, max_len: int = 4000) -> list[str]:
    """Split text into chunks that fit within Discord's embed description limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def build_ai_embeds(gemini_text: str | None, openai_text: str | None,
                    color: int) -> list[dict]:
    """Build Discord embeds for AI analysis (chunked if > 4000 chars)."""
    embeds = []
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    for label, icon, text in [
        ("Gemini 2.5 Pro", "🔵", gemini_text),
        ("ChatGPT GPT-4o",  "🟢", openai_text),
    ]:
        if not text:
            continue
        chunks = _chunk(text)
        for i, chunk in enumerate(chunks):
            title = (f"{icon} AI 종합분析 — {label}  [{date_str}]"
                     if i == 0 else f"{icon} {label} (계속 {i+1}/{len(chunks)})")
            embeds.append({
                "title":       title,
                "description": chunk,
                "color":       color,
            })

    return embeds


# ── Discord 전송 ─────────────────────────────────────────────────────────────

def send_discord(payload: dict) -> bool:
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Discord 전송 실패: {e}", file=sys.stderr)
        return False


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 데이터 수집 중...")

    spx_c   = close(fetch("^GSPC"))
    vix_c   = close(fetch("^VIX"))
    vvix_c  = close(fetch("^VVIX"))
    skew_c  = close(fetch("^SKEW"))
    vix3m_c = close(fetch("^VIX3M"))
    sdex_c  = close(fetch("^SDEX"))
    voli_c  = close(fetch("^VOLI"))
    tdex_c  = close(fetch("^TDEX"))
    cor1m_c = close(fetch("^COR1M"))
    if cor1m_c is None:
        cor1m_c = fetch_cboe("COR1M")

    r_vvix_vix   = ratio(vvix_c, vix_c)
    r_skew_vix   = ratio(skew_c, vix_c)
    r_sdex_voli  = ratio(sdex_c, voli_c)
    r_tdex_cor1m = ratio(tdex_c, cor1m_c)
    r_vix_vix3m  = ratio(vix_c,  vix3m_c)

    v_spx        = last(spx_c)
    v_vix        = last(vix_c)
    v_vvix       = last(vvix_c)
    v_skew       = last(skew_c)
    v_vix3m      = last(vix3m_c)
    v_tdex       = last(tdex_c)
    v_vvix_vix   = last(r_vvix_vix)
    v_skew_vix   = last(r_skew_vix)
    v_sdex_voli  = last(r_sdex_voli)
    v_tdex_cor1m = last(r_tdex_cor1m)
    v_vix_vix3m  = last(r_vix_vix3m)

    sma50_vvix_vix = last(sma(r_vvix_vix, 50))
    sma200_tdex    = last(sma(tdex_c, 200))
    ema7_vvix      = last(ema(vvix_c, 7))

    spy_final, spy_raw = build_spy_signal(
        r_vvix_vix, r_skew_vix, r_tdex_cor1m, r_sdex_voli, vvix_c
    )

    rows = build_rows(
        v_sdex_voli, v_skew_vix, v_tdex_cor1m, v_vvix_vix, sma50_vvix_vix,
        v_tdex, sma200_tdex, v_vvix, ema7_vvix, v_vix_vix3m,
        spy_final, spy_raw,
    )

    # ── 1차 전송: 지표 테이블 ───────────────────────────────────────────────
    payload = build_payload(rows, spy_final, spy_raw, v_spx, v_vix, v_vvix, v_skew)
    ok1 = send_discord(payload)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 지표 테이블 전송 {'성공' if ok1 else '실패'}")

    # ── 2차 전송: AI 종합분析 ────────────────────────────────────────────────
    secrets      = load_secrets()
    gemini_key   = secrets.get("GEMINI_API_KEY", "")
    openai_key   = secrets.get("OPENAI_API_KEY", "")

    prompt = build_analysis_prompt(
        rows, spy_final, spy_raw,
        v_spx, v_vix, v_vvix, v_skew, v_vix3m
    )

    gemini_text = _gemini_analyze(prompt, gemini_key) if gemini_key else None
    openai_text = _openai_analyze(prompt, openai_key) if openai_key else None

    color = (0x26a69a if spy_final is not None and spy_final > 0
             else 0xef5350 if spy_final is not None and spy_final < 0
             else 0x808080)

    ai_embeds = build_ai_embeds(gemini_text, openai_text, color)
    ok2 = True
    if ai_embeds:
        # Discord: 최대 10 embeds/message → 배치 전송
        for i in range(0, len(ai_embeds), 10):
            ok2 = send_discord({"embeds": ai_embeds[i:i+10]}) and ok2
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] AI 분析 전송 {'성공' if ok2 else '실패'}")
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] AI 분析 스킵 (API 키 없음)")

    detail_parts = []
    if ok1: detail_parts.append("지표 테이블 ✅")
    else:   detail_parts.append("지표 테이블 ❌")
    if ai_embeds:
        if ok2: detail_parts.append("AI 분析 ✅")
        else:   detail_parts.append("AI 분析 ❌")
    else:
        detail_parts.append("AI 분析 스킵(키 없음)")

    save_status(ok1, ok2, " | ".join(detail_parts))
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
