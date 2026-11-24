"""
chart_generator.py - Moduł bezgłowego generowania wykresów technicznych do audytu AI.
"""
from typing import Optional, Dict, Any
import mplfinance as mpf
import pandas as pd


def generate_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    output_path: str = "temp_chart.png",
    setup: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Oblicza wskaźniki na PEŁNYM df (prawdziwa SMA200!), a następnie renderuje
    120-sesyjny wykres 1D bez MACD (pod strategię Mean Reversion) z poziomami SL/TP1.
    """
    full_df = df.copy().sort_index()

    # 1. Obliczenia wskaźników NA PEŁNYCH DANYCH (przed przycięciem do 120 sesji!)
    sma20 = full_df["Close"].rolling(20).mean()
    std20 = full_df["Close"].rolling(20).std()
    bb_upper = sma20 + (2.0 * std20)
    bb_lower = sma20 - (2.0 * std20)
    ema9 = full_df["Close"].ewm(span=9, adjust=False).mean()
    sma200 = full_df["Close"].rolling(200).mean()  # Prawdziwa 200-sesyjna średnia

    # RSI(14) metodą Wildera (spójna ze standardem rynkowym)
    delta = full_df["Close"].diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 2. Dopiero teraz przycinamy okno do ostatnich 120 sesji
    window = 120
    plot_df = full_df.tail(window)
    ema9_p = ema9.tail(window)
    sma20_p = sma20.tail(window)
    bb_upper_p = bb_upper.tail(window)
    bb_lower_p = bb_lower.tail(window)
    sma200_p = sma200.tail(window)
    rsi_p = rsi.tail(window)

    # 3. Panele dodatkowe (BEZ MACD, który wprowadza bias trend-following w Vision AI)
    ap = [
        mpf.make_addplot(ema9_p, color="#ff9800", width=1.0, panel=0),
        mpf.make_addplot(sma20_p, color="#2196f3", width=1.1, panel=0),
        mpf.make_addplot(bb_upper_p, color="#4caf50", width=0.8, linestyle="dashed", panel=0),
        mpf.make_addplot(bb_lower_p, color="#4caf50", width=0.8, linestyle="dashed", panel=0),
        # Panel 2: RSI
        mpf.make_addplot(rsi_p, panel=2, color="#ab47bc", width=1.2, ylabel="RSI(14)"),
        mpf.make_addplot([70] * len(plot_df), panel=2, color="gray", linestyle="dotted", width=0.7),
        mpf.make_addplot([30] * len(plot_df), panel=2, color="gray", linestyle="dotted", width=0.7),
    ]

    if not sma200_p.isna().all():
        ap.append(mpf.make_addplot(sma200_p, color="#e91e63", width=1.5, panel=0))

    # Poziome linie SL i TP1, żeby AI widziało, czy opór leży PONIŻEJ TP1
    hlines_cfg = None
    levels_tag = ""
    if setup and setup.get("sl_price") and setup.get("tp1_price"):
        hlines_cfg = dict(
            hlines=[float(setup["sl_price"]), float(setup["tp1_price"])],
            colors=["#ef5350", "#00e676"],
            linestyle="--",
            linewidths=1.1,
            alpha=0.85,
        )
        levels_tag = f" | Red-- SL:{setup['sl_price']:.2f} | Green-- TP1:{setup['tp1_price']:.2f}"

    last_bar_date = pd.Timestamp(plot_df.index[-1]).strftime("%Y-%m-%d")
    title = (
        f"\nMEAN REVERSION 1D: {ticker} | Last Bar: {last_bar_date}\n"
        f"Orange: EMA9 | Blue: SMA20 | Pink: SMA200 | Green BB(20,2){levels_tag}"
    )

    custom_style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        rc={"font.size": 8, "figure.facecolor": "#131722", "axes.facecolor": "#131722"},
    )

    plot_kwargs: Dict[str, Any] = dict(
        type="candle",
        volume=True,
        addplot=ap,
        style=custom_style,
        title=title,
        savefig=dict(fname=output_path, dpi=120, bbox_inches="tight"),
        figsize=(12, 7),
        panel_ratios=(5, 1.5, 1.5),
    )
    if hlines_cfg:
        plot_kwargs["hlines"] = hlines_cfg

    mpf.plot(plot_df, **plot_kwargs)
    return output_path
