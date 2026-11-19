"""
chart_generator.py - Moduł bezgłowego generowania wykresów technicznych do audytu AI.
"""
import os
import mplfinance as mpf
import pandas as pd
import numpy as np

def generate_candlestick_chart(df: pd.DataFrame, ticker: str, output_path: str = "temp_chart.png") -> str:
    """
    Renderuje 120-sesyjny wykres świecowy z BB(20,2), EMA(9), SMA(200), RSI i MACD.
    Zapisuje plik PNG i zwraca ścieżkę do pliku.
    """
    plot_df = df.tail(120).copy()
    
    # 1. Obliczenia wskaźników do wykresu
    # Średnie i Bollinger
    sma20 = plot_df['Close'].rolling(20).mean()
    std20 = plot_df['Close'].rolling(20).std()
    bb_upper = sma20 + (2.0 * std20)
    bb_lower = sma20 - (2.0 * std20)
    ema9 = plot_df['Close'].ewm(span=9, adjust=False).mean()
    sma200 = plot_df['Close'].rolling(200, min_periods=30).mean()

    # RSI (14)
    delta = plot_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp12 = plot_df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = plot_df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # 2. Definicja paneli dodatkowych (addplot)
    ap = [
        # Panel 0: Główny wykres cenowy
        mpf.make_addplot(ema9, color='#ff9800', width=1.0, panel=0),         # Pomarańczowa EMA 9
        mpf.make_addplot(sma20, color='#2196f3', width=1.0, panel=0),        # Niebieska SMA 20
        mpf.make_addplot(bb_upper, color='#4caf50', width=0.8, linestyle='dashed', panel=0),
        mpf.make_addplot(bb_lower, color='#4caf50', width=0.8, linestyle='dashed', panel=0),
        mpf.make_addplot(sma200, color='#e91e63', width=1.2, panel=0),       # Różowa SMA 200
        
        # Panel 2: RSI
        mpf.make_addplot(rsi, panel=2, color='#ab47bc', width=1.2, ylabel='RSI (14)'),
        mpf.make_addplot([70]*len(plot_df), panel=2, color='gray', linestyle='dotted', width=0.7),
        mpf.make_addplot([30]*len(plot_df), panel=2, color='gray', linestyle='dotted', width=0.7),

        # Panel 3: MACD
        mpf.make_addplot(macd_line, panel=3, color='#29b6f6', width=1.0, ylabel='MACD'),
        mpf.make_addplot(signal_line, panel=3, color='#ffa726', width=1.0),
        mpf.make_addplot(macd_hist, panel=3, type='bar', color=np.where(macd_hist >= 0, '#66bb6a', '#ef5350'), width=0.7)
    ]

    # Styl wykresu zbliżony do profesjonalnych terminali
    custom_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        rc={'font.size': 8, 'figure.facecolor': '#131722', 'axes.facecolor': '#131722'}
    )

    # 3. Zapis do pliku graficznego
    mpf.plot(
        plot_df,
        type='candle',
        volume=True,           # Panel 1: Wolumen
        addplot=ap,
        style=custom_style,
        title=f"\nAUDYT TECHNICZNY RISK MANAGERA: {ticker} (1D)",
        savefig=dict(fname=output_path, dpi=120, bbox_inches='tight'),
        figsize=(12, 8),
        panel_ratios=(5, 1.5, 1.5, 1.5)
    )
    return output_path