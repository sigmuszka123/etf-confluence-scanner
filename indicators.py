"""
ETF Confluence Scanner - indicators.py
Moduł obliczeń matematycznych, wskaźników technicznych i detekcji formacji świecowych.
Obsługuje zarówno natywne, wektoryzowane obliczenia pandas/numpy (odporne na brak bibliotek),
jak i integrację z pandas-ta.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Próba importu pandas_ta z bezpiecznym fallbackiem
try:
    import pandas_ta as ta  # type: ignore
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.debug("pandas-ta nie jest zainstalowane. Użyto natywnych funkcji matematycznych.")


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Prosta średnia krocząca (SMA)."""
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Wykładnicza średnia krocząca (EMA)."""
    return series.ewm(span=window, adjust=False).mean()


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Oblicza Wstęgi Bollingera (BB 20, 2).
    
    Zwraca:
        (bb_middle, bb_upper, bb_lower)
    """
    middle = series.rolling(window=window, min_periods=window).mean()
    # TA-Lib i rynkowy standard używają ddof=0
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return middle, upper, lower


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Oblicza Relative Strength Index (RSI 14) według formuły Wildera.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wygładzanie Wildera (alpha = 1 / window)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Obsługa przypadków brzegowych (brak strat = 100, brak zysków = 0)
    rsi = rsi.fillna(50.0)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    return rsi


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Oblicza Moving Average Convergence Divergence (MACD 12, 26, 9).
    
    Zwraca:
        (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calculate_rvol(volume_series: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Oblicza wolumen relatywny (RVol) oraz 20-sesyjną średnią wolumenu.
    
    Zwraca:
        (rvol_series, sma_volume_series)
    """
    sma_vol = volume_series.rolling(window=window, min_periods=window).mean()
    # Unikamy dzielenia przez 0
    safe_sma = sma_vol.replace(0.0, np.nan)
    rvol = volume_series / safe_sma
    return rvol, sma_vol


def detect_hammer(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    min_lower_ratio: float = 2.0,
    max_upper_ratio_of_range: float = 0.15,
    close_in_upper_pct: float = 0.35
) -> bool:
    """
    Rozpoznawanie świecy Młot / Pinbar:
    1. Dolny knot >= 2 * długość korpusu.
    2. Mały lub zerowy górny knot (<= 15% całkowitego zasięgu świecy lub <= korpus).
    3. Zamknięcie (Close) w górnych 35% dziennego zasięgu (High - Low).
    """
    candle_range = high_price - low_price
    if candle_range <= 0.0:
        return False

    body = abs(close_price - open_price)
    lower_shadow = min(open_price, close_price) - low_price
    upper_shadow = high_price - max(open_price, close_price)

    # Warunek 1: Dolny knot >= 2 * długość korpusu (dla doji korpus ~0)
    has_long_lower_shadow = (lower_shadow >= min_lower_ratio * body) if body > 0 else (lower_shadow > 0)
    
    # Warunek 2: Mały lub zerowy górny knot
    has_small_upper_shadow = (upper_shadow <= max_upper_ratio_of_range * candle_range) or (upper_shadow <= max(body, 0.05 * candle_range))

    # Warunek 3: Close w górnych 35% zasięgu świecy: (High - Close) <= 0.35 * Range
    # lub równoważnie: (Close - Low) / Range >= (1 - close_in_upper_pct) = 0.65
    is_close_in_upper_range = (high_price - close_price) <= (close_in_upper_pct * candle_range)

    return bool(has_long_lower_shadow and has_small_upper_shadow and is_close_in_upper_range)


def detect_bullish_engulfing(
    prev_open: float,
    prev_close: float,
    curr_open: float,
    curr_close: float
) -> bool:
    """
    Rozpoznawanie formacji Objęcia Hossy (Bullish Engulfing):
    1. Poprzednia świeca jest czerwona (spadkowa): prev_close < prev_open.
    2. Dzisiejsza świeca jest zielona (wzrostowa): curr_close > curr_open.
    3. Dzisiejsza zielona świeca całkowicie przykrywa swoim korpusem korpus poprzedniej czerwonej:
       curr_open <= prev_close oraz curr_close >= prev_open.
    """
    prev_is_red = prev_close < prev_open
    curr_is_green = curr_close > curr_open

    if not (prev_is_red and curr_is_green):
        return False

    prev_body_bottom = prev_close
    prev_body_top = prev_open

    curr_body_bottom = curr_open
    curr_body_top = curr_close

    # Dzisiejszy korpus całkowicie obejmuje wczorajszy
    engulfs = (curr_body_bottom <= prev_body_bottom) and (curr_body_top >= prev_body_top)
    return bool(engulfs)


def calculate_fibo_levels(df: pd.DataFrame, window: int = 60) -> Dict[str, float]:
    """
    Dynamic Swing Fibonacci:
    Wyznacza Swing High (maksimum) i Swing Low (minimum) z ostatnich 60 sesji
    oraz kluczowe poziomy zniesień Fibonacciego (od szczytu w dół do dołka):
    - 0.0 (Swing High)
    - 0.382
    - 0.500
    - 0.618
    - 1.0 (Swing Low)
    """
    if len(df) < window:
        sub_df = df
    else:
        sub_df = df.iloc[-window:]

    swing_high = float(sub_df['High'].max())
    swing_low = float(sub_df['Low'].min())
    diff = swing_high - swing_low

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "diff": diff,
        "0.0": swing_high,
        "0.382": swing_high - (0.382 * diff),
        "0.500": swing_high - (0.500 * diff),
        "0.618": swing_high - (0.618 * diff),
        "1.0": swing_low
    }


def is_in_fibo_support_zone(
    low_price: float,
    close_price: float,
    fibo_levels: Dict[str, float],
    margin_pct: float = 0.01
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Sprawdza, czy dzisiejsza cena (Low lub Close) znajduje się w strefie wsparcia
    Fibonacciego (margines +/- 1% wokół poziomu 0.500 lub 0.618).
    
    Zwraca:
        (is_in_zone, level_name, level_price)
    """
    for level_key in ["0.500", "0.618"]:
        target_price = fibo_levels.get(level_key)
        if target_price is None or target_price <= 0:
            continue
        
        lower_bound = target_price * (1.0 - margin_pct)
        upper_bound = target_price * (1.0 + margin_pct)

        # Sprawdzenie czy Low lub Close znajduje się w strefie +/- 1%
        if (lower_bound <= low_price <= upper_bound) or (lower_bound <= close_price <= upper_bound):
            return True, level_key, target_price

    return False, None, None


def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Kalkuluje kompletny zestaw wskaźników technicznych dla danego DataFrame (OHLCV).
    Dodaje kolumny in-place / zwraca wzbogacony DataFrame.
    """
    cfg_ind = config.get("indicators", {})
    bb_cfg = cfg_ind.get("bollinger_bands", {"window": 20, "num_std": 2.0})
    rsi_cfg = cfg_ind.get("rsi", {"window": 14})
    macd_cfg = cfg_ind.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    ma_cfg = cfg_ind.get("moving_averages", {"ema_short": 9, "sma_medium": 20, "sma_long": 200})
    rvol_cfg = cfg_ind.get("rvol", {"window": 20})

    out = df.copy()

    # 1. Bollinger Bands
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(
        out['Close'],
        window=bb_cfg.get("window", 20),
        num_std=bb_cfg.get("num_std", 2.0)
    )
    out['BB_Middle'] = bb_mid
    out['BB_Upper'] = bb_upper
    out['BB_Lower'] = bb_lower

    # 2. RSI
    out['RSI_14'] = calculate_rsi(out['Close'], window=rsi_cfg.get("window", 14))

    # 3. MACD
    macd_line, sig_line, hist = calculate_macd(
        out['Close'],
        fast=macd_cfg.get("fast", 12),
        slow=macd_cfg.get("slow", 26),
        signal=macd_cfg.get("signal", 9)
    )
    out['MACD'] = macd_line
    out['MACD_Signal'] = sig_line
    out['MACD_Hist'] = hist

    # 4. Moving Averages
    out['EMA_9'] = calculate_ema(out['Close'], window=ma_cfg.get("ema_short", 9))
    out['SMA_20'] = calculate_sma(out['Close'], window=ma_cfg.get("sma_medium", 20))
    out['SMA_200'] = calculate_sma(out['Close'], window=ma_cfg.get("sma_long", 200))

    # 5. RVol
    rvol, sma_vol = calculate_rvol(out['Volume'], window=rvol_cfg.get("window", 20))
    out['RVol'] = rvol
    out['SMA_Volume_20'] = sma_vol

    return out
