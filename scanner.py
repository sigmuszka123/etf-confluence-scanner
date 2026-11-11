"""
ETF Confluence Scanner - scanner.py
Główny moduł silnika skanera:
1. Wczytywanie bazy tickerów.
2. Pobieranie danych paczkami (Batch Download via yfinance) z obsługą błędów.
3. Filtracja płynności (SMA 20 Volume >= 5 000).
4. Wyliczanie wskaźników i ewaluacja Matrycy Konfluencji (minimum 3 z 5 kryteriów).
5. Moduł Kalkulatora Ryzyka (Entry, Stop Loss, TP1, TP2, R:R).
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

import indicators

logger = logging.getLogger(__name__)

# Próba importu yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance nie jest zainstalowane w bieżącym środowisku.")


class ETFScanner:
    """
    Skaner rynku ETF wyszukujący setupy konfluencji Price Action + Wskaźniki.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._setup_logging()

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """Bezpieczne ładowanie pliku konfiguracyjnego YAML."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Plik konfiguracyjny {path} nie istnieje.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """Konfiguracja modułu logowania."""
        log_cfg = self.config.get("logging", {})
        log_level_str = log_cfg.get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_format = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logging.basicConfig(level=log_level, format=log_format)

    def load_tickers(self) -> pd.DataFrame:
        """
        Wczytuje listę tickerów z pliku CSV zdefiniowanego w config.yaml.
        Weryfikuje obecność wymaganych kolumn (ticker, name, sector).
        """
        csv_file = self.config.get("data", {}).get("tickers_file", "tickers.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Brak pliku bazy tickerów: {csv_file}")

        df = pd.read_csv(csv_file)
        required_cols = {"ticker", "name", "sector"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Plik {csv_file} musi zawierać kolumny: {required_cols}")

        df['ticker'] = df['ticker'].astype(str).str.strip()
        df['name'] = df['name'].astype(str).str.strip()
        df['sector'] = df['sector'].astype(str).str.strip()
        df = df[df['ticker'] != ""].drop_duplicates(subset=['ticker'])
        logger.info("Załadowano %d tickerów z pliku %s", len(df), csv_file)
        return df

    def fetch_market_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Zbiorcze pobieranie danych giełdowych (Batch Download) przy użyciu yfinance.
        Zmniejsza liczbę requestów HTTP i chroni przed Rate Limitami.
        Zwraca słownik {ticker: DataFrame z kolumnami OHLCV}.
        """
        if not HAS_YFINANCE:
            logger.error("Biblioteka yfinance jest niedostępna. Nie można pobrać danych rynkowych.")
            return {}

        period = self.config.get("data", {}).get("period", "180d")
        interval = self.config.get("data", {}).get("interval", "1d")

        logger.info("Rozpoczynanie pobierania danych zbiorczych dla %d tickerów (period=%s, interval=%s)...",
                    len(tickers), period, interval)

        try:
            # yf.download z parametrem group_by="ticker"
            batch_df = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False
            )
        except Exception as e:
            logger.error("Błąd podczas wywołania yf.download: %s", e)
            return {}

        result: Dict[str, pd.DataFrame] = {}
        if batch_df is None or batch_df.empty:
            logger.warning("yf.download zwróciło pusty zbiór danych.")
            return result

        # Parsowanie struktury kolumn (MultiIndex vs pojedynczy ticker)
        for ticker in tickers:
            ticker_df = self._extract_single_ticker_df(batch_df, ticker)
            if ticker_df is not None and not ticker_df.empty:
                result[ticker] = ticker_df

        logger.info("Pomyślnie przetworzono dane dla %d / %d tickerów.", len(result), len(tickers))
        return result

    @staticmethod
    def _extract_single_ticker_df(batch_df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Wyodrębnia dane pojedynczego tickera z DataFrame MultiIndex yfinance.
        Obsługuje warianty: MultiIndex (ticker, metric) oraz (metric, ticker).
        """
        try:
            if isinstance(batch_df.columns, pd.MultiIndex):
                if ticker in batch_df.columns.levels[0]:
                    df_t = batch_df[ticker].copy()
                elif ticker in batch_df.columns.levels[1]:
                    df_t = batch_df.xs(ticker, axis=1, level=1).copy()
                else:
                    return None
            else:
                # Tylko jeden ticker był pobrany
                df_t = batch_df.copy()

            # Normalizacja kolumn do wielkich liter
            df_t.columns = [str(c).capitalize() for c in df_t.columns]
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_t.columns for col in required):
                return None

            # Usunięcie wierszy z brakującymi cenami Close
            df_t = df_t.dropna(subset=['Close'])
            if len(df_t) < 20:
                return None

            df_t = df_t.sort_index()
            return df_t
        except Exception as err:
            logger.debug("Błąd ekstrakcji danych dla %s: %s", ticker, err)
            return None

    def evaluate_ticker(
        self,
        ticker: str,
        name: str,
        sector: str,
        df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Analizuje pojedynczy instrument:
        1. Weryfikuje filtr płynności (SMA 20 Volume >= 5 000).
        2. Wylicza wskaźniki techniczne i geometrie Fibonacciego.
        3. Weryfikuje 5 kryteriów konfluencji.
        4. Jeżeli spełnione >= 3 kryteria, oblicza parametry Risk Management.
        """
        min_vol_sma = self.config.get("liquidity_filter", {}).get("min_volume_sma_20", 5000)

        # 1. FILTR PŁYNNOŚCI
        vol_series = df['Volume']
        if len(vol_series) < 20:
            logger.debug("[%s] Zbyt krótka historia danych (< 20 sesji).", ticker)
            return None

        sma_vol_20 = vol_series.rolling(20).mean().iloc[-1]
        if np.isnan(sma_vol_20) or sma_vol_20 < min_vol_sma:
            logger.debug(
                "[%s] Odrzucono przez filtr płynności: SMA(Vol,20)=%.0f < %d jednostek.",
                ticker, 0 if np.isnan(sma_vol_20) else sma_vol_20, min_vol_sma
            )
            return None

        # 2. KALKULACJA WSKAŹNIKÓW
        df_ind = indicators.calculate_all_indicators(df, self.config)
        fibo_window = self.config.get("indicators", {}).get("fibonacci", {}).get("lookback_window", 60)
        fibo_levels = indicators.calculate_fibo_levels(df, window=fibo_window)

        curr = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        prev2 = df_ind.iloc[-3] if len(df_ind) >= 3 else prev

        # Podstawowe dane cenowe
        close_curr = float(curr['Close'])
        close_prev = float(prev['Close'])
        open_curr = float(curr['Open'])
        high_curr = float(curr['High'])
        low_curr = float(curr['Low'])
        open_prev = float(prev['Open'])
        change_pct = ((close_curr - close_prev) / close_prev) * 100.0

        # Odczyty wskaźników
        rsi_val = float(curr['RSI_14'])
        rsi_prev_val = float(prev['RSI_14'])
        rsi_prev2_val = float(prev2['RSI_14'])

        bb_mid = float(curr['BB_Middle'])
        bb_upper = float(curr['BB_Upper'])
        bb_lower = float(curr['BB_Lower'])

        rvol = float(curr['RVol']) if not np.isnan(curr['RVol']) else 1.0
        hist_curr = float(curr['MACD_Hist'])
        hist_prev = float(prev['MACD_Hist'])

        # 3. EWALUACJA 5 KRYTERIÓW MATRYCY KONFLUENCJI

        # Kryterium 1: Wyprzedanie
        # RSI < 35 LUB RSI przecięło 30 od dołu w górę w ciągu ostatnich 2 sesji
        crossed_30_up = (rsi_prev_val < 30.0 and rsi_val >= 30.0) or (rsi_prev2_val < 30.0 and rsi_prev_val >= 30.0)
        c1_oversold = bool((rsi_val < 35.0) or crossed_30_up)

        if crossed_30_up:
            rsi_status = "Odbicie w górę ponad 30"
        elif rsi_val < 30.0:
            rsi_status = "Silnie wyprzedany (<30)"
        elif rsi_val < 35.0:
            rsi_status = "Strefa wyprzedania (<35)"
        else:
            rsi_status = "Neutralny (>35)"

        # Kryterium 2: Poziom / Geometria
        # Dzisiejsze Low uderzyło w dolną wstęgę Bollingera LUB w strefę Fibo (0.500 / 0.618)
        bb_touched = bool(low_curr <= (bb_lower * 1.002))
        fibo_margin = self.config.get("indicators", {}).get("fibonacci", {}).get("margin_pct", 0.01)
        fibo_in_zone, fibo_level_key, fibo_target_price = indicators.is_in_fibo_support_zone(
            low_curr, close_curr, fibo_levels, margin_pct=fibo_margin
        )
        c2_level = bool(bb_touched or fibo_in_zone)

        bb_status = "Dotknięcie dolnej wstęgi (Wsparcie)" if bb_touched else "Wewnątrz wstęg"
        if fibo_in_zone and fibo_target_price is not None:
            fibo_status = f"Strefa {fibo_level_key} (~{fibo_target_price:.2f} EUR)"
        else:
            fibo_status = "Brak kluczowego wsparcia Fibo"

        # Kryterium 3: Świecowe (Price Action)
        # Młot (Pinbar) LUB Objęcie Hossy (Bullish Engulfing)
        is_hammer = indicators.detect_hammer(open_curr, high_curr, low_curr, close_curr)
        is_engulfing = indicators.detect_bullish_engulfing(open_prev, close_prev, open_curr, close_curr)
        c3_candle = bool(is_hammer or is_engulfing)

        if is_hammer and is_engulfing:
            candle_pattern = "Młot + Objęcie Hossy"
        elif is_hammer:
            candle_pattern = "Młot (Pinbar)"
        elif is_engulfing:
            candle_pattern = "Objęcie Hossy (Bullish Engulfing)"
        else:
            candle_pattern = "Brak formacji odwrócenia"

        # Kryterium 4: Wolumen
        # Wolumen absorpcyjny (RVol >= 1.4x średniej 20-dniowej)
        rvol_threshold = self.config.get("indicators", {}).get("rvol", {}).get("absorption_threshold", 1.4)
        c4_volume = bool(rvol >= rvol_threshold)

        # Kryterium 5: Impet (MACD)
        # Histogram MACD ujemny, ale dzisiejszy słupek jest krótszy od wczorajszego (wygaszanie pędu)
        # lub histogram wyszedł z ujemnego na dodatni
        c5_momentum = bool((hist_prev < 0 and hist_curr > hist_prev))

        if c5_momentum:
            macd_status = "Wygaszanie pędu podaży (histogram rośnie)"
        elif hist_curr > hist_prev:
            macd_status = "Wzrostowy histogram"
        else:
            macd_status = "Pogłębianie spadków (podaż aktywna)"

        # 4. SCORING KONFLUENCJI (MINIMUM 3 Z 5)
        criteria_list = [c1_oversold, c2_level, c3_candle, c4_volume, c5_momentum]
        score = sum(criteria_list)
        min_required = self.config.get("scoring", {}).get("min_criteria_required", 3)

        logger.debug(
            "[%s] Wynik konfluencji: %d/5 (Wyp=%s, Geometria=%s, Świeca=%s, Vol=%s, Impet=%s)",
            ticker, score, c1_oversold, c2_level, c3_candle, c4_volume, c5_momentum
        )

        if score < min_required:
            return None

        # 5. RISK MANAGEMENT
        sl_buffer_pct = self.config.get("risk_management", {}).get("sl_buffer_pct", 0.005)

        # Dołek formacji świecowej lub poziomu Fibo
        formation_low = min(low_curr, float(prev['Low'])) if is_engulfing else low_curr
        if fibo_in_zone and fibo_target_price is not None:
            support_low = min(formation_low, fibo_target_price)
        else:
            support_low = formation_low

        # Twardy Stop Loss: 0.5% poniżej dołka formacji / wsparcia Fibo
        sl_price = round(support_low * (1.0 - sl_buffer_pct), 2)
        # Ochrona przed SL >= Close (np. gdyby świeca zamknęła się na samym dnie)
        if sl_price >= close_curr:
            sl_price = round(close_curr * 0.98, 2)

        sl_pct = round(((close_curr - sl_price) / close_curr) * 100.0, 2)

        # Take Profit 1: Środkowa wstęga SMA 20 lub najbliższy opór
        tp1_price = round(bb_mid, 2)
        if tp1_price <= close_curr:
            # Jeżeli cena już jest powyżej SMA20, TP1 to poziom Fibo 0.382 lub BB Upper
            tp1_price = round(max(bb_upper, fibo_levels.get("0.382", close_curr * 1.03)), 2)

        tp1_pct = round(((tp1_price - close_curr) / close_curr) * 100.0, 2)

        # Take Profit 2: Poziom Fibo 0.0 (lokalny szczyt z 60 dni)
        tp2_price = round(fibo_levels["swing_high"], 2)
        if tp2_price <= close_curr:
            tp2_price = round(close_curr * 1.05, 2)

        tp2_pct = round(((tp2_price - close_curr) / close_curr) * 100.0, 2)

        # Stosunek Zysku do Ryzyka (Risk / Reward Ratio)
        risk = close_curr - sl_price
        reward_tp1 = tp1_price - close_curr
        reward_tp2 = tp2_price - close_curr

        rr_tp1 = (reward_tp1 / risk) if risk > 0 else 0.0
        rr_tp2 = (reward_tp2 / risk) if risk > 0 else 0.0
        rr_ratio = round(rr_tp2, 2)

        logger.info(
            "🎯 [%s] ZNALEZIONO SETUP KONFLUENCJI (Score: %d/5)! Close=%.2f, SL=%.2f (-%.2f%%), TP1=%.2f (+%.2f%%), TP2=%.2f (+%.2f%%), R:R=1:%.2f",
            ticker, score, close_curr, sl_price, sl_pct, tp1_price, tp1_pct, tp2_price, tp2_pct, rr_ratio
        )

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "close": close_curr,
            "change_pct": change_pct,
            "rsi_val": rsi_val,
            "rsi_status": rsi_status,
            "fibo_status": fibo_status,
            "bb_status": bb_status,
            "candle_pattern": candle_pattern,
            "rvol": rvol,
            "macd_status": macd_status,
            "sl_price": sl_price,
            "sl_pct": sl_pct,
            "tp1_price": tp1_price,
            "tp1_pct": tp1_pct,
            "tp2_price": tp2_price,
            "tp2_pct": tp2_pct,
            "rr_ratio": rr_ratio,
            "rr_tp1": rr_tp1,
            "score": score
        }

    def run_scan(self) -> List[Dict[str, Any]]:
        """
        Główna metoda wykonawcza:
        1. Wczytuje bazę tickerów.
        2. Pobiera dane zbiorczo.
        3. Iteruje i weryfikuje każdy ETF.
        4. Zwraca listę zakwalifikowanych setupów.
        """
        tickers_df = self.load_tickers()
        tickers_list = tickers_df['ticker'].tolist()

        data_dict = self.fetch_market_data(tickers_list)
        if not data_dict:
            logger.warning("Brak danych rynkowych do analizy.")
            return []

        qualified_setups: List[Dict[str, Any]] = []

        for _, row in tickers_df.iterrows():
            ticker = row['ticker']
            name = row['name']
            sector = row['sector']

            if ticker not in data_dict:
                continue

            df = data_dict[ticker]
            setup = self.evaluate_ticker(ticker, name, sector, df)
            if setup is not None:
                qualified_setups.append(setup)

        logger.info(
            "Zakończono skanowanie. Przeanalizowano %d instrumentów, zakwalifikowano %d setupów konfluencji.",
            len(data_dict), len(qualified_setups)
        )
        return qualified_setups
