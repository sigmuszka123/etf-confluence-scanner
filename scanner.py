"""
ETF Confluence Scanner - scanner.py
Główny moduł silnika skanera zintegrowany z generatorem wykresów i Audytorem AI.
"""

import os
import time
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import yaml

import indicators
from chart_generator import generate_candlestick_chart
from ai_auditor import AIAuditor
from notifier import TelegramNotifier

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance nie jest zainstalowane w bieżącym środowisku.")

try:
    from zoneinfo import ZoneInfo
    WARSAW_TZ = ZoneInfo("Europe/Warsaw")
except Exception:
    WARSAW_TZ = None


class ETFScanner:
    """
    Skaner rynku ETF/Stock wyszukujący setupy konfluencji Price Action + Wskaźniki.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._setup_logging()

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Plik konfiguracyjny {path} nie istnieje.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        log_level_str = log_cfg.get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_format = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logging.basicConfig(level=log_level, format=log_format)

    def load_tickers(self) -> pd.DataFrame:
        csv_file = self.config.get("data", {}).get("tickers_file", "tickers.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Brak pliku bazy tickerów: {csv_file}")

        df = pd.read_csv(csv_file)
        required_cols = {"ticker", "name", "sector"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Plik {csv_file} musi zawierać kolumny: {required_cols}")

        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["name"] = df["name"].astype(str).str.strip()
        df["sector"] = df["sector"].astype(str).str.strip()
        df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"])
        logger.info("Załadowano %d tickerów z pliku %s", len(df), csv_file)
        return df

    def _download_in_batches(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        batch_size: int,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, len(tickers), batch_size), 1):
            batch = tickers[start : start + batch_size]
            try:
                batch_df = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
            except Exception as e:
                logger.error("Błąd yf.download (%s) dla paczki %d/%d: %s", period, batch_idx, total_batches, e)
                continue

            if batch_df is None or batch_df.empty:
                continue

            min_rows = 1 if period == "5d" else 20
            for ticker in batch:
                ticker_df = self._extract_single_ticker_df(batch_df, ticker, min_rows=min_rows)
                if ticker_df is not None and not ticker_df.empty:
                    result[ticker] = ticker_df

            if batch_idx < total_batches:
                time.sleep(0.25)

        return result

    def fetch_market_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        if not HAS_YFINANCE:
            logger.error("Biblioteka yfinance jest niedostępna.")
            return {}

        data_cfg = self.config.get("data", {})
        period = data_cfg.get("period", "2y")
        # Zamiana niestandardowego '400d' na natywny okres Yahoo '2y' (unika bugów z ostatnią świecą)
        if period == "400d":
            period = "2y"
        interval = data_cfg.get("interval", "1d")
        batch_size = max(1, int(data_cfg.get("batch_size", 50)))

        logger.info(
            "Pobieranie danych w paczkach po %d dla %d tickerów (period=%s, interval=%s)...",
            batch_size,
            len(tickers),
            period,
            interval,
        )

        result = self._download_in_batches(tickers, period=period, interval=interval, batch_size=batch_size)

        if result:
            date_counts = Counter(pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d") for df in result.values())
            logger.info("Rozkład dat ostatnich świec (po %s): %s", period, dict(date_counts.most_common(5)))

        # Dogrywka (workaround na bug Yahoo/yfinance z NaN na ostatniej świecy dla długich okresów)
        stale_tickers = [t for t, df in result.items() if not self.is_data_fresh(df, t)]
        if stale_tickers and interval == "1d":
            logger.info(
                "Wykryto %d tickerów bez najnowszej świecy. Uruchamiam dogrywkę period='5d'...",
                len(stale_tickers),
            )
            recent_data = self._download_in_batches(
                stale_tickers, period="5d", interval=interval, batch_size=batch_size
            )
            patched = 0
            for t, df_recent in recent_data.items():
                df_hist = result[t]
                if df_recent.index[-1] > df_hist.index[-1]:
                    merged = pd.concat([df_hist, df_recent])
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                    result[t] = merged
                    patched += 1

            date_counts_after = Counter(pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d") for df in result.values())
            logger.info(
                "Dogrywka '5d' uzupełniła %d / %d tickerów. Rozkład dat po dogrywce: %s",
                patched,
                len(stale_tickers),
                dict(date_counts_after.most_common(5)),
            )

        logger.info("Pomyślnie przetworzono dane dla %d / %d tickerów.", len(result), len(tickers))
        return result

    @staticmethod
    def _extract_single_ticker_df(
        batch_df: pd.DataFrame,
        ticker: str,
        min_rows: int = 20,
    ) -> Optional[pd.DataFrame]:
        try:
            if isinstance(batch_df.columns, pd.MultiIndex):
                if ticker in batch_df.columns.levels[0]:
                    df_t = batch_df[ticker].copy()
                elif ticker in batch_df.columns.levels[1]:
                    df_t = batch_df.xs(ticker, axis=1, level=1).copy()
                else:
                    return None
            else:
                df_t = batch_df.copy()

            df_t.columns = [str(c).capitalize() for c in df_t.columns]
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df_t.columns for col in required):
                return None

            # Usuń niepełne świece (czasem Yahoo zwraca wiersz z NaN w Open/High/Low)
            df_t = df_t.dropna(subset=["Open", "High", "Low", "Close"])
            df_t = df_t[df_t["Close"] > 0]
            if len(df_t) < min_rows:
                return None

            # Ujednolicenie indeksu (bez przesunięcia daty przez konwersję do UTC)
            idx = pd.to_datetime(df_t.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            df_t.index = idx.normalize()
            df_t = df_t.sort_index()
            return df_t
        except Exception as err:
            logger.debug("Błąd ekstrakcji danych dla %s: %s", ticker, err)
            return None

    @staticmethod
    def is_data_fresh(df: pd.DataFrame, ticker: str) -> bool:
        """
        Sprawdza, czy ostatnia świeca odpowiada ostatniej zakończonej sesji giełdowej.
        Chroni przed cichym analizowaniem zlagowanych danych (np. piątku we wtorek rano).
        """
        if df is None or df.empty:
            return False

        last_candle_date = pd.Timestamp(df.index[-1]).date()
        now_cet = datetime.now(WARSAW_TZ) if WARSAW_TZ is not None else datetime.now()

        # Po 22:00 CET oczekujemy dzisiejszej sesji; w nocy/rano poprzedniego dnia roboczego
        expected_date = now_cet.date() if now_cet.hour >= 22 else (now_cet.date() - timedelta(days=1))
        while expected_date.weekday() >= 5:
            expected_date -= timedelta(days=1)

        if last_candle_date < expected_date:
            logger.debug(
                "[%s] Pominięto nieaktualne dane: ostatnia świeca %s (oczekiwano >= %s)",
                ticker,
                last_candle_date,
                expected_date,
            )
            return False
        return True

    @staticmethod
    def is_etf_instrument(ticker: str, name: str, sector: str) -> bool:
        """Rozpoznaje czy instrument jest funduszem ETF/ETC czy akcją pojedynczej spółki."""
        n = str(name).strip().upper()
        s = str(sector).strip().upper()
        etf_keywords = (
            "ETF", "UCITS", "ETC", "ETN", "ISHARES", "VANGUARD", "SPDR",
            "INVESCO", "XTRACKERS", "LYXOR", "AMUNDI", "WISDOMTREE",
            "VANECK", "GLOBAL X", "PROSHARES", "SCHWAB", "L&G",
        )
        if any(k in n for k in etf_keywords):
            return True
        if any(k in s for k in ("INDEX", "ETF", "COMMODITIES", "FACTORS")):
            return True
        return False

    def evaluate_ticker(
        self,
        ticker: str,
        name: str,
        sector: str,
        df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        liq_cfg = self.config.get("liquidity_filter", {})
        is_etf = self.is_etf_instrument(ticker, name, sector)

        # Adaptacyjny dobór progów płynności (specyfika ETF UCITS vs Akcje)
        if is_etf and "etfs_europe" in liq_cfg:
            min_vol_sma = liq_cfg["etfs_europe"].get("min_volume_sma_20", 3000)
            min_dollar_vol = liq_cfg["etfs_europe"].get("min_dollar_volume", 250000)
        else:
            min_vol_sma = liq_cfg.get("min_volume_sma_20", 30000)
            min_dollar_vol = liq_cfg.get("min_dollar_volume", 1000000)

        # 1. FILTR PŁYNNOŚCI
        vol_series = df["Volume"]
        if len(vol_series) < 20:
            return None

        sma_vol_20 = vol_series.rolling(20).mean().iloc[-1]
        if np.isnan(sma_vol_20) or sma_vol_20 < min_vol_sma:
            return None

        dollar_vol_20 = sma_vol_20 * float(df["Close"].iloc[-1])
        if dollar_vol_20 < min_dollar_vol:
            return None

        # 2. KALKULACJA WSKAŹNIKÓW
        df_ind = indicators.calculate_all_indicators(df, self.config)
        fibo_window = self.config.get("indicators", {}).get("fibonacci", {}).get("lookback_window", 60)
        fibo_levels = indicators.calculate_fibo_levels(df, window=fibo_window)

        curr = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        prev2 = df_ind.iloc[-3] if len(df_ind) >= 3 else prev
        last_bar_date = pd.Timestamp(df_ind.index[-1]).strftime("%Y-%m-%d")

        close_curr = float(curr["Close"])
        close_prev = float(prev["Close"])
        open_curr = float(curr["Open"])
        high_curr = float(curr["High"])
        low_curr = float(curr["Low"])
        open_prev = float(prev["Open"])
        change_pct = ((close_curr - close_prev) / close_prev) * 100.0

        # Ochrona przed załamaniem kursu spółki (Anti-Crash dla akcji):
        # Unikamy łapania spadającego noża na spółce po wynikach/newsach o skrajnym tąpnięciu (< -6%)
        if not is_etf and change_pct < -6.0:
            return None

        rsi_val = float(curr["RSI_14"])
        rsi_prev_val = float(prev["RSI_14"])
        rsi_prev2_val = float(prev2["RSI_14"])

        bb_mid = float(curr["BB_Middle"])
        bb_upper = float(curr["BB_Upper"])
        bb_lower = float(curr["BB_Lower"])

        rvol = float(curr["RVol"]) if not np.isnan(curr["RVol"]) else 1.0
        hist_curr = float(curr["MACD_Hist"])
        hist_prev = float(prev["MACD_Hist"])

        # Filtr ekstremalnego szoku wolumenowego (np. wyniki/dywidenda typu RVol 20x)
        risk_cfg = self.config.get("risk_management", {})
        max_rvol_shock = float(risk_cfg.get("max_rvol_shock", 5.0))
        if rvol > max_rvol_shock:
            return None

        # 3. EWALUACJA 5 KRYTERIÓW
        crossed_30_up = (rsi_prev_val < 30.0 and rsi_val >= 30.0) or (
            rsi_prev2_val < 30.0 and rsi_prev_val >= 30.0
        )
        c1_oversold = bool((rsi_val < 35.0) or crossed_30_up)
        rsi_status = (
            "Odbicie ponad 30"
            if crossed_30_up
            else (
                "Silnie wyprzedany (<30)"
                if rsi_val < 30.0
                else ("Wyprzedany (<35)" if rsi_val < 35.0 else "Neutralny (>35)")
            )
        )

        bb_touched = bool(low_curr <= (bb_lower * 1.002))
        fibo_margin = self.config.get("indicators", {}).get("fibonacci", {}).get("margin_pct", 0.01)
        fibo_in_zone, fibo_level_key, fibo_target_price = indicators.is_in_fibo_support_zone(
            low_curr, close_curr, fibo_levels, margin_pct=fibo_margin
        )
        c2_level = bool(bb_touched or fibo_in_zone)

        currency = TelegramNotifier.detect_currency(ticker)
        bb_status = "Dotknięcie dolnej wstęgi" if bb_touched else "Wewnątrz wstęg"
        fibo_status = (
            f"Strefa {fibo_level_key} (~{fibo_target_price:.2f} {currency})"
            if (fibo_in_zone and fibo_target_price)
            else "Brak wsparcia Fibo"
        )

        is_hammer = indicators.detect_hammer(open_curr, high_curr, low_curr, close_curr)
        is_engulfing = indicators.detect_bullish_engulfing(open_prev, close_prev, open_curr, close_curr)
        c3_candle = bool(is_hammer or is_engulfing)
        candle_pattern = (
            "Młot + Objęcie Hossy"
            if (is_hammer and is_engulfing)
            else (
                "Młot (Pinbar)"
                if is_hammer
                else ("Objęcie Hossy" if is_engulfing else "Brak formacji odwrócenia")
            )
        )

        rvol_threshold = self.config.get("indicators", {}).get("rvol", {}).get("absorption_threshold", 1.4)
        c4_volume = bool(rvol >= rvol_threshold)
        c5_momentum = bool(hist_prev < 0 and hist_curr > hist_prev)
        macd_status = (
            "Wygaszanie pędu podaży"
            if c5_momentum
            else ("Wzrostowy histogram" if hist_curr > hist_prev else "Pogłębianie spadków")
        )

        score = sum([c1_oversold, c2_level, c3_candle, c4_volume, c5_momentum])
        min_required = self.config.get("scoring", {}).get("min_criteria_required", 3)
        if score < min_required:
            return None

        # 4. TWARDE FILTRY RYZYKA
        use_atr_sl = risk_cfg.get("use_atr_sl", True)
        atr_sl_mult = float(risk_cfg.get("atr_sl_multiplier", 2.0))
        sl_buffer_pct = float(risk_cfg.get("sl_buffer_pct", 0.005))
        max_sl_pct = float(risk_cfg.get("max_sl_distance_pct", 8.5))
        max_candle_pct = float(risk_cfg.get("max_candle_change_pct", 3.0))
        min_rr_tp1 = float(risk_cfg.get("min_rr_to_tp1", 0.60))
        min_rr_tp2 = float(risk_cfg.get("min_rr_to_tp2", 1.40))
        req_uptrend = risk_cfg.get("require_uptrend_sma200", False)

        atr_val = (
            float(curr["ATR_14"])
            if "ATR_14" in curr and not np.isnan(curr["ATR_14"])
            else float((df["High"] - df["Low"]).tail(14).mean())
        )
        atr_sl_price = round(max(0.01, close_curr - (atr_sl_mult * atr_val)), 2)
        atr_sl_pct = round(((close_curr - atr_sl_price) / close_curr) * 100.0, 2) if close_curr > 0 else 0.0

        formation_low = min(low_curr, float(prev["Low"])) if is_engulfing else low_curr
        support_low = min(formation_low, fibo_target_price) if (fibo_in_zone and fibo_target_price) else formation_low
        structural_sl = round(support_low * (1.0 - sl_buffer_pct), 2)

        sl_price = atr_sl_price if use_atr_sl else structural_sl
        if sl_price >= close_curr:
            sl_price = round(close_curr * 0.98, 2)
        sl_pct = round(((close_curr - sl_price) / close_curr) * 100.0, 2)

        # Filtr SL
        if sl_pct > max_sl_pct or sl_pct < 0.8:
            return None

        # Filtr wielkości świecy (Anti-Chasing)
        candle_body_pct = ((close_curr - open_curr) / open_curr) * 100.0 if open_curr > 0 else 0.0
        large_candle_pct = max(change_pct, candle_body_pct)
        is_large_candle = bool(large_candle_pct > 3.5)
        if large_candle_pct > max_candle_pct:
            return None

        risk = close_curr - sl_price
        if risk <= 0:
            return None

        # Geometria TP1 / TP2
        if close_curr >= bb_mid:
            tp1_price = round(bb_upper, 2)
        else:
            tp1_price = round(bb_mid, 2)

        tp2_price = round(max(fibo_levels["swing_high"], bb_upper * 1.02), 2)
        if tp2_price <= tp1_price:
            tp2_price = round(tp1_price + 1.0 * risk, 2)

        tp1_pct = round(((tp1_price - close_curr) / close_curr) * 100.0, 2)
        tp2_pct = round(((tp2_price - close_curr) / close_curr) * 100.0, 2)

        rr_tp1 = (tp1_price - close_curr) / risk
        rr_tp2 = (tp2_price - close_curr) / risk
        rr_ratio = round(rr_tp2, 2)

        if rr_tp1 < min_rr_tp1 or rr_tp2 < min_rr_tp2:
            return None

        sma_200 = float(curr["SMA_200"]) if "SMA_200" in curr and not np.isnan(curr["SMA_200"]) else None
        is_uptrend = bool(close_curr > sma_200) if sma_200 is not None else None
        if req_uptrend and is_uptrend is False:
            return None

        trend_status = (
            "✅ HOSSA (>SMA200)"
            if is_uptrend
            else ("⚠️ BESSA (<SMA200)" if is_uptrend is False else "ℹ️ BRAK SMA200")
        )

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "asset_type": "ETF" if is_etf else "Equity",
            "currency": currency,
            "last_bar_date": last_bar_date,
            "close": close_curr,
            "change_pct": change_pct,
            "trend_status": trend_status,
            "is_uptrend": is_uptrend,
            "sma_200": sma_200,
            "is_large_candle": is_large_candle,
            "large_candle_pct": round(large_candle_pct, 2),
            "atr_val": round(atr_val, 2),
            "atr_sl_price": atr_sl_price,
            "atr_sl_pct": atr_sl_pct,
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
            "score": score,
        }

    def run_scan(self) -> List[Dict[str, Any]]:
        """
        Główna pętla wykonawcza:
        1. Wczytuje bazę tickerów.
        2. Pobiera dane historyczne w paczkach (+ dogrywka 5d dla zlagowanych świec).
        3. Weryfikuje świeżość danych (brak cichego cofania do starej sesji).
        4. Filtruje matematycznie przez evaluate_ticker.
        5. Renderuje wykres PNG (prawdziwa SMA200, poziomy SL/TP1) i pyta Agenta AI.
        6. Zwraca wyłącznie zatwierdzone setupy i sprząta odrzucone wykresy PNG.
        """
        tickers_df = self.load_tickers()
        tickers_list = tickers_df["ticker"].tolist()

        data_dict = self.fetch_market_data(tickers_list)
        if not data_dict:
            logger.warning("Brak danych rynkowych do analizy.")
            return []

        auditor = AIAuditor()
        qualified_setups: List[Dict[str, Any]] = []
        stale_count = 0

        for _, row in tickers_df.iterrows():
            ticker = row["ticker"]
            name = row["name"]
            sector = row["sector"]

            if ticker not in data_dict:
                continue

            df = data_dict[ticker]

            # 0. Weryfikacja świeżości ostatniej świecy
            if not self.is_data_fresh(df, ticker):
                stale_count += 1
                continue

            setup = self.evaluate_ticker(ticker, name, sector, df)
            if setup is None:
                continue

            # 1. Wygeneruj wykres świecowy PNG (z poziomami SL/TP1 z setupu)
            safe_ticker = ticker.replace("/", "_").replace("^", "")
            chart_file = f"chart_{safe_ticker}.png"
            try:
                generate_candlestick_chart(df, ticker, output_path=chart_file, setup=setup)
                setup["chart_path"] = chart_file
            except Exception as chart_err:
                logger.error("Błąd generowania wykresu dla %s: %s", ticker, chart_err)
                setup["chart_path"] = None

            # 2. Audyt AI (jeśli wykres istnieje)
            if setup.get("chart_path"):
                ai_verdict = auditor.audit_setup(setup["chart_path"], setup)
            else:
                ai_verdict = {
                    "verdict": "BYPASS",
                    "veto_rule": "NONE",
                    "approval_reasons": [],
                    "actionable_plan": "Brak wykresu do audytu.",
                }

            setup["ai_review"] = ai_verdict
            verdict = ai_verdict.get("verdict")
            bar_dt = setup.get("last_bar_date", "?")

            # 3. Selekcja: przepuść jeśli PASS lub awaria konfiguracji (BYPASS)
            if verdict in ["PASS", "BYPASS"]:
                qualified_setups.append(setup)
                logger.info(
                    "✅ [%s | %s] ZATWIERDZONY PRZEZ AI (Werdykt: %s, Score: %s/10 | Za: %s)",
                    ticker,
                    bar_dt,
                    verdict,
                    ai_verdict.get("confidence_score", "N/A"),
                    ai_verdict.get("approval_reasons", []),
                )
            elif verdict == "ERROR":
                logger.error(
                    "⚠️ [%s | %s] BŁĄD AUDYTU AI: %s",
                    ticker,
                    bar_dt,
                    ai_verdict.get("actionable_plan"),
                )
                self._cleanup_chart(setup.get("chart_path"))
            else:
                logger.warning(
                    "🚫 [%s | %s] ODRZUCONY PRZEZ AUDYTORA AI (VETO: %s | Struktura: %s | Powody: %s)",
                    ticker,
                    bar_dt,
                    ai_verdict.get("veto_rule", "N/A"),
                    ai_verdict.get("market_structure"),
                    ai_verdict.get("devil_advocate_reasons"),
                )
                self._cleanup_chart(setup.get("chart_path"))

        if stale_count > 0:
            logger.info("Pominięto %d tickerów z nieaktualną ostatnią świecą (lag/święto).", stale_count)

        logger.info(
            "Zakończono skanowanie. Zakwalifikowano i zweryfikowano przez AI: %d setupów.",
            len(qualified_setups),
        )
        return qualified_setups

    @staticmethod
    def _cleanup_chart(chart_path: Optional[str]) -> None:
        if chart_path and os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except OSError as err:
                logger.debug("Nie udało się usunąć wykresu %s: %s", chart_path, err)
