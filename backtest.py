"""
========================================================================================
ETF Confluence Scanner - backtest.py
========================================================================================
Profesjonalny silnik testowania wstecznego (Walk-Forward Day-by-Day Backtester)
dla systemu ETF/Stock Confluence Scanner.

Główne cechy architektury:
1. Brak Lookahead Bias: Obliczenia wskaźników bazują wyłącznie na informacjach
   dostępnych do dnia T włącznie (causal rolling windows).
2. Realistyczna egzekucja: Wejście w pozycję po cenie Open w dniu T+1.
3. Zaawansowane zarządzanie pozycją:
   - Twardy Stop Loss pod dołkiem formacji/wsparciem Fibo.
   - TP1 (50% pozycji) na poziomie SMA 20 w dniu wejścia -> przestawienie SL na Break-Even.
   - TP2 (pozostałe 50%) na 60-sesyjnym szczycie (Fibo 0.0).
   - Konserwatywna reguła Intraday: przy kolizji Low <= SL oraz High >= TP1 zawsze SL.
   - Time Stop: Zamykanie z ręki po cenie Close 20. sesji.
4. Lokalne cache'owanie danych (data_cache/): Zbiorcze pobieranie przez yfinance
   oraz odczyt z dysku przy kolejnych uruchomieniach (ochrona przed Rate Limit).
5. Pełne raportowanie ilościowe (Win Rate, R-Multiple, Profit Factor, Max Drawdown, EV).
========================================================================================
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# Import lokalnego modułu wskaźników i skanera
import indicators
from scanner import ETFScanner

# Opcjonalny import yfinance z bezpiecznym fallbackiem
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logger = logging.getLogger("ConfluenceBacktester")


# ======================================================================================
# STRUKTURA DANYCH TRANSAKCJI (TRADE OBJECT)
# ======================================================================================

@dataclass
class Trade:
    """Reprezentacja pojedynczej transakcji w silniku symulacji."""
    ticker: str
    name: str
    sector: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    initial_sl: float
    risk: float                 # Entry - Initial SL (wartość 1.0 R w walucie)
    tp1_price: float
    tp2_price: float
    score: int
    rvol: float = 0.0
    rsi_14: float = 50.0
    asset_type: str = "ETF"     # 'ETF' lub 'Equity'
    
    # Stan i wynik transakcji
    exit_date: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    exit_reason: str = ""       # 'SL', 'SL (Gap)', 'TP1+TP2', 'TP1+BE', 'TP1+BE (Gap)', 'TimeStop', 'TP1+TimeStop'
    holding_days: int = 0
    tp1_hit: bool = False
    tp1_date: Optional[pd.Timestamp] = None
    r_multiple: float = 0.0     # Wynik końcowy w jednostkach R (po kosztach)


# ======================================================================================
# METRYKI I RAPORT WYDAJNOŚCIOWY (QUANT METRICS)
# ======================================================================================

@dataclass
class BacktestMetrics:
    """Zestawienie kluczowych metryk ilościowych modelu."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    win_rate_pct: float = 0.0
    loss_rate_pct: float = 0.0
    
    avg_r_multiple: float = 0.0
    expected_value_r: float = 0.0
    
    gross_profit_r: float = 0.0
    gross_loss_r: float = 0.0
    profit_factor: float = 0.0
    
    max_drawdown_pct: float = 0.0
    max_drawdown_r: float = 0.0
    
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    payoff_ratio: float = 0.0
    
    best_trade_r: float = 0.0
    worst_trade_r: float = 0.0
    avg_holding_days: float = 0.0
    
    # Rozkład powodów zamknięcia
    exit_counts: Dict[str, int] = field(default_factory=dict)
    
    # Rozbicie wg klas aktywów (ETF vs Akcje) oraz wg lat (Out-of-Sample / Stabilność w czasie)
    by_asset_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_year: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class BacktestResult:
    """Kontener wyników symulacji z metodami analitycznymi i prezentacyjnymi."""

    def __init__(self, trades: List[Trade], initial_capital: float = 100000.0, risk_per_trade_pct: float = 0.01):
        self.trades = sorted(trades, key=lambda t: t.exit_date if t.exit_date is not None else t.entry_date)
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> BacktestMetrics:
        m = BacktestMetrics()
        m.total_trades = len(self.trades)
        if m.total_trades == 0:
            return m

        r_values = np.array([t.r_multiple for t in self.trades], dtype=float)
        holding_days = np.array([t.holding_days for t in self.trades], dtype=float)

        wins = r_values[r_values > 0.0]
        losses = r_values[r_values < 0.0]
        bes = r_values[r_values == 0.0]

        m.winning_trades = len(wins)
        m.losing_trades = len(losses)
        m.breakeven_trades = len(bes)

        # 1. Liczba setupów (N)
        # 2. Skuteczność (Win Rate %)
        m.win_rate_pct = (m.winning_trades / m.total_trades) * 100.0
        m.loss_rate_pct = (m.losing_trades / m.total_trades) * 100.0

        # 3. Średni R-Multiple (R)
        m.avg_r_multiple = float(np.mean(r_values))

        # 4. Profit Factor
        m.gross_profit_r = float(np.sum(wins)) if len(wins) > 0 else 0.0
        m.gross_loss_r = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0

        if m.gross_loss_r > 0.0:
            m.profit_factor = m.gross_profit_r / m.gross_loss_r
        else:
            m.profit_factor = float("inf") if m.gross_profit_r > 0 else 0.0

        # 5. Maksymalne obsunięcie kapitału (Max Drawdown %)
        # Symulacja krzywej kapitału z zarządzaniem ryzykiem (np. 1% kapitału na transakcję)
        equity = self.initial_capital
        peak_equity = equity
        max_dd_pct = 0.0

        cum_r = 0.0
        peak_r = 0.0
        max_dd_r = 0.0

        for r in r_values:
            # Aktualizacja kapitału portfela
            trade_pnl = equity * self.risk_per_trade_pct * r
            equity += trade_pnl
            if equity > peak_equity:
                peak_equity = equity
            dd_pct = ((peak_equity - equity) / peak_equity) * 100.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            # Drawdown w jednostkach R
            cum_r += r
            if cum_r > peak_r:
                peak_r = cum_r
            dd_r = peak_r - cum_r
            if dd_r > max_dd_r:
                max_dd_r = dd_r

        m.max_drawdown_pct = max_dd_pct
        m.max_drawdown_r = max_dd_r

        # 6. Oczekiwana Wartość Transakcji (Expected Value per Trade w R)
        m.avg_win_r = float(np.mean(wins)) if len(wins) > 0 else 0.0
        m.avg_loss_r = float(np.mean(losses)) if len(losses) > 0 else 0.0
        m.payoff_ratio = (m.avg_win_r / abs(m.avg_loss_r)) if abs(m.avg_loss_r) > 0 else 0.0

        # Formuła EV: (P_win * Avg_Win) - (P_loss * Avg_Loss) = E[R]
        m.expected_value_r = m.avg_r_multiple

        m.best_trade_r = float(np.max(r_values))
        m.worst_trade_r = float(np.min(r_values))
        m.avg_holding_days = float(np.mean(holding_days))

        # Rozkład powodów wyjścia
        for t in self.trades:
            reason = t.exit_reason or "Inne"
            m.exit_counts[reason] = m.exit_counts.get(reason, 0) + 1

        # Rozbicie na klasy aktywów (ETF vs Akcje)
        all_types = sorted(set(getattr(t, "asset_type", "ETF") for t in self.trades))
        for atype in all_types:
            sub = [t for t in self.trades if getattr(t, "asset_type", "ETF") == atype]
            sub_r = [t.r_multiple for t in sub]
            sub_w = len([r for r in sub_r if r > 0])
            gp = sum(r for r in sub_r if r > 0)
            gl = abs(sum(r for r in sub_r if r < 0))
            pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
            m.by_asset_type[atype] = {
                "n": len(sub),
                "win_rate": (sub_w / len(sub)) * 100.0 if sub else 0.0,
                "avg_r": float(np.mean(sub_r)) if sub else 0.0,
                "profit_factor": pf
            }

        # Rozbicie w czasie / Analiza Out-of-Sample (wg lat wejścia)
        all_years = sorted(set(t.entry_date.year for t in self.trades if t.entry_date is not None))
        for y in all_years:
            sub_y = [t for t in self.trades if t.entry_date.year == y]
            sub_y_r = [t.r_multiple for t in sub_y]
            sub_y_w = len([r for r in sub_y_r if r > 0])
            gp = sum(r for r in sub_y_r if r > 0)
            gl = abs(sum(r for r in sub_y_r if r < 0))
            pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
            m.by_year[y] = {
                "n": len(sub_y),
                "win_rate": (sub_y_w / len(sub_y)) * 100.0 if sub_y else 0.0,
                "avg_r": float(np.mean(sub_y_r)) if sub_y else 0.0,
                "profit_factor": pf
            }

        return m

    def print_summary(self, elapsed_sec: float = 0.0, total_tickers: int = 0) -> None:
        """Drukuje w konsoli czytelny, profesjonalny raport w formacie Quant Summary."""
        m = self.metrics
        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "INF (Brak strat)"

        print("\n" + "=" * 88)
        print("      ETF & EQUITY CONFLUENCE SCANNER — RAPORT TESTOWANIA WSTECZNEGO (1:1 SCANNER)")
        print("=" * 88)
        print(f" Liczba przeanalizowanych tickerów:  {total_tickers:,}")
        print(f" Czas wykonania symulacji:           {elapsed_sec:.2f} s")
        print(f" Model egzekucji i zarządzania:      Wejście @ Open(T+1) | SL pod formacją/Fibo | TP1 @ SMA20/UpperBB | TP2 @ 60d High")
        print("-" * 88)
        print(" GŁÓWNE METRYKI ILOŚCIOWE (KLUCZOWE WYMAGANIA PROJEKTU):")
        print("-" * 88)
        print(f"  1. Liczba wygenerowanych setupów (N):          {m.total_trades:>8d}")
        print(f"  2. Skuteczność (Win Rate %):                   {m.win_rate_pct:>8.2f} %  ({m.winning_trades}W / {m.losing_trades}L / {m.breakeven_trades}BE)")
        print(f"  3. Średni R-Multiple (R):                      {m.avg_r_multiple:>+8.2f} R")
        print(f"  4. Profit Factor:                              {pf_str:>8s}")
        print(f"  5. Maksymalne obsunięcie kapitału (Max DD %):  {m.max_drawdown_pct:>8.2f} %  (Max DD: -{m.max_drawdown_r:.2f} R)")
        print(f"  6. Oczekiwana Wartość Transakcji (EV w R):     {m.expected_value_r:>+8.2f} R  / transakcję")
        print("-" * 88)
        print(" SZCZEGÓŁOWA ANALIZA STATYSTYCZNA I ROZKŁAD ZAMKNIĘĆ:")
        print("-" * 88)
        print(f"  • Średni zysk (Avg Win):                       {m.avg_win_r:>+8.2f} R")
        print(f"  • Średnia strata (Avg Loss):                   {m.avg_loss_r:>+8.2f} R")
        print(f"  • Stosunek Zysk/Strata (Payoff Ratio):         {m.payoff_ratio:>8.2f}")
        print(f"  • Najlepsza transakcja (Best Trade):           {m.best_trade_r:>+8.2f} R")
        print(f"  • Najgorsza transakcja (Worst Trade):          {m.worst_trade_r:>+8.2f} R")
        print(f"  • Średni czas trwania pozycji (Avg Hold):      {m.avg_holding_days:>8.1f} sesji")
        print("  • Rozkład powodów zamknięcia pozycji:")
        for reason, count in sorted(m.exit_counts.items(), key=lambda x: -x[1]):
            pct = (count / m.total_trades) * 100.0 if m.total_trades > 0 else 0.0
            print(f"      - {reason:<20s}: {count:>5d} ({pct:>5.1f} %)")

        if m.by_asset_type:
            print("-" * 88)
            print(" WYNIKI WG KLASY AKTYWÓW (ETF UCITS VS AKCJE INDYWIDUALNE):")
            print("-" * 88)
            for atype, stats in m.by_asset_type.items():
                pf_s = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float("inf") else "INF"
                print(f"  • {atype:<15s}: N={stats['n']:>4d} | Win Rate: {stats['win_rate']:>5.1f}% | Avg R: {stats['avg_r']:>+5.2f} R | Profit Factor: {pf_s:>5s}")

        if m.by_year:
            print("-" * 88)
            print(" STABILNOŚĆ W CZASIE (ROZKŁAD ROCZNY / ANALIZA OUT-OF-SAMPLE):")
            print("-" * 88)
            for year, stats in m.by_year.items():
                pf_s = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float("inf") else "INF"
                print(f"  • Rok {year}:         N={stats['n']:>4d} | Win Rate: {stats['win_rate']:>5.1f}% | Avg R: {stats['avg_r']:>+5.2f} R | Profit Factor: {pf_s:>5s}")
        print("=" * 88 + "\n")

    def to_dataframe(self) -> pd.DataFrame:
        """Eksportuje listę transakcji do pandas DataFrame."""
        records = []
        for t in self.trades:
            records.append({
                "ticker": t.ticker,
                "name": t.name,
                "sector": t.sector,
                "score": t.score,
                "signal_date": t.signal_date,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "sl_price": t.initial_sl,
                "risk": t.risk,
                "tp1_price": t.tp1_price,
                "tp2_price": t.tp2_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "holding_days": t.holding_days,
                "tp1_hit": t.tp1_hit,
                "r_multiple": t.r_multiple
            })
        return pd.DataFrame(records)


# ======================================================================================
# SILNIK TESTOWANIA WSTECZNEGO (CONFLUENCE BACKTESTER)
# ======================================================================================

class ConfluenceBacktester:
    """
    Silnik symulacji Walk-Forward dla ETF Confluence Scanner.
    Realizuje bezbłędną causalityczną analizę day-by-day bez Lookahead Bias.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        cache_dir: str = "data_cache",
        tickers_file: str = "tickers.csv",
        min_score: int = 4,
        time_stop: int = 20,
        sl_buffer_pct: float = 0.005,
        lookback_window: int = 60,
        min_volume_sma_20: float = 5000.0,
        cache_format: str = "csv",
        fee_pct: float = 0.0005,
        slippage_pct: float = 0.0005,
        min_price: float = 2.0,
        min_risk_pct: float = 0.015,
        max_open_positions: Optional[int] = 10,
        max_per_sector: Optional[int] = 2,
        max_daily_entries: Optional[int] = 3
    ):
        self.config_path = config_path
        self.cache_dir = cache_dir
        self.tickers_file = tickers_file
        self.min_score = min_score
        self.time_stop = time_stop
        self.sl_buffer_pct = sl_buffer_pct
        self.lookback_window = lookback_window
        self.min_volume_sma_20 = min_volume_sma_20
        self.cache_format = cache_format.lower()
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.min_price = min_price
        self.min_risk_pct = min_risk_pct
        self.max_open_positions = max_open_positions
        self.max_per_sector = max_per_sector
        self.max_daily_entries = max_daily_entries
        
        # Wczytanie konfiguracji jeśli dostępna
        self.config = self._load_config()
        self._apply_config_overrides()

    def _load_config(self) -> Dict[str, Any]:
        """Wczytuje config.yaml jeśli istnieje na dysku."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Nie udało się odczytać %s: %s", self.config_path, e)
        return {}

    def _apply_config_overrides(self) -> None:
        """Aktualizuje domyślne parametry na podstawie config.yaml."""
        if not self.config:
            return
        
        # Scoring
        cfg_score = self.config.get("scoring", {}).get("min_criteria_required")
        if cfg_score is not None:
            self.min_score = int(cfg_score)

        # Risk
        cfg_risk = self.config.get("risk_management", {})
        if "sl_buffer_pct" in cfg_risk:
            self.sl_buffer_pct = float(cfg_risk["sl_buffer_pct"])

        # Liquidity
        cfg_liq = self.config.get("liquidity_filter", {})
        if "min_volume_sma_20" in cfg_liq:
            self.min_volume_sma_20 = float(cfg_liq["min_volume_sma_20"])

        # Fibo window
        cfg_fibo = self.config.get("indicators", {}).get("fibonacci", {})
        if "lookback_window" in cfg_fibo:
            self.lookback_window = int(cfg_fibo["lookback_window"])

    def load_tickers(self) -> pd.DataFrame:
        """Wczytuje listę instrumentów z pliku tickers.csv."""
        if not os.path.exists(self.tickers_file):
            raise FileNotFoundError(f"Plik listy tickerów nie istnieje: {self.tickers_file}")

        df = pd.read_csv(self.tickers_file)
        # Normalizacja kolumn
        cols = {c.lower().strip(): c for c in df.columns}
        if "ticker" not in cols:
            raise ValueError(f"Plik {self.tickers_file} musi zawierać kolumnę 'ticker'.")

        ticker_col = cols["ticker"]
        name_col = cols.get("name", ticker_col)
        sector_col = cols.get("sector", ticker_col)

        clean_df = pd.DataFrame({
            "ticker": df[ticker_col].astype(str).str.strip(),
            "name": df[name_col].astype(str).str.strip() if name_col in df else df[ticker_col],
            "sector": df[sector_col].astype(str).str.strip() if sector_col in df else "General"
        })
        clean_df = clean_df[clean_df["ticker"] != ""].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        return clean_df

    # ----------------------------------------------------------------------------------
    # CACHE & POBIERANIE DANYCH RYNKOWYCH
    # ----------------------------------------------------------------------------------

    def load_or_fetch_data(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        batch_size: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Główna metoda pobierania danych:
        1. Najpierw sprawdza lokalny cache w folderze data_cache/.
        2. Brakujące instrumenty pobiera zbiorczo przez yfinance (Batch Download).
        3. W przypadku braku yfinance / blokady sieci, generuje realistyczne syntetyczne dane
           rynkowe i zapisuje do cache, gwarantując 100% stabilności.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        market_data: Dict[str, pd.DataFrame] = {}
        missing_tickers: List[str] = []

        # 1. Odczyt z dysku
        rejected_sanity = 0
        for ticker in tickers:
            df = self._read_from_cache(ticker)
            if df is not None and not df.empty and len(df) >= 30:
                if self._is_data_sane(df, ticker):
                    market_data[ticker] = df
                else:
                    rejected_sanity += 1
            else:
                missing_tickers.append(ticker)

        if not missing_tickers:
            logger.info("Załadowano %d tickerów z lokalnego cache: %s (odrzucono uszkodzonych: %d)",
                        len(market_data), self.cache_dir, rejected_sanity)
            return market_data

        logger.info("Załadowano %d z dysku. Próba pobrania z Yahoo Finance dla %d tickerów...",
                    len(market_data), len(missing_tickers))

        # 2. Pobieranie z yfinance (jeśli dostępne)
        if HAS_YFINANCE:
            downloaded = self._download_from_yfinance(missing_tickers, period=period, interval=interval, batch_size=batch_size)
            for ticker, df in downloaded.items():
                if df is not None and len(df) >= 30:
                    if self._is_data_sane(df, ticker):
                        self._save_to_cache(ticker, df)
                        market_data[ticker] = df
                    else:
                        rejected_sanity += 1
        else:
            logger.warning("Biblioteka yfinance nie jest dostępna. Pominięto pobieranie dla %d tickerów.",
                           len(missing_tickers))

        # 3. Pominięcie tickerów, których nie udało się pobrać z Yahoo Finance (brak syntetycznego fallbacku)
        still_missing = [t for t in missing_tickers if t not in market_data]
        if still_missing:
            logger.info("Pominięto %d instrumentów (brak danych w Yahoo Finance lub odrzucone przez Data Sanity Filter).",
                        len(still_missing))

        return market_data

    @staticmethod
    def _is_data_sane(df: pd.DataFrame, ticker: str, max_daily_jump_pct: float = 0.40) -> bool:
        """
        Data Sanity Filter:
        Weryfikuje integralność danych OHLCV i wykrywa błędy bazy Yahoo Finance,
        takie jak przeskoki waluty 100x (pensy <-> funty na giełdzie w Londynie .L, np. NVT.L)
        lub nieskorygowane splity.
        Jeśli między kolejnymi dniami cena spada lub rośnie o więcej niż 40%,
        cały ticker zostaje odrzucony jako uszkodzone dane.
        """
        if df is None or len(df) < 2:
            return False

        if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
            logger.warning("[%s] Odrzucono przez Data Sanity Filter: wykryto ceny <= 0.", ticker)
            return False

        prev_close = df["Close"].shift(1)
        close_ret = (df["Close"] / prev_close) - 1.0
        open_ret = (df["Open"] / prev_close) - 1.0

        max_close_jump = float(close_ret.abs().max()) if not close_ret.dropna().empty else 0.0
        max_open_jump = float(open_ret.abs().max()) if not open_ret.dropna().empty else 0.0
        max_jump = max(max_close_jump, max_open_jump)

        if max_jump > max_daily_jump_pct:
            logger.warning(
                "[%s] Odrzucono przez Data Sanity Filter: wykryto nienaturalny skok/spadek ceny (%.1f%% > %.0f%%, możliwy błąd waluty 100x).",
                ticker, max_jump * 100.0, max_daily_jump_pct * 100.0
            )
            return False

        return True

    def _read_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Wczytuje DataFrame pojedynczego tickera z pliku CSV lub Parquet w data_cache/."""
        # Bezpieczna nazwa pliku
        safe_name = ticker.replace("/", "_")
        csv_path = os.path.join(self.cache_dir, f"{safe_name}.csv")
        parquet_path = os.path.join(self.cache_dir, f"{safe_name}.parquet")

        try:
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                return self._normalize_dataframe(df)
            elif os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                return self._normalize_dataframe(df)
        except Exception as e:
            logger.debug("Błąd odczytu cache dla %s: %s", ticker, e)
        return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """Zapisuje DataFrame pojedynczego tickera do lokalnego folderu cache."""
        safe_name = ticker.replace("/", "_")
        csv_path = os.path.join(self.cache_dir, f"{safe_name}.csv")
        try:
            df.to_csv(csv_path)
        except Exception as e:
            logger.warning("Błąd zapisu cache dla %s: %s", ticker, e)

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalizuje nazwy kolumn i indeks czasowy."""
        if df is None or df.empty:
            return None
        df = df.copy()
        df.columns = [str(c).capitalize() for c in df.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return None
        df = df[required].dropna(subset=["Close"]).sort_index()
        # Konwersja na typy float
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        return df if len(df) >= 20 else None

    def _download_from_yfinance(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        batch_size: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """Pobiera dane paczkami za pomocą yfinance."""
        result: Dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), batch_size):
            chunk = tickers[i:i + batch_size]
            try:
                batch_df = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False
                )
                if batch_df is None or batch_df.empty:
                    continue

                for t in chunk:
                    try:
                        if isinstance(batch_df.columns, pd.MultiIndex):
                            if t in batch_df.columns.levels[0]:
                                df_t = batch_df[t].copy()
                            elif t in batch_df.columns.levels[1]:
                                df_t = batch_df.xs(t, axis=1, level=1).copy()
                            else:
                                continue
                        else:
                            df_t = batch_df.copy()

                        norm = self._normalize_dataframe(df_t)
                        if norm is not None:
                            result[t] = norm
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Błąd paczki yfinance: %s", e)
        return result

    # ----------------------------------------------------------------------------------
    # WALK-FORWARD SIMULATION ENGINE (DAY-BY-DAY, CAUSAL)
    # ----------------------------------------------------------------------------------

    def prepare_causal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kalkuluje kompletny zestaw wskaźników z indicators.py w sposób wektoryzowany
        oraz dodatkowe wskaźniki eksperymentu v3: RSI(2) i ATR(14).
        Operacje rolling/ewm(adjust=False) są ściśle przyczynowe (causal), co matematycznie
        eliminuje Lookahead Bias.
        """
        ind_df = indicators.calculate_all_indicators(df, self.config)

        # RSI(2) do krótkoterminowego wyprzedania
        ind_df["RSI_2"] = indicators.calculate_rsi(df["Close"], window=2)

        # ATR(14) według formuły Wildera (ściśle przyczynowy)
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        ind_df["ATR_14"] = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

        return ind_df

    def simulate_ticker(
        self,
        ticker: str,
        name: str,
        sector: str,
        df: pd.DataFrame
    ) -> List[Trade]:
        """
        Przeprowadza symulację Walk-Forward Day-by-Day dla pojedynczego instrumentu (v3: Mean Reversion w Hossie).
        
        Logika decyzyjna w dniu sesyjnym T:
        1. Filtr płynności: SMA(Volume, 20) >= min_volume_sma_20.
        2. Filtr Trendu Głównego: Close > SMA(200).
        3. Warunek Wyprzedania: RSI(14) < 38 LUB Close < BB_Lower(20, 2) LUB RSI(2) < 15.
        4. Anti-Chasing: (Close - Open) / Open < 0.025.
        5. Jeśli wszystkie warunki spełnione -> Wejście @ Open(T+1):
           - SL = Entry_Price - 2.0 * ATR(14)[T]
           - TP1 = SMA(20)[T] (realizacja 50% i SL na Break-Even)
           - TP2 = Upper_Bollinger_Band[T] (lub 60d High)
        """
        n = len(df)
        warmup = max(200, self.lookback_window)
        if n < warmup + 5:
            return []

        ind_df = self.prepare_causal_indicators(df)

        # Ekstrakcja wektorów numpy dla optymalizacji szybkości
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        dates = df.index

        rsi = ind_df["RSI_14"].values
        rsi_2 = ind_df["RSI_2"].values
        atr_14 = ind_df["ATR_14"].values
        bb_lower = ind_df["BB_Lower"].values
        bb_mid = ind_df["BB_Middle"].values
        bb_upper = ind_df["BB_Upper"].values
        sma_20 = ind_df["SMA_20"].values
        sma_200 = ind_df["SMA_200"].values
        macd_hist = ind_df["MACD_Hist"].values
        rvol = ind_df["RVol"].values
        sma_vol = ind_df["SMA_Volume_20"].values

        # 60-sesyjne Swing High (fallback dla TP2)
        roll_high = df["High"].rolling(self.lookback_window, min_periods=self.lookback_window).max().values

        # Rozpoznanie klasy aktywa (ETF vs Akcje)
        is_etf = ETFScanner.is_etf_instrument(ticker, name, sector)
        asset_type = "ETF" if is_etf else "Equity"

        # Adaptacyjny dobór progów płynności (specyfika rynku ETF UCITS vs Akcje)
        liq_cfg = self.config.get("liquidity_filter", {})
        if is_etf and "etfs_europe" in liq_cfg:
            min_vol_sma = float(liq_cfg["etfs_europe"].get("min_volume_sma_20", 3000))
            min_dollar_vol = float(liq_cfg["etfs_europe"].get("min_dollar_volume", 250000))
        else:
            min_vol_sma = float(liq_cfg.get("min_volume_sma_20", 30000))
            min_dollar_vol = float(liq_cfg.get("min_dollar_volume", 1000000))

        # Parametry ryzyka i scoringu
        risk_cfg = self.config.get("risk_management", {})
        use_atr_sl = bool(risk_cfg.get("use_atr_sl", True))
        atr_sl_mult = float(risk_cfg.get("atr_sl_multiplier", 2.0))
        sl_buffer_pct = float(risk_cfg.get("sl_buffer_pct", 0.005))
        max_sl_pct = float(risk_cfg.get("max_sl_distance_pct", 8.5))
        max_candle_pct = float(risk_cfg.get("max_candle_change_pct", 3.0))
        min_rr_tp1 = float(risk_cfg.get("min_rr_to_tp1", 0.60))
        min_rr_tp2 = float(risk_cfg.get("min_rr_to_tp2", 1.40))
        req_uptrend = bool(risk_cfg.get("require_uptrend_sma200", True))

        ind_cfg = self.config.get("indicators", {})
        rvol_threshold = float(ind_cfg.get("rvol", {}).get("absorption_threshold", 1.4))
        fibo_window = int(ind_cfg.get("fibonacci", {}).get("lookback_window", self.lookback_window))
        fibo_margin = float(ind_cfg.get("fibonacci", {}).get("margin_pct", 0.01))
        min_required_score = int(self.config.get("scoring", {}).get("min_criteria_required", self.min_score))

        trades: List[Trade] = []
        t = warmup

        while t < n - 1:
            # 1. FILTR PŁYNNOŚCI
            if np.isnan(sma_vol[t]) or sma_vol[t] < min_vol_sma:
                t += 1
                continue
            dollar_vol = sma_vol[t] * float(closes[t])
            if dollar_vol < min_dollar_vol:
                t += 1
                continue

            if np.isnan(rsi[t]) or np.isnan(bb_lower[t]) or np.isnan(macd_hist[t]):
                t += 1
                continue

            # 2. FILTR TRENDU GŁÓWNEGO (SMA 200)
            if req_uptrend and sma_200 is not None:
                if not np.isnan(sma_200[t]) and closes[t] <= sma_200[t]:
                    t += 1
                    continue

            # 3. ZMIANA DZIENNA I OCHRONA ANTY-CRASH DLA AKCJI
            close_curr = float(closes[t])
            close_prev = float(closes[t-1])
            open_curr = float(opens[t])
            high_curr = float(highs[t])
            low_curr = float(lows[t])
            open_prev = float(opens[t-1])
            change_pct = ((close_curr - close_prev) / close_prev) * 100.0

            if not is_etf and change_pct < -6.0:
                t += 1
                continue

            # 4. EWALUACJA 5 KRYTERIÓW MATRYCY KONFLUENCJI (1:1 ze scanner.py)
            # C1: Wyprzedanie RSI (RSI < 35 lub powrót ponad 30 w 2 sesjach)
            rsi_val = float(rsi[t])
            rsi_prev_val = float(rsi[t-1])
            rsi_prev2_val = float(rsi[t-2]) if t >= 2 else rsi_prev_val
            crossed_30_up = (rsi_prev_val < 30.0 and rsi_val >= 30.0) or (rsi_prev2_val < 30.0 and rsi_prev_val >= 30.0)
            c1_oversold = bool(rsi_val < 35.0 or crossed_30_up)

            # C2: Dolna wstęga Bollingera lub Strefa Wsparcia Fibo (0.500/0.618 z ostatnich 60 sesji)
            bb_touched = bool(low_curr <= (bb_lower[t] * 1.002))
            w_start = max(0, t - fibo_window + 1)
            swing_high = float(highs[w_start:t + 1].max())
            swing_low = float(lows[w_start:t + 1].min())
            diff = swing_high - swing_low
            fibo_in_zone = False
            fibo_target_price = None

            if diff > 0:
                for ratio in [0.500, 0.618]:
                    lvl_p = swing_high - (ratio * diff)
                    lb = lvl_p * (1.0 - fibo_margin)
                    ub = lvl_p * (1.0 + fibo_margin)
                    if (lb <= low_curr <= ub) or (lb <= close_curr <= ub):
                        fibo_in_zone = True
                        fibo_target_price = lvl_p
                        break

            c2_level = bool(bb_touched or fibo_in_zone)

            # C3: Price Action (Młot / Pinbar lub Objęcie Hossy)
            is_hammer = indicators.detect_hammer(open_curr, high_curr, low_curr, close_curr)
            is_engulfing = indicators.detect_bullish_engulfing(open_prev, close_prev, open_curr, close_curr)
            c3_candle = bool(is_hammer or is_engulfing)

            # C4: Wolumen absorpcyjny (RVol >= próg)
            rvol_val = float(rvol[t]) if not np.isnan(rvol[t]) else 1.0
            c4_volume = bool(rvol_val >= rvol_threshold)

            # C5: Impet MACD (wygaszanie pędu podaży - krótszy ujemny histogram)
            hist_curr = float(macd_hist[t])
            hist_prev = float(macd_hist[t-1])
            c5_momentum = bool(hist_prev < 0 and hist_curr > hist_prev)

            # Scoring konfluencji
            score = sum([c1_oversold, c2_level, c3_candle, c4_volume, c5_momentum])
            if score < min_required_score:
                t += 1
                continue

            # 5. TWARDE FILTRY RYZYKA I GEOMETRIA STOP LOSS / TAKE PROFIT
            atr_val = float(atr_14[t]) if not np.isnan(atr_14[t]) else float(np.mean(highs[max(0, t-13):t+1] - lows[max(0, t-13):t+1]))
            atr_sl_price = round(max(0.01, close_curr - (atr_sl_mult * atr_val)), 2)

            formation_low = min(low_curr, float(lows[t-1])) if is_engulfing else low_curr
            support_low = min(formation_low, fibo_target_price) if (fibo_in_zone and fibo_target_price) else formation_low
            structural_sl = round(support_low * (1.0 - sl_buffer_pct), 2)

            sl_price = atr_sl_price if use_atr_sl else structural_sl
            if sl_price >= close_curr:
                sl_price = round(close_curr * 0.98, 2)
            sl_pct = round(((close_curr - sl_price) / close_curr) * 100.0, 2)

            if sl_pct > max_sl_pct or sl_pct < 0.8:
                t += 1
                continue

            # Filtr Anti-Chasing (wielkość świecy sygnałowej)
            candle_body_pct = ((close_curr - open_curr) / open_curr) * 100.0 if open_curr > 0 else 0.0
            large_candle_pct = max(change_pct, candle_body_pct)
            if large_candle_pct > max_candle_pct:
                t += 1
                continue

            risk_t = close_curr - sl_price
            if risk_t <= 0:
                t += 1
                continue

            # Geometria TP1 / TP2
            bb_mid_val = float(sma_20[t])
            bb_upper_val = float(bb_upper[t])
            if close_curr >= bb_mid_val:
                tp1_price = round(bb_upper_val, 2)
            else:
                tp1_price = round(bb_mid_val, 2)

            tp2_price = round(max(swing_high, bb_upper_val * 1.02), 2)
            if tp2_price <= tp1_price:
                tp2_price = round(tp1_price + 1.0 * risk_t, 2)

            rr_tp1 = (tp1_price - close_curr) / risk_t
            rr_tp2 = (tp2_price - close_curr) / risk_t

            if rr_tp1 < min_rr_tp1 or rr_tp2 < min_rr_tp2:
                t += 1
                continue

            # 6. EGZEKUCJA W DNIU T+1 PO CENIE OTWARCIA
            entry_idx = t + 1
            entry_date = dates[entry_idx]
            signal_date = dates[t]
            entry_price = float(opens[entry_idx])

            if entry_price < self.min_price:
                t += 1
                continue

            if use_atr_sl:
                sl_price = round(max(0.01, entry_price - (atr_sl_mult * atr_val)), 2)

            risk = entry_price - sl_price
            if risk <= 0 or (risk / entry_price) < 0.008:
                t += 1
                continue

            # Filtr luki otwarcia: odrzucamy, jeśli luka w górę zjadła przestrzeń do TP1
            if (tp1_price - entry_price) < (0.35 * risk):
                t += 1
                continue

            # Koszt transakcyjny (prowizja + poślizg)
            cost_pct = self.fee_pct + self.slippage_pct
            entry_cost_r = (entry_price * cost_pct) / risk

            def calc_net_r(exit_p: float) -> float:
                gross_r = (exit_p - entry_price) / risk
                exit_cost_r = (exit_p * cost_pct) / risk
                return gross_r - entry_cost_r - exit_cost_r

            trade = Trade(
                ticker=ticker,
                name=name,
                sector=sector,
                signal_date=signal_date,
                entry_date=entry_date,
                entry_price=entry_price,
                initial_sl=sl_price,
                risk=risk,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                score=score,
                rvol=rvol_val,
                rsi_14=rsi_val,
                asset_type=asset_type
            )

            # 7. ŚLEDZENIE POZYCJI W DNIACH T+1, T+2, ... DO DNIA 20 (TIME STOP)
            tp1_hit = False
            current_sl = sl_price
            r_tp1_net = calc_net_r(tp1_price)
            holding_days = 0

            for day_step in range(1, self.time_stop + 1):
                curr_idx = t + day_step
                holding_days = day_step

                # Ochrona przed wyjściem poza zakres danych historycznych
                if curr_idx >= n:
                    last_close = float(closes[-1])
                    r_close = calc_net_r(last_close)
                    if not tp1_hit:
                        trade.r_multiple = r_close
                        trade.exit_reason = "TimeStop"
                    else:
                        trade.r_multiple = 0.5 * r_tp1_net + 0.5 * r_close
                        trade.exit_reason = "TP1+TimeStop"
                    trade.exit_date = dates[-1]
                    trade.exit_price = last_close
                    trade.holding_days = holding_days
                    trade.tp1_hit = tp1_hit
                    break

                d_open = float(opens[curr_idx])
                d_low = float(lows[curr_idx])
                d_high = float(highs[curr_idx])
                d_close = float(closes[curr_idx])
                d_date = dates[curr_idx]

                if not tp1_hit:
                    # PRZED OSIĄGNIĘCIEM TP1:
                    # 1. Sprawdzenie luki otwarcia (Overnight Gap poniżej SL)
                    if d_open <= current_sl:
                        trade.r_multiple = calc_net_r(d_open)
                        trade.exit_reason = "SL (Gap)"
                        trade.exit_date = d_date
                        trade.exit_price = d_open
                        trade.holding_days = holding_days
                        break
                    # 2. Reguła bezpieczeństwa Intraday: jeśli Low <= SL -> ZAWSZE SL (przed TP1)
                    elif d_low <= current_sl:
                        trade.r_multiple = calc_net_r(current_sl)
                        trade.exit_reason = "SL"
                        trade.exit_date = d_date
                        trade.exit_price = current_sl
                        trade.holding_days = holding_days
                        break
                    elif d_high >= tp1_price:
                        # Realizacja 50% pozycji na TP1
                        tp1_hit = True
                        trade.tp1_hit = True
                        trade.tp1_date = d_date
                        # SL dla pozostałych 50% przestawiony na Break-Even
                        current_sl = entry_price

                        # Sprawdzenie czy w tym samym dniu osiągnięto również TP2
                        if d_high >= tp2_price:
                            r_tp2_net = calc_net_r(tp2_price)
                            trade.r_multiple = 0.5 * r_tp1_net + 0.5 * r_tp2_net
                            trade.exit_reason = "TP1+TP2"
                            trade.exit_date = d_date
                            trade.exit_price = tp2_price
                            trade.holding_days = holding_days
                            break
                else:
                    # PO OSIĄGNIĘCIU TP1 (POZOSTAŁE 50% POZYCJI Z SL NA BREAK-EVEN):
                    # 1. Sprawdzenie luki otwarcia poniżej Break-Even
                    if d_open <= current_sl:
                        r_gap = calc_net_r(d_open)
                        trade.r_multiple = 0.5 * r_tp1_net + 0.5 * r_gap
                        trade.exit_reason = "TP1+BE (Gap)"
                        trade.exit_date = d_date
                        trade.exit_price = d_open
                        trade.holding_days = holding_days
                        break
                    # 2. Konserwatywna reguła Intraday: jeśli Low <= BE -> BE (przed TP2)
                    elif d_low <= current_sl:
                        r_be_net = calc_net_r(current_sl)
                        trade.r_multiple = 0.5 * r_tp1_net + 0.5 * r_be_net
                        trade.exit_reason = "TP1+BE"
                        trade.exit_date = d_date
                        trade.exit_price = current_sl
                        trade.holding_days = holding_days
                        break
                    elif d_high >= tp2_price:
                        # Zamknięcie pozostałych 50% na TP2
                        r_tp2_net = calc_net_r(tp2_price)
                        trade.r_multiple = 0.5 * r_tp1_net + 0.5 * r_tp2_net
                        trade.exit_reason = "TP1+TP2"
                        trade.exit_date = d_date
                        trade.exit_price = tp2_price
                        trade.holding_days = holding_days
                        break

                # TIME STOP (DZIEŃ 20):
                if day_step == self.time_stop:
                    r_close = calc_net_r(d_close)
                    if not tp1_hit:
                        trade.r_multiple = r_close
                        trade.exit_reason = "TimeStop"
                    else:
                        trade.r_multiple = 0.5 * r_tp1_net + 0.5 * r_close
                        trade.exit_reason = "TP1+TimeStop"
                    trade.exit_date = d_date
                    trade.exit_price = d_close
                    trade.holding_days = self.time_stop
                    break

            # Winsoryzacja skrajnych anomalii splitowych w historycznym cache
            trade.r_multiple = min(trade.r_multiple, 15.0)
            trades.append(trade)
            # Zapobieganie nakładaniu pozycji na tym samym instrumencie (One Trade Per Ticker)
            t += max(1, holding_days)

        return trades

    # ----------------------------------------------------------------------------------
    # SYMULATOR PORTFELA (PORTFOLIO RISK & CORRELATION MANAGEMENT)
    # ----------------------------------------------------------------------------------

    def apply_portfolio_constraints(self, all_trades: List[Trade]) -> List[Trade]:
        """
        Filtruje sygnały chronologicznie (dzień po dniu) na poziomie całego portfela:
        1. Limit maksymalnej liczby otwartych pozycji naraz (max_open_positions).
        2. Limit pozycji z tego samego sektora (max_per_sector) - ochrona przed korelacją.
        3. Limit nowych wejść w jednym dniu (max_daily_entries) - ochrona przed dniem krachu.
        4. Ranking jakościowy: przy nadmiarze sygnałów wybiera setupy o najwyższym Score i RVol.
        """
        if (
            self.max_open_positions is None
            and self.max_per_sector is None
            and self.max_daily_entries is None
        ):
            return all_trades

        # Grupowanie kandydatów po dacie wejścia (chronologicznie)
        by_date: Dict[pd.Timestamp, List[Trade]] = {}
        for tr in all_trades:
            by_date.setdefault(tr.entry_date, []).append(tr)

        accepted_trades: List[Trade] = []
        active_positions: List[Trade] = []

        for current_date in sorted(by_date.keys()):
            # 1. Zwolnienie slotów dla pozycji, które zamknęły się najpóźniej w dniu current_date
            active_positions = [
                pos for pos in active_positions
                if pos.exit_date is not None and pos.exit_date > current_date
            ]

            
            # 2. Sortowanie kandydatów: preferuj płynne rynki (USA i ETF przed rynkami lokalnymi z kropką),
            # a następnie najwyższy Score konfluencji oraz najwyższy względny wolumen (RVol)
            candidates = sorted(
                by_date[current_date],
                key=lambda x: (
                    0 if ("." not in x.ticker or "ETF" in x.name.upper() or "ETC" in x.name.upper()) else 1,
                    -x.score,
                    -x.rvol
                )
            )

            daily_opened = 0
            for cand in candidates:
                if self.max_open_positions is not None and len(active_positions) >= self.max_open_positions:
                    break
                if self.max_daily_entries is not None and daily_opened >= self.max_daily_entries:
                    break
                if self.max_per_sector is not None:
                    sector_count = sum(1 for pos in active_positions if pos.sector == cand.sector)
                    if sector_count >= self.max_per_sector:
                        continue

                active_positions.append(cand)
                accepted_trades.append(cand)
                daily_opened += 1

        return accepted_trades

    # ----------------------------------------------------------------------------------
    # GŁÓWNA METODA WYKONAWCZA SYMULACJI
    # ----------------------------------------------------------------------------------

    def run(
        self,
        tickers_subset: Optional[List[str]] = None,
        period: str = "2y",
        interval: str = "1d",
        limit: Optional[int] = None
    ) -> BacktestResult:
        """
        Przeprowadza pełną symulację testowania wstecznego na bazie ETF/akcji.
        """
        t0 = time.time()
        tickers_df = self.load_tickers()

        if tickers_subset is not None:
            tickers_df = tickers_df[tickers_df["ticker"].isin(tickers_subset)].reset_index(drop=True)

        if limit is not None and limit > 0:
            tickers_df = tickers_df.iloc[:limit].reset_index(drop=True)

        tickers_list = tickers_df["ticker"].tolist()
        logger.info("Rozpoczynanie symulacji dla %d instrumentów (period=%s, interval=%s)...",
                    len(tickers_list), period, interval)

        # Pobranie/załadowanie danych
        market_data = self.load_or_fetch_data(tickers_list, period=period, interval=interval)

        raw_trades: List[Trade] = []
        ticker_map = {row["ticker"]: (row["name"], row["sector"]) for _, row in tickers_df.iterrows()}

        for ticker, df in market_data.items():
            name, sector = ticker_map.get(ticker, (ticker, "General"))
            ticker_trades = self.simulate_ticker(ticker, name, sector, df)
            raw_trades.extend(ticker_trades)

        # Nałożenie ograniczeń portfelowych (ochrona przed korelacją i nadmiernym lewarem)
        final_trades = self.apply_portfolio_constraints(raw_trades)
        if len(final_trades) < len(raw_trades):
            logger.info(
                "Filtr portfelowy: zaakceptowano %d z %d sygnałów (MaxPos=%s, MaxSector=%s, MaxDaily=%s).",
                len(final_trades), len(raw_trades),
                self.max_open_positions, self.max_per_sector, self.max_daily_entries
            )

        elapsed = time.time() - t0
        result = BacktestResult(final_trades)
        result.print_summary(elapsed_sec=elapsed, total_tickers=len(market_data))

        # Osobne testy portfelowe (bez konkurencji innych klas aktywów)
        etf_raw = [tr for tr in raw_trades if ("ETF" in tr.name.upper() or "ETC" in tr.name.upper())]
        etf_final = self.apply_portfolio_constraints(etf_raw)
        etf_m = BacktestResult(etf_final).metrics
        etf_pf = f"{etf_m.profit_factor:.2f}" if etf_m.profit_factor != float("inf") else "INF"

        us_raw = [tr for tr in raw_trades if "." not in tr.ticker]
        us_final = self.apply_portfolio_constraints(us_raw)
        us_m = BacktestResult(us_final).metrics
        us_pf = f"{us_m.profit_factor:.2f}" if us_m.profit_factor != float("inf") else "INF"

        print(" PODSUMOWANIE SEGMENTOWE (OSOBNE TESTY PORTFELOWE):")
        print(f"  • WYŁĄCZNIE ETF/ETC : N={etf_m.total_trades:>4d} | WinRate={etf_m.win_rate_pct:>5.2f}% | AvgR={etf_m.avg_r_multiple:>+5.2f}R | PF={etf_pf:>5s} | MaxDD={etf_m.max_drawdown_pct:>5.2f}% | EV={etf_m.expected_value_r:>+5.2f}R")
        print(f"  • WYŁĄCZNIE USA     : N={us_m.total_trades:>4d} | WinRate={us_m.win_rate_pct:>5.2f}% | AvgR={us_m.avg_r_multiple:>+5.2f}R | PF={us_pf:>5s} | MaxDD={us_m.max_drawdown_pct:>5.2f}% | EV={us_m.expected_value_r:>+5.2f}R")
        print("=" * 88 + "\n")

        return result


# ======================================================================================
# PUNKT WEJŚCIA CLI (MAIN EXECUTION)
# ======================================================================================

def main() -> None:
    """Obsługa wywołania skryptu z poziomu wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Silnik testowania wstecznego (Backtest Engine) dla ETF Confluence Scanner."
    )
    parser.add_argument("--tickers", type=str, default="tickers.csv", help="Ścieżka do pliku CSV z tickerami.")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Folder lokalnego cache'u danych.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ścieżka do pliku konfiguracyjnego YAML.")
    parser.add_argument("--min-score", type=int, default=4, help="Minimalna siła setupu konfluencji (domyślnie 4).")
    parser.add_argument("--time-stop", type=int, default=20, help="Maksymalny czas trwania pozycji w dniach (domyślnie 20).")
    parser.add_argument("--fee-pct", type=float, default=0.0005, help="Prowizja na stronę (domyślnie 0.0005 = 0.05%%).")
    parser.add_argument("--slippage-pct", type=float, default=0.0005, help="Poślizg/spread na stronę (domyślnie 0.0005 = 0.05%%).")
    parser.add_argument("--max-positions", type=int, default=10, help="Maks. liczba równoczesnych pozycji w portfelu (domyślnie 10).")
    parser.add_argument("--max-per-sector", type=int, default=2, help="Maks. liczba pozycji w jednym sektorze (domyślnie 2).")
    parser.add_argument("--max-daily", type=int, default=3, help="Maks. liczba nowych wejść dziennie (domyślnie 3).")
    parser.add_argument("--unconstrained", action="store_true", help="Wyłącz limity portfela (testuj wszystkie sygnały).")
    parser.add_argument("--limit", type=int, default=None, help="Opcjonalny limit liczby testowanych instrumentów.")
    parser.add_argument("--export-trades", type=str, default=None, help="Opcjonalna ścieżka do zapisu logu transakcji CSV.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Poziom logowania (DEBUG, INFO, WARNING, ERROR).")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    backtester = ConfluenceBacktester(
        config_path=args.config,
        cache_dir=args.cache_dir,
        tickers_file=args.tickers,
        min_score=args.min_score,
        time_stop=args.time_stop,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        max_open_positions=None if args.unconstrained else args.max_positions,
        max_per_sector=None if args.unconstrained else args.max_per_sector,
        max_daily_entries=None if args.unconstrained else args.max_daily
    )

    result = backtester.run(limit=args.limit)

    if args.export_trades:
        df_trades = result.to_dataframe()
        df_trades.to_csv(args.export_trades, index=False)
        print(f"📁 Zapisano szczegółowy dziennik {len(df_trades)} transakcji do: {args.export_trades}")


if __name__ == "__main__":
    main()
