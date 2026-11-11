"""
ETF Confluence Scanner - test_scanner.py
Zestaw kompleksowych testów jednostkowych weryfikujących poprawność matematyczną,
detekcję formacji świecowych, matrycę scoringu i kalkulator ryzyka.
"""

import unittest
import pandas as pd
import numpy as np

import indicators
from scanner import ETFScanner
from notifier import TelegramNotifier


class TestIndicators(unittest.TestCase):

    def setUp(self):
        # 100 dni syntetycznych danych
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        closes = np.linspace(100, 80, 100) # trend spadkowy
        self.df = pd.DataFrame({
            "Open": closes + 0.2,
            "High": closes + 1.0,
            "Low": closes - 1.0,
            "Close": closes,
            "Volume": [10000] * 99 + [20000]
        }, index=dates)

    def test_bollinger_bands(self):
        mid, upper, lower = indicators.calculate_bollinger_bands(self.df["Close"], window=20, num_std=2.0)
        self.assertEqual(len(mid), 100)
        # Ostatnia wartość: Upper > Mid > Lower
        self.assertGreater(upper.iloc[-1], mid.iloc[-1])
        self.assertGreater(mid.iloc[-1], lower.iloc[-1])
        # Sprawdzenie symetrii wstęg
        std = (upper.iloc[-1] - mid.iloc[-1]) / 2.0
        self.assertAlmostEqual(upper.iloc[-1] - mid.iloc[-1], mid.iloc[-1] - lower.iloc[-1], places=4)

    def test_rsi(self):
        rsi = indicators.calculate_rsi(self.df["Close"], window=14)
        self.assertEqual(len(rsi), 100)
        # Przy jednostajnym spadku cen, RSI powinno być silnie wyprzedane (< 30)
        last_rsi = rsi.iloc[-1]
        self.assertLess(last_rsi, 30.0)
        self.assertGreaterEqual(last_rsi, 0.0)

    def test_macd(self):
        macd, sig, hist = indicators.calculate_macd(self.df["Close"], fast=12, slow=26, signal=9)
        self.assertEqual(len(hist), 100)
        # Różnica MACD - Signal równa Histogramowi
        self.assertAlmostEqual(hist.iloc[-1], macd.iloc[-1] - sig.iloc[-1], places=5)

    def test_rvol(self):
        rvol, sma_vol = indicators.calculate_rvol(self.df["Volume"], window=20)
        # Dzisiejszy wolumen 20 000 przy wcześniejszych 10 000
        # SMA20 = (19*10000 + 20000) / 20 = 10500
        # RVol = 20000 / 10500 = 1.9047
        expected_sma = (19 * 10000 + 20000) / 20.0
        expected_rvol = 20000 / expected_sma
        self.assertAlmostEqual(sma_vol.iloc[-1], expected_sma, places=2)
        self.assertAlmostEqual(rvol.iloc[-1], expected_rvol, places=2)
        self.assertGreaterEqual(rvol.iloc[-1], 1.4)

    def test_hammer_detection(self):
        # Prawidłowy młot (Pinbar):
        # Open=100, Close=101 (korpus = 1), Low=96 (dolny knot = 100-96 = 4 >= 2*1), High=101.2 (górny knot = 0.2)
        # Zasięg = 101.2 - 96 = 5.2. Close w górnych 35% zasięgu: (101.2 - 101) / 5.2 = 0.038 <= 0.35
        is_hammer = indicators.detect_hammer(open_price=100.0, high_price=101.2, low_price=96.0, close_price=101.0)
        self.assertTrue(is_hammer)

        # Nieprawidłowy młot: brak dolnego knota
        not_hammer = indicators.detect_hammer(open_price=100.0, high_price=105.0, low_price=99.5, close_price=104.0)
        self.assertFalse(not_hammer)

    def test_bullish_engulfing_detection(self):
        # Poprzednia świeca czerwona: Open=100, Close=95
        # Dzisiejsza świeca zielona obejmująca: Open=94, Close=102
        is_engulfing = indicators.detect_bullish_engulfing(
            prev_open=100.0, prev_close=95.0, curr_open=94.0, curr_close=102.0
        )
        self.assertTrue(is_engulfing)

        # Nieprawidłowe objęcie: poprzednia świeca była zielona
        not_engulfing = indicators.detect_bullish_engulfing(
            prev_open=95.0, prev_close=100.0, curr_open=94.0, curr_close=102.0
        )
        self.assertFalse(not_engulfing)

    def test_fibonacci_levels(self):
        fibo = indicators.calculate_fibo_levels(self.df, window=60)
        # Swing High i Swing Low
        self.assertEqual(fibo["0.0"], fibo["swing_high"])
        self.assertEqual(fibo["1.0"], fibo["swing_low"])
        diff = fibo["swing_high"] - fibo["swing_low"]
        self.assertAlmostEqual(fibo["0.500"], fibo["swing_high"] - (0.5 * diff), places=4)
        self.assertAlmostEqual(fibo["0.618"], fibo["swing_high"] - (0.618 * diff), places=4)

        # Test strefy wsparcia Fibo (+/- 1%)
        price_near_500 = fibo["0.500"] * 1.005 # +0.5% od poziomu
        in_zone, level_key, level_price = indicators.is_in_fibo_support_zone(
            low_price=price_near_500, close_price=price_near_500, fibo_levels=fibo, margin_pct=0.01
        )
        self.assertTrue(in_zone)
        self.assertEqual(level_key, "0.500")


class TestScannerAndRisk(unittest.TestCase):

    def setUp(self):
        self.scanner = ETFScanner("config.yaml")

    def test_liquidity_filter_rejection(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        low_vol_df = pd.DataFrame({
            "Open": [100.0] * 30,
            "High": [101.0] * 30,
            "Low": [99.0] * 30,
            "Close": [100.0] * 30,
            "Volume": [1000] * 30 # Tylko 1 000 wolumenu (< 5 000)
        }, index=dates)

        setup = self.scanner.evaluate_ticker("TEST.DE", "Low Liquidity ETF", "Test", low_vol_df)
        self.assertIsNone(setup) # Odrzucony przez filtr płynności

    def test_confluence_qualification_and_risk_manager(self):
        # Tworzymy 80 sesji, w których ostatnia sesja ma konfluencję:
        # RSI wyprzedany, test dolnej wstęgi BB, świeca młota, RVol >= 1.4x, MACD hamuje
        dates = pd.date_range("2026-01-01", periods=80, freq="D")
        prices = np.linspace(100, 75, 80)
        prices[-1] = 76.0

        df = pd.DataFrame({
            "Open": prices,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": [10000] * 79 + [30000] # RVol ~ 3.0
        }, index=dates)

        # Młot na dolnej wstędze
        df.loc[dates[-1], "Open"] = 75.5
        df.loc[dates[-1], "Low"] = 72.0 # dolny knot = 3.5
        df.loc[dates[-1], "High"] = 76.1
        df.loc[dates[-1], "Close"] = 76.0

        setup = self.scanner.evaluate_ticker("SXR8.DE", "iShares Core S&P 500", "Broad Index", df)
        self.assertIsNotNone(setup)
        self.assertGreaterEqual(setup["score"], 3)
        self.assertLess(setup["sl_price"], setup["close"])
        self.assertGreater(setup["tp1_price"], setup["close"])
        self.assertGreater(setup["tp2_price"], setup["close"])
        self.assertGreater(setup["rr_ratio"], 0.0)


class TestTelegramNotifier(unittest.TestCase):

    def test_format_alert_message(self):
        dummy_setup = {
            "ticker": "SXR8.DE",
            "name": "iShares Core S&P 500 UCITS ETF (Acc)",
            "sector": "Broad Index (US)",
            "close": 540.0,
            "change_pct": 0.5,
            "rsi_val": 29.5,
            "rsi_status": "Odbicie w górę ponad 30",
            "fibo_status": "Strefa 0.618 (~538.00 EUR)",
            "bb_status": "Dotknięcie dolnej wstęgi (Wsparcie)",
            "candle_pattern": "Młot (Pinbar)",
            "rvol": 1.75,
            "macd_status": "Wygaszanie pędu podaży (histogram rośnie)",
            "sl_price": 535.0,
            "sl_pct": 0.93,
            "tp1_price": 555.0,
            "tp1_pct": 2.78,
            "tp2_price": 570.0,
            "tp2_pct": 5.56,
            "rr_ratio": 6.0,
            "rr_tp1": 3.0,
            "score": 5
        }
        msg = TelegramNotifier.format_alert_message(dummy_setup)
        self.assertIn("🚨 **WYKRYTO SETUP KONFLUENCJI: SXR8.DE", msg)
        self.assertIn("📈 **Cena:** 540.00 EUR", msg)
        self.assertIn("• **RSI (14):** 29.5", msg)
        self.assertIn("• **Stosunek R:R (do TP2):** 1 : 6.00", msg)
        self.assertIn("*Zanegowanie tezy następuje po zamknięciu świecy poniżej SL.*", msg)


if __name__ == "__main__":
    unittest.main()
