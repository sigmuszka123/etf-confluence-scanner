"""
ETF Confluence Scanner - main.py
Główny punkt wejścia (Orkiestrator systemu).
Uruchamia proces skanowania po zamknięciu sesji giełdowej w Europie:
1. Ładuje zmienne środowiskowe (.env).
2. Wczytuje konfigurację i bazę tickerów.
3. Pobiera dane, weryfikuje płynność i wylicza konfluencję.
4. Wysyła powiadomienia na Telegram dla zakwalifikowanych setupów.
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from scanner import ETFScanner
from notifier import TelegramNotifier

logger = logging.getLogger("main")


def load_environment_variables(env_path: str = ".env") -> None:
    """
    Bezpieczne ładowanie zmiennych z pliku .env.
    Używa python-dotenv jeśli jest dostępne, lub natywnego bezpiecznego parsera.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path)
        logger.debug("Załadowano .env przy użyciu python-dotenv.")
    except ImportError:
        # Lekki, natywny parser .env jako fallback
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val
            logger.debug("Załadowano .env za pomocą natywnego parsera.")


def generate_mock_market_data(tickers_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generuje realistyczne dane syntetyczne (OHLCV) dla testów w środowisku offline/CI.
    Symuluje 1-2 setupy konfluencji (wyprzedanie + młot/engulfing na dolnym BB/Fibo).
    """
    logger.info("Generowanie danych syntetycznych (Mock Mode)...")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=120, freq='B')
    result: Dict[str, pd.DataFrame] = {}

    for idx, row in tickers_df.iterrows():
        ticker = row['ticker']
        # Bazowa cena 50-200 EUR
        base_price = 50.0 + (idx * 5.0) % 150.0
        daily_returns = np.random.normal(loc=0.0002, scale=0.012, size=len(dates))
        
        # Dla pierwszych 2 ETF-ów wymuszamy idealną korektę i odbicie (Setup Konfluencji)
        if idx in (0, 3):
            # Ostatnie 15 dni to spadek (RSI < 30)
            daily_returns[-15:-1] = -0.015
            daily_returns[-1] = 0.005 # Odbicie na ostatniej sesji

        cum_returns = np.cumprod(1.0 + daily_returns)
        closes = base_price * cum_returns

        highs = closes * (1.0 + np.abs(np.random.normal(0, 0.005, size=len(dates))))
        lows = closes * (1.0 - np.abs(np.random.normal(0, 0.005, size=len(dates))))
        opens = (closes + lows) / 2.0
        volumes = np.random.randint(15000, 80000, size=len(dates))

        # Jeżeli to targetowany setup konfluencji, wymodeluj świecę młota i absorpcję wolumenu
        if idx in (0, 3):
            # Ostatnia świeca: Młot na wyprzedanym rynku
            opens[-1] = closes[-1] * 0.998
            lows[-1] = closes[-1] * 0.970  # Długi dolny knot
            highs[-1] = closes[-1] * 1.002 # Minimalny górny knot
            volumes[-1] = int(np.mean(volumes[-20:]) * 1.8) # RVol = 1.8x

        df_mock = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)

        result[ticker] = df_mock

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ETF Confluence Scanner - Skaner techniczny ETF-ów UCITS (1D) z powiadomieniami Telegram."
    )
    parser.add_argument("--config", default="config.yaml", help="Ścieżka do pliku konfiguracji YAML")
    parser.add_argument("--env", default=".env", help="Ścieżka do pliku środowiskowego .env")
    parser.add_argument("--dry-run", action="store_true", help="Tryb testowy: nie wysyła realnych żądań HTTP do Telegrama")
    parser.add_argument("--test-alert", action="store_true", help="Wysyła przykładowy alert testowy w celu weryfikacji połączenia")
    parser.add_argument("--mock-data", action="store_true", help="Używa syntetycznych danych do testów bez dostępu do sieci")
    args = parser.parse_args()

    # 1. Ładowanie zmiennych środowiskowych
    load_environment_variables(args.env)

    # 2. Inicjalizacja skanera
    try:
        scanner = ETFScanner(config_path=args.config)
    except Exception as e:
        logger.error("Błąd inicjalizacji skanera: %s", e)
        return 1

    # 3. Inicjalizacja notyfikatora Telegram
    tg_cfg = scanner.config.get("telegram", {})
    is_dry_run = args.dry_run or tg_cfg.get("dry_run", False)
    notifier = TelegramNotifier(
        bot_token=os.getenv(tg_cfg.get("token_env_var", "TELEGRAM_BOT_TOKEN")),
        chat_id=os.getenv(tg_cfg.get("chat_id_env_var", "TELEGRAM_CHAT_ID")),
        timeout=tg_cfg.get("request_timeout", 10),
        max_retries=tg_cfg.get("max_retries", 3),
        dry_run=is_dry_run
    )

    # Obsługa flagi testowej --test-alert
    if args.test_alert:
        logger.info("Wysyłanie próbnego alertu testowego...")
        dummy_setup = {
            "ticker": "SXR8.DE",
            "name": "iShares Core S&P 500 UCITS ETF (Acc)",
            "sector": "Broad Index (US)",
            "close": 542.80,
            "change_pct": 0.85,
            "rsi_val": 28.4,
            "rsi_status": "Odbicie w górę ponad 30",
            "fibo_status": "Strefa 0.618 (~540.20 EUR)",
            "bb_status": "Dotknięcie dolnej wstęgi (Wsparcie)",
            "candle_pattern": "Młot (Pinbar)",
            "rvol": 1.65,
            "macd_status": "Wygaszanie pędu podaży (histogram rośnie)",
            "sl_price": 537.50,
            "sl_pct": 0.98,
            "tp1_price": 558.10,
            "tp1_pct": 2.82,
            "tp2_price": 578.40,
            "tp2_pct": 6.56,
            "rr_ratio": 6.72,
            "rr_tp1": 2.88,
            "score": 5
        }
        msg = notifier.format_alert_message(dummy_setup)
        success = notifier.send_message(msg)
        return 0 if success else 1

    # 4. Wykonanie procesu skanowania
    logger.info("=== [ROZPOCZĘCIE SKANOWANIA ETF CONFLUENCE SCANNER] ===")
    tickers_df = scanner.load_tickers()

    if args.mock_data:
        data_dict = generate_mock_market_data(tickers_df)
        qualified_setups: List[Dict[str, Any]] = []
        for _, row in tickers_df.iterrows():
            ticker = row['ticker']
            if ticker in data_dict:
                setup = scanner.evaluate_ticker(ticker, row['name'], row['sector'], data_dict[ticker])
                if setup:
                    qualified_setups.append(setup)
    else:
        qualified_setups = scanner.run_scan()

    # 5. Dystrybucja powiadomień
    if not qualified_setups:
        logger.info(
            "ℹ️ Brak instrumentów spełniających kryteria konfluencji (min. 3/5). "
            "Zgodnie z zasadą braku fałszywych alarmów, żaden alert nie został wysłany."
        )
        return 0

    logger.info("🚀 Wykryto %d setup(ów) wysokiego prawdopodobieństwa! Rozpoczynam wysyłanie alertów...", len(qualified_setups))

    for idx, setup in enumerate(qualified_setups, 1):
        alert_msg = notifier.format_alert_message(setup)
        notifier.send_message(alert_msg)
        
        # Krótka pauza między komunikatami w celu ochrony przed rate limitem Telegrama
        if idx < len(qualified_setups):
            time.sleep(0.5)

    logger.info("=== [ZAKOŃCZONO SKANOWANIE POMYŚLNIE] ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
