"""
ETF Confluence Scanner - main.py
Główny punkt wejścia (Orkiestrator systemu).
Uruchamia proces skanowania po zamknięciu sesji giełdowej:
1. Ładuje zmienne środowiskowe (.env).
2. Wczytuje konfigurację i bazę tickerów.
3. Wykonuje skanowanie ilościowe (filtracja płynności + konfluencja + twarde filtry R:R).
4. Generuje wykresy techniczne 1D (mplfinance).
5. Poddaje zakwalifikowane setupy audytowi multimodalnego Agenta AI (Gemini 2.5 Flash).
6. Wysyła alerty z wykresami na Telegram i sprząta pliki tymczasowe.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from scanner import ETFScanner
from notifier import TelegramNotifier
from chart_generator import generate_candlestick_chart
from ai_auditor import AIAuditor

logger = logging.getLogger("main")


def load_environment_variables(env_path: str = ".env") -> None:
    """
    Bezpieczne ładowanie zmiennych z pliku .env.
    Obsługuje python-dotenv lub natywny parser liniowy jako fallback.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path)
        logger.debug("Załadowano .env przy użyciu python-dotenv.")
    except ImportError:
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
    """
    logger.info("Generowanie danych syntetycznych (Mock Mode)...")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=120, freq='B')
    result: Dict[str, pd.DataFrame] = {}

    for idx, row in tickers_df.iterrows():
        ticker = row['ticker']
        base_price = 50.0 + (idx * 5.0) % 150.0
        daily_returns = np.random.normal(loc=0.0002, scale=0.012, size=len(dates))
        
        # Wymuszamy korektę i odbicie
        if idx in (0, 3):
            daily_returns[-15:-1] = -0.015
            daily_returns[-1] = 0.005

        cum_returns = np.cumprod(1.0 + daily_returns)
        closes = base_price * cum_returns
        highs = closes * (1.0 + np.abs(np.random.normal(0, 0.005, size=len(dates))))
        lows = closes * (1.0 - np.abs(np.random.normal(0, 0.005, size=len(dates))))
        opens = (closes + lows) / 2.0
        volumes = np.random.randint(55000, 120000, size=len(dates))  # Płynność > 50k

        if idx in (0, 3):
            opens[-1] = closes[-1] * 0.995
            lows[-1] = closes[-1] * 0.980
            highs[-1] = closes[-1] * 1.002
            volumes[-1] = int(np.mean(volumes[-20:]) * 1.8)

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
        description="ETF/Stock Confluence Scanner - Automatyczny skaner z audytem AI i powiadomieniami Telegram."
    )
    parser.add_argument("--config", default="config.yaml", help="Ścieżka do pliku konfiguracji YAML")
    parser.add_argument("--env", default=".env", help="Ścieżka do pliku środowiskowego .env")
    parser.add_argument("--dry-run", action="store_true", help="Tryb testowy: brak realnych żądań HTTP do Telegrama")
    parser.add_argument("--test-alert", action="store_true", help="Wysyła przykładowy alert testowy w celu weryfikacji połączenia")
    parser.add_argument("--mock-data", action="store_true", help="Używa syntetycznych danych do testów bez dostępu do sieci")
    args = parser.parse_args()

    # 1. Konfiguracja środowiska
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
        timeout=tg_cfg.get("request_timeout", 15),
        max_retries=tg_cfg.get("max_retries", 3),
        dry_run=is_dry_run
    )

    # Obsługa flagi testowej --test-alert
    if args.test_alert:
        logger.info("Wysyłanie próbnego alertu testowego (z werdyktem AI)...")
        dummy_setup = {
            "ticker": "RPM",
            "name": "RPM International Inc.",
            "sector": "Materials",
            "currency": "USD",
            "close": 104.26,
            "change_pct": 1.54,
            "trend_status": "⚠️ BESSA (<SMA200)",
            "rsi_val": 53.6,
            "rsi_status": "Neutralny (>35)",
            "fibo_status": "Strefa 0.618 (~101.09 USD)",
            "bb_status": "Wewnątrz wstęg",
            "candle_pattern": "Przenikanie / Wybicie SMA20",
            "rvol": 1.43,
            "macd_status": "Wygaszanie pędu podaży",
            "sl_price": 100.58,
            "sl_pct": 3.66,
            "tp1_price": 109.09,
            "tp1_pct": 4.63,
            "tp2_price": 110.87,
            "tp2_pct": 6.34,
            "rr_ratio": 1.73,
            "rr_tp1": 1.27,
            "score": 4,
            "ai_review": {
                "verdict": "PASS",
                "confidence_score": 8,
                "market_structure": "Wyraźna obrona dołka HL (101.02) po listopadowym impulsie, odzyskanie SMA20",
                "immediate_barrier": "109.09 (Górna wstęga Bollingera)",
                "invalidation_level": "100.58 (Wybicie dołka świecy obrony)",
                "devil_advocate_reasons": [
                    "Kurs wciąż poniżej długoterminowej SMA200",
                    "Możliwe cofnięcie do testu 103.15 przed ruchem w górę"
                ],
                "actionable_plan": "Szukać wejścia na reteście 103.20-103.80. Negacja poniżej 100.58."
            }
        }
        msg = notifier.format_alert_message(dummy_setup)
        success = notifier.send_message(msg)
        return 0 if success else 1

    # 4. Wykonanie procesu skanowania
    logger.info("=== [ROZPOCZĘCIE SKANOWANIA CONFLUENCE SCANNER] ===")
    tickers_df = scanner.load_tickers()

    if args.mock_data:
        data_dict = generate_mock_market_data(tickers_df)
        auditor = AIAuditor()
        qualified_setups: List[Dict[str, Any]] = []

        for _, row in tickers_df.iterrows():
            ticker = row['ticker']
            if ticker in data_dict:
                df = data_dict[ticker]
                setup = scanner.evaluate_ticker(ticker, row['name'], row['sector'], df)
                if setup:
                    chart_file = f"chart_{ticker}.png"
                    try:
                        generate_candlestick_chart(df, ticker, output_path=chart_file)
                        setup["chart_path"] = chart_file
                    except Exception:
                        setup["chart_path"] = None

                    ai_verdict = auditor.audit_setup(chart_file, setup) if setup.get("chart_path") else {"verdict": "BYPASS"}
                    setup["ai_review"] = ai_verdict
                    if ai_verdict.get("verdict") in ["PASS", "BYPASS"]:
                        qualified_setups.append(setup)
    else:
        qualified_setups = scanner.run_scan()

    # 5. Dystrybucja powiadomień (Wykres + Tekst)
    if not qualified_setups:
        logger.info(
            "ℹ️ Brak setupów spełniających twarde kryteria ilościowe oraz wizualny audyt AI. "
            "Portfel chroniony przed pozycjami o ujemnym EV. Zero fałszywych alarmów."
        )
        return 0

    logger.info("🚀 Zakwalifikowano %d setup(ów)! Rozpoczynam wysyłanie alertów...", len(qualified_setups))

    for idx, setup in enumerate(qualified_setups, 1):
        alert_msg = notifier.format_alert_message(setup)
        chart_path = setup.get("chart_path")

        # Priorytet: wyślij wykres z podpisem (send_photo), a jeśli brak wykresu: wyślij sam tekst
        if chart_path and os.path.exists(chart_path):
            sent = notifier.send_photo(photo_path=chart_path, caption=alert_msg)
            # Higiena dysku: usunięcie pliku tymczasowego PNG po udanej lub nieudanej próbie
            try:
                os.remove(chart_path)
                logger.debug("Usunięto plik tymczasowy wykresu: %s", chart_path)
            except OSError as e:
                logger.warning("Nie udało się usunąć pliku %s: %s", chart_path, e)
        else:
            sent = notifier.send_message(alert_msg)

        if not sent:
            logger.warning("Nie udało się dostarczyć alertu dla %s na Telegram.", setup.get("ticker"))

        # Zabezpieczenie przed rate limitem Telegram API (30 wiadomości/sekundę)
        if idx < len(qualified_setups):
            time.sleep(1.0)

    logger.info("=== [ZAKOŃCZONO SKANOWANIE POMYŚLNIE] ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
