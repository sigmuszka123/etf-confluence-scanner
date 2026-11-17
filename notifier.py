"""
ETF Confluence Scanner - notifier.py
Moduł integracji z API Telegrama (wysyłanie powiadomień Markdown z obsługą błędów i retry).
"""

import os
import time
import logging
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Odpowiedzialny za formatowanie i niezawodne wysyłanie alertów przez Telegram Bot API.
    Obsługuje retry z wykładniczym backoffem, tryb dry-run i bezpieczne logowanie.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        dry_run: bool = False
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run

        # Weryfikacja czy token nie jest domyślnym placeholderem
        if self._is_placeholder(self.bot_token) or self._is_placeholder(self.chat_id):
            logger.warning(
                "Telegram credentials nie zostały skonfigurowane w .env lub zawierają placeholder. "
                "Notifier uruchomiony w trybie DRY-RUN (alert trafi wyłącznie do logów)."
            )
            self.dry_run = True

    @staticmethod
    def _is_placeholder(val: Optional[str]) -> bool:
        if not val:
            return True
        val_lower = val.strip().lower()
        return any(p in val_lower for p in ["your_bot_token", "example", "123456789:abc", "none"])

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Wysyła sformatowaną wiadomość tekstową na skonfigurowany czat Telegram.
        W przypadku braku poświadczeń lub włączonego trybu dry_run, drukuje treść do logów.
        """
        if self.dry_run:
            logger.info("=== [DRY-RUN / TELEGRAM MESSAGE PREVIEW] ===\n%s\n===========================================", text)
            return True

        if not self.bot_token or not self.chat_id:
            logger.error("Brak TELEGRAM_BOT_TOKEN lub TELEGRAM_CHAT_ID. Wiadomość nie została wysłana.")
            return False

        api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(api_url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    logger.info("Powiadomienie pomyślnie wysłane na Telegram (Czat: %s).", self.chat_id)
                    return True
                
                # Błędy klienta (np. 400 Bad Request z powodu błędnego Markdown)
                if response.status_code == 400 and parse_mode == "Markdown":
                    logger.warning("Błąd parsowania Markdown w Telegramie (400). Próba wysłania jako czysty tekst.")
                    payload["parse_mode"] = None  # fallback do czystego tekstu
                    fallback_resp = requests.post(api_url, json=payload, timeout=self.timeout)
                    if fallback_resp.status_code == 200:
                        logger.info("Wiadomość wysłana pomyślnie w trybie Plain Text po błędzie Markdown.")
                        return True

                logger.warning(
                    "Telegram API zwróciło status %s: %s (Próba %d/%d)",
                    response.status_code, response.text, attempt, self.max_retries
                )
            except (requests.ConnectionError, requests.Timeout) as net_err:
                logger.warning("Błąd sieciowy podczas komunikacji z Telegram API: %s (Próba %d/%d)", net_err, attempt, self.max_retries)
            except Exception as e:
                logger.error("Nieoczekiwany wyjątek podczas wysyłania wiadomości Telegram: %s", e)

            if attempt < self.max_retries:
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)

        logger.error("Nie udało się wysłać wiadomości na Telegram po %d próbach.", self.max_retries)
        return False

    @staticmethod
    def detect_currency(ticker: str) -> str:
        """Wykrywa walutę notowania na podstawie sufiksu giełdowego tickera."""
        t = str(ticker).strip().upper()
        if "." not in t:
            return "USD"
        if t.endswith(".L"):
            return "GBp"
        if t.endswith((".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".HE", ".LS")):
            return "EUR"
        if t.endswith(".SW"):
            return "CHF"
        if t.endswith(".ST"):
            return "SEK"
        if t.endswith(".CO"):
            return "DKK"
        if t.endswith(".WA"):
            return "PLN"
        return "USD"

    @staticmethod
    def format_alert_message(setup: Dict[str, Any]) -> str:
        """
        Formatuje dane zidentyfikowanego setupu konfluencji zgodnie ze ścisłą specyfikacją.
        """
        currency = setup.get("currency") or TelegramNotifier.detect_currency(setup["ticker"])
        trend_status = setup.get("trend_status", "Brak danych (SMA200)")

        # Sprawdzenie czy R:R do TP1 < 1.0 w celu dodania ostrzeżenia
        rr_tp1 = setup.get("rr_tp1", 0.0)
        warnings = ""
        if rr_tp1 < 1.0:
            warnings += f"\n⚠️ *Uwaga: Stosunek R:R do TP1 wynosi {rr_tp1:.2f} (< 1.0) – podwyższone ryzyko dla pierwszego celu.*"

        if setup.get("is_large_candle"):
            large_pct = setup.get("large_candle_pct", setup.get("change_pct", 0.0))
            warnings += f"\n⚠️ Uwaga: Duża świeca (+{large_pct:.2f}%) – wysokie ryzyko cofnięcia na otwarciu i szerokiego SL"

        atr_val = setup.get("atr_val")
        atr_sl_price = setup.get("atr_sl_price")
        atr_sl_pct = setup.get("atr_sl_pct")
        atr_line = ""
        if atr_val is not None and atr_sl_price is not None and atr_sl_pct is not None:
            atr_line = f"• **Zmienność (ATR 14):** {atr_val:.2f} | **Bezpieczny SL (2xATR):** {atr_sl_price:.2f} (-{atr_sl_pct:.2f}%)\n"

        msg = (
            f"🚨 **WYKRYTO SETUP KONFLUENCJI: {setup['ticker']} ({setup['name']})** 🚨\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 **Cena:** {setup['close']:.2f} {currency} | **Zmiana 1D:** {setup['change_pct']:+.2f}%\n"
            f"📈 **Trend (SMA 200):** {trend_status}\n"
            f"📊 **Odczyty wskaźników:**\n"
            f"• **RSI (14):** {setup['rsi_val']:.1f} (Status: {setup['rsi_status']})\n"
            f"• **Poziom Fibo:** {setup['fibo_status']}\n"
            f"• **Wstęgi Bollingera:** {setup['bb_status']}\n"
            f"• **Świeca:** {setup['candle_pattern']}\n"
            f"• **Wolumen (RVol):** {setup['rvol']:.2f}x (Średnia 20D)\n"
            f"• **MACD Histogram:** {setup['macd_status']}\n"
            f"\n"
            f"🎯 **Parametry zlecenia (Risk Management):**\n"
            f"• **Sugerowane wejście:** ~{setup['close']:.2f}\n"
            f"• **Twardy Stop Loss:** {setup['sl_price']:.2f} (-{setup['sl_pct']:.2f}%)\n"
            f"{atr_line}"
            f"• **Take Profit 1:** {setup['tp1_price']:.2f} (+{setup['tp1_pct']:.2f}%)\n"
            f"• **Take Profit 2:** {setup['tp2_price']:.2f} (+{setup['tp2_pct']:.2f}%)\n"
            f"• **Stosunek R:R (do TP2):** 1 : {setup['rr_ratio']:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Zanegowanie tezy następuje po zamknięciu świecy poniżej SL.*"
            f"{warnings}"
        )
        return msg
