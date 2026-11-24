"""
ETF Confluence Scanner - notifier.py
Moduł integracji z API Telegrama z obsługą wysyłania zdjęć (sendPhoto) oraz audytu AI.
"""

import os
import time
import logging
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Odpowiedzialny za formatowanie i niezawodne wysyłanie alertów (tekst + wykresy) przez Telegram Bot API.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = 15,
        max_retries: int = 3,
        dry_run: bool = False
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run

        if self._is_placeholder(self.bot_token) or self._is_placeholder(self.chat_id):
            logger.warning("Telegram credentials nie zostały skonfigurowane. Tryb DRY-RUN aktywny.")
            self.dry_run = True

    @staticmethod
    def _is_placeholder(val: Optional[str]) -> bool:
        if not val:
            return True
        val_lower = val.strip().lower()
        return any(p in val_lower for p in ["your_bot_token", "example", "123456789:abc", "none"])

    def send_photo(self, photo_path: str, caption: str, parse_mode: str = "Markdown") -> bool:
        """
        Wysyła wykres PNG wraz z podpisem. Obsługuje limit 1024 znaków dla caption.
        """
        if self.dry_run:
            logger.info(
                "=== [DRY-RUN / TELEGRAM PHOTO ALERT] ===\nPlik: %s\nPodpis:\n%s\n========================================",
                photo_path,
                caption,
            )
            return True

        if not self.bot_token or not self.chat_id:
            logger.error("Brak poświadczeń Telegrama. Zdjęcie nie zostało wysłane.")
            return False

        if not os.path.exists(photo_path):
            logger.warning("Plik wykresu %s nie istnieje. Wysyłanie alertu jako tekst.", photo_path)
            return self.send_message(caption, parse_mode=parse_mode)

        # Telegram ma twardy limit 1024 znaków na podpis pod zdjęciem
        if len(caption) > 1024:
            ai_marker = "\n🤖 "
            if ai_marker in caption:
                setup_part, ai_part = caption.split(ai_marker, 1)
                photo_caption = setup_part[:980].rstrip() + "\n*(Pełny audyt AI poniżej 👇)*"
                followup_text = "🤖 " + ai_part
            else:
                cut_idx = caption.rfind("\n", 0, 950)
                if cut_idx == -1:
                    cut_idx = 950
                photo_caption = caption[:cut_idx].rstrip() + "\n\n*(Ciąg dalszy poniżej...)*"
                followup_text = caption[cut_idx:].lstrip()

            success = self._execute_send_photo(photo_path, photo_caption, parse_mode)
            if success and followup_text:
                self.send_message(followup_text, parse_mode=parse_mode)
            return success

        return self._execute_send_photo(photo_path, caption, parse_mode)

    def _execute_send_photo(self, photo_path: str, caption: str, parse_mode: str) -> bool:
        api_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

        for attempt in range(1, self.max_retries + 1):
            try:
                with open(photo_path, "rb") as photo_file:
                    data = {
                        "chat_id": self.chat_id,
                        "caption": caption,
                        "parse_mode": parse_mode
                    }
                    files = {"photo": photo_file}
                    response = requests.post(api_url, data=data, files=files, timeout=self.timeout)

                if response.status_code == 200:
                    logger.info("Wykres z analizą pomyślnie wysłany na Telegram.")
                    return True

                if response.status_code == 400 and parse_mode == "Markdown":
                    logger.warning("Błąd Markdown w sendPhoto. Próba wysłania jako czysty tekst...")
                    with open(photo_path, "rb") as photo_file:
                        data.pop("parse_mode", None)
                        files = {"photo": photo_file}
                        fallback_resp = requests.post(api_url, data=data, files=files, timeout=self.timeout)
                    if fallback_resp.status_code == 200:
                        return True

                logger.warning(
                    "Telegram sendPhoto zwróciło status %s: %s (Próba %d/%d)",
                    response.status_code,
                    response.text,
                    attempt,
                    self.max_retries,
                )
            except Exception as e:
                logger.error("Błąd połączenia podczas sendPhoto: %s", e)

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        return False

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        if self.dry_run:
            logger.info("=== [DRY-RUN / TELEGRAM MESSAGE] ===\n%s\n==================================", text)
            return True

        if not self.bot_token or not self.chat_id:
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
                    return True
                if response.status_code == 400 and parse_mode == "Markdown":
                    payload.pop("parse_mode", None)
                    fallback_resp = requests.post(api_url, json=payload, timeout=self.timeout)
                    if fallback_resp.status_code == 200:
                        return True
            except Exception as e:
                logger.error("Błąd send_message: %s", e)

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        return False

    @staticmethod
    def detect_currency(ticker: str) -> str:
        t = str(ticker).strip().upper()
        if "." not in t:
            return "USD"
        if t.endswith(".L"):
            return "GBp"
        if t.endswith((".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".HE", ".LS")):
            return "EUR"
        if t.endswith(".WA"):
            return "PLN"
        return "USD"

    @staticmethod
    def format_alert_message(setup: Dict[str, Any]) -> str:
        currency = setup.get("currency") or TelegramNotifier.detect_currency(setup["ticker"])
        trend_status = setup.get("trend_status", "Brak danych")

        # Sekcja Audytu AI (jeśli dostępna)
        ai_section = ""
        ai = setup.get("ai_review")
        if ai and ai.get("verdict") not in [None, "BYPASS", "ERROR"]:
            veto_rule = ai.get("veto_rule")
            veto_line = f"• **Reguła VETO:** {veto_rule}\n" if veto_rule and veto_rule != "NONE" else ""

            approval_list = ai.get("approval_reasons") or []
            approval_block = ""
            if approval_list:
                approvals = "\n".join([f"  • {r}" for r in approval_list])
                approval_block = f"• **Argumenty ZA (Zatwierdzenie):**\n{approvals}\n"

            devil_list = ai.get("devil_advocate_reasons") or []
            devil_block = ""
            if devil_list:
                reasons = "\n".join([f"  • {r}" for r in devil_list])
                devil_block = f"• **Adwokat Diabła (Ryzyka):**\n{reasons}\n"

            ai_section = (
                f"\n🤖 **WERDYKT AUDYTORA AI (Gemini 3.8 Flash):**\n"
                f"• **Ocena:** {ai.get('verdict')} (Zaufanie: {ai.get('confidence_score', 'N/A')}/10)\n"
                f"{veto_line}"
                f"• **Struktura:** {ai.get('market_structure')}\n"
                f"• **Opór:** {ai.get('immediate_barrier')}\n"
                f"{approval_block}"
                f"{devil_block}"
                f"• **Plan IF->THEN:** {ai.get('actionable_plan')}\n"
            )

        msg = (
            f"🚨 **SETUP KONFLUENCJI: {setup['ticker']} ({setup['name']})** 🚨\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 **Cena:** {setup['close']:.2f} {currency} ({setup['change_pct']:+.2f}%)\n"
            f"📈 **Trend:** {trend_status}\n"
            f"📊 **Wskaźniki:** RSI: {setup['rsi_val']:.1f} | RVol: {setup['rvol']:.2f}x | {setup['candle_pattern']}\n"
            f"🎯 **Risk Management:**\n"
            f"• **Wejście:** ~{setup['close']:.2f}\n"
            f"• **Stop Loss:** {setup['sl_price']:.2f} (-{abs(setup['sl_pct']):.2f}%)\n"
            f"• **TP1:** {setup['tp1_price']:.2f} (+{setup['tp1_pct']:.2f}%) | R:R 1:{setup['rr_tp1']:.2f}\n"
            f"• **TP2:** {setup['tp2_price']:.2f} (+{setup['tp2_pct']:.2f}%) | R:R 1:{setup['rr_ratio']:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
            f"{ai_section}"
        )
        return msg
