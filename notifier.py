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

        if not self.bot_token or not
