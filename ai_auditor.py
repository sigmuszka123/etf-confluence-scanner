"""
ai_auditor.py - Moduł bezstronnego audytu ryzyka Price Action za pomocą Gemini 2.5 Flash.
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """
Jesteś bezwzględnym, bezstronnym Risk Managerem w funduszu ilościowym.
Twoim celem jest chłodna weryfikacja techniczna nadesłanego wykresu 1D oraz parametrów setupu.
Odrzucaj myślenie życzeniowe. Szukaj słabych punktów.

Twoje zadanie:
1. Sprawdź strukturę rynku z lewej strony (Higher Highs / Higher Lows czy Lower Highs / Lower Lows?).
2. Zidentyfikuj, czy bezpośrednio nad ceną nie ma silnego oporu horyzontalnego (szczyt z ostatnich 2-3 miesięcy), który blokuje ruch.
3. Oceń anatomię ostatniej świecy: czy nie zostawiła długiego górnego knota odrzucenia od średnich?
4. Zwróć wynik WYŁĄCZNIE jako czysty obiekt JSON (bez znaczników markdown ```json):
{
  "verdict": "PASS" lub "REJECT",
  "confidence_score": 1-10,
  "market_structure": "zwięzły opis struktury (np. Zdrowa korekta HL w hossie)",
  "immediate_barrier": "poziom cenowy najbliższego twardego oporu z wykresu",
  "invalidation_level": "poziom cenowy, po którego przebiciu scenariusz wzrostowy jest w 100% obalony",
  "devil_advocate_reasons": ["argument przeciw 1", "argument przeciw 2"],
  "actionable_plan": "krótka instrukcja IF -> THEN na otwarcie sesji"
}
"""

class AIAuditor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("Brak klucza GEMINI_API_KEY. Audyt AI będzie pomijany (fallback).")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def audit_setup(self, chart_path: str, setup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Przesyła wygenerowany wykres PNG wraz z metrykami z bota do Gemini 2.5 Flash.
        """
        if not self.client:
            return {"verdict": "BYPASS", "actionable_plan": "Brak konfiguracji AI API."}

        try:
            image = Image.open(chart_path)
            prompt = f"""
            Zweryfikuj poniższy setup konfluencji z algorytmu:
            - Ticker: {setup['ticker']} ({setup['name']})
            - Cena zamknięcia: {setup['close']} {setup['currency']} (Zmiana: +{setup['change_pct']}%)
            - Sugerowany Stop Loss: {setup['sl_price']} (-{setup['sl_pct']}%)
            - Cel TP1 (SMA20/BB): {setup['tp1_price']} | Cel TP2: {setup['tp2_price']}
            - R:R do TP1: {setup['rr_tp1']:.2f} | R:R do TP2: {setup['rr_ratio']:.2f}
            - RSI(14): {setup['rsi_val']:.1f} | RVol: {setup['rvol']:.2f}x
            """

            response = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,  # Skrajnie niska temperatura dla deterministycznej oceny ryzyka
                    response_mime_type="application/json"
                )
            )

            # Bezpieczne parsowanie odpowiedzi JSON
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            return json.loads(raw_text)

        except Exception as e:
            logger.error("Błąd podczas audytu AI: %s", e)
            return {
                "verdict": "ERROR",
                "actionable_plan": f"Błąd wykonania zapytania AI: {str(e)}"
            }
