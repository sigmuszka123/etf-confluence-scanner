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
Jesteś analitykiem ilościowym specjalizującym się w strategiach MEAN REVERSION (powrotu do średniej) na wykresach dziennych (1D).
Nadesłany wykres przeszedł już twarde filtry matematyczne (R:R, płynność, zmienność). Twoim zadaniem NIE jest szukanie pretekstu do odrzucenia, lecz wyłącznie wyłapanie skrajnych zagrożeń strukturalnych (veto).

DOMYŚLNA DECYZJA TO "PASS".
Użyj "REJECT" TYLKO wtedy, gdy na wykresie występuje co najmniej jeden z poniższych twardych warunków dyskwalifikujących (VETO).

CZEGO NIE WOLNO TRAKTOWAĆ JAKO POWODU DO "REJECT" (to naturalne cechy Mean Reversion):
- Cena znajdująca się poniżej SMA20 lub SMA50 (to normalny stan w wyprzedaniu).
- Negatywny lub opadający wskaźnik MACD oraz krótkoterminowa seria czerwonych świec (kilkudniowa korekta to nie krach).
- Niski wolumen na wczesnym etapie wyhamowania spadku.
- Sama obecność SMA20/SMA50 nad ceną (średnie ruchome NIE są twardym oporem horyzontalnym).
- Matematyczna ocena R:R (parametry zostały już zweryfikowane ilościowo; nie odrzucaj za samą liczbę R:R).

TWARDE WARUNKI VETO (zwróć "REJECT", jeśli zachodzi którykolwiek):
1. ZŁAMANA MAKROSTRUKTURA (3-6 miesięcy): Wyraźny, wielomiesięczny trend spadkowy (sekwencja makro LH/LL) lub trwałe przełamanie płaskiej/opadającej SMA200 w dół bez oznak budowania bazy.
2. SZOK INFORMACYJNY / WODOSPAD: Pionowa wyprzedaż z dużą luką spadkową (gap down) lub ekstremalnym skokiem wolumenu podażowego (szok wynikowy/kapitulacja), gdzie ostatnia świeca zamyka się na samym minimum bez żadnego śladu wyhamowania.
3. TWARDY OPÓR HORYZONTALNY PRZED TP1: Wyraźny, wielokrotnie testowany szczyt cenowy lub strefa konsolidacji/dystrybucji z ostatnich 2-3 miesięcy leżąca bezpośrednio nad ceną i wyraźnie PONIŻEJ poziomu TP1, realnie blokująca dojście do celu.
4. ŚWIECA SILNEGO ODRZUCENIA: Ostatnia świeca ma bardzo długi górny knot (ponad 50% jej całkowitej wysokości) z zamknięciem blisko minimum, sygnalizujący agresywną podaż na próbie odbicia.
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
