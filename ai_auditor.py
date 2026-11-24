"""
ai_auditor.py - Moduł bezstronnego audytu ryzyka Price Action za pomocą Gemini 3.8 Flash.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List, Literal
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class AuditResult(BaseModel):
    """Wymuszony schemat odpowiedzi JSON. Kolejność pól wymusza analizę przed werdyktem."""
    market_structure: str = Field(
        description="Zwięzły opis makrostruktury 3-6M oraz bieżącej korekty (np. Makro hossa nad SMA200, krótkoterminowe wyprzedanie)."
    )
    immediate_barrier: str = Field(
        description="Poziom cenowy najbliższego twardego oporu horyzontalnego z wykresu."
    )
    invalidation_level: str = Field(
        description="Poziom cenowy, po którego przebiciu scenariusz odbicia jest w 100% obalony."
    )
    devil_advocate_reasons: List[str] = Field(
        description="2-3 konkretne ryzyka do obserwacji (wypełnij także przy PASS)."
    )
    veto_rule: Literal[
        "NONE",
        "BROKEN_MACRO",
        "NEWS_SHOCK",
        "OVERHEAD_SUPPLY",
        "REJECTION_WICK",
    ] = Field(
        description="Kod złamanej twardej reguły VETO lub 'NONE', jeśli żaden warunek VETO nie zachodzi."
    )
    verdict: Literal["PASS", "REJECT"] = Field(
        description="'PASS' jeśli veto_rule == 'NONE'; 'REJECT' jeśli wystąpił którykolwiek warunek VETO."
    )
    confidence_score: int = Field(
        description="Pewność oceny w skali całkowitej od 1 do 10."
    )
    actionable_plan: str = Field(
        description="Krótka instrukcja IF -> THEN na otwarcie sesji T+1."
    )


SYSTEM_INSTRUCTION = """
Jesteś analitykiem ilościowym specjalizującym się w strategiach MEAN REVERSION (powrotu do średniej) na wykresach dziennych (1D).
Nadesłany wykres przeszedł już twarde filtry matematyczne (R:R, płynność, zmienność). Twoim zadaniem NIE jest szukanie pretekstu do odrzucenia, lecz wyłącznie wyłapanie skrajnych zagrożeń strukturalnych (VETO).

DOMYŚLNA DECYZJA TO "PASS" (veto_rule = "NONE").
Użyj "REJECT" TYLKO wtedy, gdy na wykresie występuje co najmniej jeden z 4 twardych warunków dyskwalifikujących (VETO).

CZEGO NIE WOLNO TRAKTOWAĆ JAKO POWODU DO "REJECT" (to naturalne cechy Mean Reversion):
- Cena znajdująca się poniżej SMA20 lub SMA50 (to normalny stan w wyprzedaniu).
- Negatywny lub opadający wskaźnik MACD oraz krótkoterminowa seria czerwonych świec (kilkudniowa korekta to nie krach).
- Niski wolumen na wczesnym etapie wyhamowania spadku.
- Sama obecność SMA20/SMA50 nad ceną (średnie ruchome NIE są twardym oporem horyzontalnym).
- Matematyczna ocena R:R (parametry zostały już zweryfikowane ilościowo w Pythonie).

TWARDE WARUNKI VETO (zwróć "REJECT" i odpowiedni kod veto_rule, jeśli zachodzi którykolwiek):
1. BROKEN_MACRO: Wyraźny, wielomiesięczny (3-6M) trend spadkowy (sekwencja makro LH/LL) lub trwałe przełamanie płaskiej/opadającej SMA200 w dół bez oznak budowania bazy.
2. NEWS_SHOCK: Pionowa wyprzedaż z dużą luką spadkową (gap down) lub ekstremalnym skokiem wolumenu podażowego (szok wynikowy/kapitulacja), gdzie ostatnia świeca zamyka się na samym minimum bez żadnego śladu wyhamowania.
3. OVERHEAD_SUPPLY: Wyraźny, wielokrotnie testowany szczyt cenowy lub strefa konsolidacji/dystrybucji z ostatnich 2-3 miesięcy leżąca bezpośrednio nad ceną i wyraźnie PONIŻEJ poziomu TP1, realnie blokująca dojście do celu.
4. REJECTION_WICK: Ostatnia świeca ma bardzo długi górny knot (ponad 50% jej całkowitej wysokości) z zamknięciem blisko minimum, sygnalizujący agresywną podaż na próbie odbicia.
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
        Przesyła wygenerowany wykres PNG wraz z kluczowymi poziomami cenowymi do Gemini 3.8 Flash.
        """
        if not self.client:
            return {"verdict": "BYPASS", "actionable_plan": "Brak konfiguracji AI API."}

        try:
            change_pct = float(setup["change_pct"])
            sl_pct = abs(float(setup["sl_pct"]))

            prompt = (
                "Zweryfikuj poniższy setup MEAN REVERSION na załączonym wykresie 1D:\n"
                f"- Ticker: {setup['ticker']} ({setup['name']})\n"
                f"- Cena zamknięcia (T): {setup['close']} {setup['currency']} (Zmiana 1D: {change_pct:+.2f}%)\n"
                f"- Poziom Stop Loss: {setup['sl_price']} (-{sl_pct:.2f}%)\n"
                f"- Poziom TP1 (cel bazowy): {setup['tp1_price']} | Poziom TP2: {setup['tp2_price']}\n"
                f"- Kontekst wyprzedania: RSI(14) = {setup['rsi_val']:.1f} | RVol = {setup['rvol']:.2f}x\n"
                "Sprawdź wyłącznie, czy na wykresie nie zachodzi żaden z 4 twardych warunków VETO."
            )

            with Image.open(chart_path) as image:
                response = self.client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=AuditResult,
                    ),
                )

            if response.parsed is not None:
                result = response.parsed.model_dump()
            else:
                result = json.loads(response.text)

            # Twarde zabezpieczenie spójności: jeśli model nie wskazał żadnej reguły VETO, to jest PASS
            if result.get("veto_rule") == "NONE" and result.get("verdict") == "REJECT":
                logger.warning(
                    "[%s] Niespójność AI (veto_rule=NONE przy REJECT) - wymuszam PASS.",
                    setup["ticker"],
                )
                result["verdict"] = "PASS"

            return result

        except Exception as e:
            logger.error("Błąd podczas audytu AI dla %s: %s", setup.get("ticker", "?"), e)
            return {
                "verdict": "ERROR",
                "veto_rule": "NONE",
                "actionable_plan": f"Błąd wykonania zapytania AI: {str(e)}",
            }
