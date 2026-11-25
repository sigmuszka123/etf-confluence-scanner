"""
ai_auditor.py - Moduł bezstronnego audytu ryzyka Price Action za pomocą Gemini 3.8 Flash.
Wersja z pełną obsługą 3 stanów (PASS / WATCHLIST / REJECT) oraz retry na błędy 503.
"""
import os
import time
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
        description="2-3 konkretne ryzyka do obserwacji (wypełnij przy każdym werdykcie)."
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
    verdict: Literal["PASS", "WATCHLIST", "REJECT"] = Field(
        description=(
            "'REJECT' jeśli veto_rule != 'NONE'. "
            "'WATCHLIST' jeśli veto_rule == 'NONE', ale ostatnia świeca jest spadkowa bez dolnego knota/formacji odwrócenia. "
            "'PASS' jeśli veto_rule == 'NONE' ORAZ widać oznaki wyhamowania/obrony wsparcia."
        )
    )
    approval_reasons: List[str] = Field(
        description=(
            "Jeśli verdict to 'PASS' lub 'WATCHLIST': 2-3 konkretne obserwacje techniczne wspierające strefę odbicia. "
            "Jeśli verdict == 'REJECT': zwróć pustą listę []."
        )
    )
    confidence_score: int = Field(
        ge=1,
        le=10,
        description="Pewność oceny w skali całkowitej od 1 do 10 (dla WATCHLIST maksymalnie 6)."
    )
    actionable_plan: str = Field(
        description="Krótka instrukcja IF -> THEN na otwarcie sesji T+1."
    )


SYSTEM_INSTRUCTION = """
Jesteś analitykiem ilościowym specjalizującym się w strategiach MEAN REVERSION (powrotu do średniej) na wykresach dziennych (1D).
Nadesłany wykres przeszedł już twarde filtry matematyczne (R:R, płynność, zmienność). Twoim zadaniem jest ocena struktury wykresu i wyłapanie zagrożeń (VETO) oraz rozróżnienie gotowego setupu (PASS) od setupu wymagającego potwierdzenia (WATCHLIST).

CZEGO NIE WOLNO TRAKTOWAĆ JAKO POWODU DO "REJECT" (naturalne cechy Mean Reversion):
- Cena znajdująca się poniżej SMA20 lub SMA50 (to normalny stan w wyprzedaniu).
- Negatywny lub opadający MACD oraz kilkudniowa korekta (to nie krach).
- Niski wolumen na wczesnym etapie wyhamowania spadku.
- Sama obecność SMA20/SMA50 nad ceną (średnie ruchome NIE są twardym oporem horyzontalnym).

1. TWARDE WARUNKI VETO (zwróć "REJECT" i odpowiedni kod veto_rule, jeśli zachodzi którykolwiek):
- BROKEN_MACRO: Wielomiesięczny (3-6M) trend spadkowy (sekwencja makro LH/LL) lub trwałe przełamanie płaskiej/opadającej SMA200 w dół bez oznak budowania bazy.
- NEWS_SHOCK: Pionowa wyprzedaż z dużą luką spadkową (gap down) lub ekstremalnym skokiem wolumenu kapitulacyjnego z zamknięciem na samym minimum bez żadnego śladu wyhamowania.
- OVERHEAD_SUPPLY: Wyraźny, wielokrotnie testowany szczyt cenowy lub strefa dystrybucji z ostatnich 2-3 miesięcy leżąca bezpośrednio nad ceną i wyraźnie PONIŻEJ TP1.
- REJECTION_WICK: Ostatnia świeca ma bardzo długi górny knot (>50% wysokości) z zamknięciem blisko minimum.

2. REGUŁA WATCHLIST (brak VETO, czyli veto_rule = "NONE", ale brak sygnału obrony):
- Jeśli ostatnia świeca jest spadkowa i zamyka się blisko dziennego minimum BEZ wyraźnego dolnego knota ani formacji odwrócenia, NIE dawaj oceny wyższej niż 6/10 i oznacz werdykt jako "WATCHLIST" zamiast "PASS" (chyba że cena dotyka bezpośrednio rosnącej SMA200 z widocznym wyhamowaniem pędu).

3. REGUŁA PASS (brak VETO + widoczna obrona):
- Zwróć "PASS", gdy veto_rule == "NONE" oraz na wykresie widać wyhamowanie podaży, dolny knot popytowy, formację odwrócenia lub bezpośredni test silnego klastra wsparcia z reakcją kupujących.
"""


class AIAuditor:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.max_retries = max_retries
        if not self.api_key:
            logger.warning("Brak klucza GEMINI_API_KEY. Audyt AI będzie pomijany (fallback).")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def audit_setup(self, chart_path: str, setup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Przesyła wygenerowany wykres PNG wraz z kluczowymi poziomami cenowymi do Gemini 3.8 Flash.
        Obsługuje ponowienia (retry) w przypadku chwilowych błędów 503/429.
        """
        if not self.client:
            return {
                "verdict": "BYPASS",
                "veto_rule": "NONE",
                "approval_reasons": [],
                "actionable_plan": "Brak konfiguracji AI API.",
            }

        change_pct = float(setup["change_pct"])
        sl_pct = abs(float(setup["sl_pct"]))

        prompt = (
            "Zweryfikuj poniższy setup MEAN REVERSION na załączonym wykresie 1D:\n"
            f"- Ticker: {setup['ticker']} ({setup['name']})\n"
            f"- Cena zamknięcia (T): {setup['close']} {setup['currency']} (Zmiana 1D: {change_pct:+.2f}%)\n"
            f"- Poziom Stop Loss: {setup['sl_price']} (-{sl_pct:.2f}%)\n"
            f"- Poziom TP1 (cel bazowy): {setup['tp1_price']} | Poziom TP2: {setup['tp2_price']}\n"
            f"- Kontekst wyprzedania: RSI(14) = {setup['rsi_val']:.1f} | RVol = {setup['rvol']:.2f}x\n"
            "Sprawdź warunki VETO. Jeśli nie ma VETO, zdecyduj między PASS (widoczna obrona) "
            "a WATCHLIST (świeca spadkowa bez potwierdzenia obrony, maks. 6/10)."
        )

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
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

                # TWARDE ZABEZPIECZENIA SPÓJNOŚCI LOGICZNEJ
                veto = result.get("veto_rule", "NONE")
                verdict = result.get("verdict", "REJECT")

                if veto != "NONE" and verdict != "REJECT":
                    logger.warning("[%s] Niespójność AI (veto=%s przy %s) - wymuszam REJECT.", setup["ticker"], veto, verdict)
                    result["verdict"] = "REJECT"
                    result["approval_reasons"] = []

                elif veto == "NONE" and verdict == "REJECT":
                    logger.warning("[%s] Niespójność AI (veto=NONE przy REJECT) - zamieniam na WATCHLIST.", setup["ticker"])
                    result["verdict"] = "WATCHLIST"

                if result.get("verdict") == "WATCHLIST" and result.get("confidence_score", 0) > 6:
                    result["confidence_score"] = 6

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "Błąd API AI dla %s (próba %d/%d): %s",
                    setup.get("ticker", "?"), attempt, self.max_retries, e
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("Nie udało się przeprowadzić audytu AI dla %s po %d próbach: %s",
                     setup.get("ticker", "?"), self.max_retries, last_error)
        return {
            "verdict": "ERROR",
            "veto_rule": "NONE",
            "approval_reasons": [],
            "actionable_plan": f"Błąd wykonania zapytania AI: {str(last_error)}",
        }
