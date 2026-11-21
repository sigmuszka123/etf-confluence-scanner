# ETF & Equity Confluence Scanner 📈🇪🇺 🇺🇸

Zaawansowany, modularny i odporny na błędy system skanowania ilościowego oraz testowania wstecznego dla **europejskich funduszy ETF UCITS** (XETRA `.DE`, LSE `.L`) oraz **płynnych akcji globalnych (USA / Europa)**, operujący na interwale dziennym (1D). 

System łączy algorytmiczną filtrację płynności, 5-kryteriową matrycę konfluencji technicznej, twarde filtry zarządzania ryzykiem, weryfikację setupów przez multimodalnego Agenta AI (Gemini Flash) oraz powiadomienia Telegram. Silnik symulacji wstecznej (`backtest.py`) jest w 100% zsynchronizowany ze skanerem produkcyjnym (`scanner.py`).

---

## 🏗️ Architektura Projektu

```text
├── .github/
│   └── workflows/
│       └── run_scanner.yml   # Workflow GitHub Actions (Pn-Pt 17:35 CET)
├── .env.example              # Wzór zmiennych środowiskowych (Telegram + Gemini API)
├── ai_auditor.py             # Multimodalny Agent AI (Gemini Flash jako Adwokat Diabła)
├── backtest.py               # Silnik Walk-Forward Day-by-Day (1:1 z logiką skanera)
├── chart_generator.py        # Bezgłowe generowanie wykresów technicznych (mplfinance)
├── config.yaml               # Konfiguracja parametrów, progów scoringu i filtrów
├── indicators.py             # Matematyka wskaźników (BB, RSI, MACD, MA, Fibo, PA)
├── main.py                   # Główny orkiestrator CLI i dystrybucji alertów
├── notifier.py               # Integracja z API Telegrama (sendPhoto + Markdown)
├── requirements.txt          # Zależności Python
├── scanner.py                # Silnik skanera, filtracja płynności i scoring
├── test_scanner.py           # Zestaw testów jednostkowych (unittest)
└── tickers.csv               # Baza >1100 instrumentów (ETF UCITS + globalne akcje)
```

---

## ⚡ 5 Kryteriów Matrycy Konfluencji

Alert i sygnał transakcyjny generowany jest wyłącznie wtedy, gdy dany instrument spełnia **co najmniej 3 z poniższych 5 kryteriów konfluencji**:

1. **Wyprzedanie (Oversold):** `RSI(14) < 35` LUB powrót RSI ponad poziom 30 w ciągu ostatnich 2 sesji (`RSI[T] >= 30` i `RSI[T-1] < 30`).
2. **Poziom / Geometria:** Dzisiejsze Low uderzyło w dolną wstęgę Bollingera (BB 20, 2) LUB kurs (Low/Close) testuje dynamiczną strefę zniesienia Fibonacciego (0.500 lub 0.618 z ostatnich 60 sesji z marginesem +/- 1%).
3. **Price Action:** Wykryto świecę Młota (Pinbar o dolnym knocie $\ge 2\times$ korpus) LUB formację Objęcia Hossy (Bullish Engulfing).
4. **Wolumen absorpcyjny:** Bieżący wolumen relatywny `RVol >= 1.4x` (w stosunku do 20-sesyjnej średniej SMA Volume).
5. **Impet (MACD):** Wygaszanie pędu podaży (krótszy ujemny słupek histogramu MACD: `Hist[T-1] < 0` oraz `Hist[T] > Hist[T-1]`).

---

## 💧 Adaptacyjny Filtr Płynności (ETF vs Akcje)

Z uwagi na różnice w mikrostrukturze rynku europejskiego i amerykańskiego, filtr płynności w `scanner.py` i `backtest.py` działa adaptacyjnie:
* **Europejskie fundusze ETF UCITS (XETRA, LSE):** Płynność na europejskich ETF-ach w dużej mierze animowana jest poza arkuszem przez Autoryzowanych Uczestników (AP) i platformy RFQ/OTC, a cena jednostkowa bywa wysoka (np. 100–600 EUR za jednostkę). Wymóg to min. **3 000 jednostek** wolumenu i min. **250 000 EUR** dziennego obrotu.
* **Akcje pojedyncze (USA / rynki globalne):** Wymóg min. **30 000 akcji** oraz min. **1 000 000 USD** średniego obrotu dziennego.

---

## 🎯 Moduł Zarządzania Ryzykiem (Hard Risk Filters)

Dla każdego kandydata system weryfikuje twarde reguły wejścia:
1. **Trend długoterminowy:** Warunek `Close > SMA(200)` (eliminacja łapania dołków w trendzie spadkowym).
2. **Anti-Crash (Akcje):** Odrzucenie spółek, które w dniu sygnału tąpnęły o więcej niż `-6.0%` (ochrona przed lukami po wynikach finansowych).
3. **Anti-Chasing:** Zasięg korpusu lub zmiana dzienna świecy nie może przekraczać `3.5%` (zakaz gonienia uciekającego rynku).
4. **Twardy Stop Loss:** Poziom 0.5% poniżej dołka świecy/formacji lub wsparcia Fibo (odległość SL musi wynosić od `0.8%` do `6.0%`).
5. **Take Profit 1 & 2:**
   - **TP1 (50% pozycji):** Środkowa wstęga Bollingera SMA(20) (lub górna wstęga, jeśli kurs jest ponad średnią). Osiągnięcie TP1 powoduje przestawienie Stop Loss na Break-Even.
   - **TP2 (50% pozycji):** Szczyt 60-sesyjny (Fibo 0.0) lub górna wstęga Bollingera.
   - Minimalna relacja Risk/Reward: `R:R do TP1 >= 0.75` oraz `R:R do TP2 >= 1.60`.

---

## 🤖 Audytor AI (Gemini Flash jako Adwokat Diabła)

Zanim alert trafi na Telegram, zakwalifikowane setupy są renderowane jako wykres techniczny PNG (mplfinance) i przesyłane do multimodalnego modelu Google Gemini Flash (`ai_auditor.py`). 

Model otrzymuje systemową instrukcję bezwzględnego Risk Managera:
* Identyfikuje opory horyzontalne z ostatnich 2–3 miesięcy bezpośrednio nad kursem.
* Weryfikuje strukturę rynku (Higher Highs / Higher Lows).
* Sprawdza, czy świeca nie zostawiła długiego górnego knota odrzucenia.
* Zwraca ustrukturyzowany werdykt JSON (`PASS` / `REJECT`, ocenę zaufania, punkty krytyczne i plan IF->THEN).

---

## 📊 Silnik Testowania Wstecznego (`backtest.py`)

Silnik symulacji Walk-Forward Day-by-Day jest **w 100% zsynchronizowany z logiką produkcyjną `scanner.py`**:
* **Brak Lookahead Bias:** Obliczenia wskaźników i dynamicznych poziomów Fibonacciego opierają się wyłącznie na danych dostępnych do dnia $T$ włącznie (dowolny historyczny poziom Fibo jest liczony z causalnego okna 60 sesji wstecz).
* **Realistyczna Egzekucja:** Sygnał generowany jest na zamknięciu dnia $T$, a egzekucja następuje po cenie otwarcia dnia $T+1$ (`Open[T+1]`).
* **Konserwatywna Reguła Intraday:** Jeśli w ciągu tej samej sesji nastąpi kolizja `Low <= SL` oraz `High >= TP1`, backtest zawsze pesymistycznie zalicza stratę (SL).
* **Obsługa Luk Cenowych (Gap Risk):** Luki otwarcia poniżej SL rozliczane są po cenie faktycznego otwarcia sesji.
* **Rozbicie Wyników:** Raport uwzględnia podział na klasy aktywów (ETF vs Akcje) oraz stabilność w czasie (rozkład wg lat — analiza Out-of-Sample).

---

## 🚀 Szybki Start (Lokalnie)

### 1. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 2. Konfiguracja zmiennych środowiskowych
Skopiuj plik `.env.example` i wprowadź dane:
```bash
cp .env.example .env
nano .env
```
Wprowadź:
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TELEGRAM_CHAT_ID=123456789
GEMINI_API_KEY=AIzaSy...
```

### 3. Uruchomienie skanera i backtestu

* **Pełne skanowanie rynkowe:**
  ```bash
  python main.py
  ```

* **Test w trybie syntetycznym (bez dostępu do sieci):**
  ```bash
  python main.py --mock-data
  ```

* **Uruchomienie testów jednostkowych:**
  ```bash
  python -m unittest -v test_scanner.py
  ```

* **Uruchomienie symulacji wstecznej (Backtest):**
  ```bash
  python backtest.py --period 2y
  ```

---

## 🎓 Metodologia i Znane Ograniczenia Akademickie

1. **Survivorship Bias (Błąd przeżywalności):** Zbiór w `tickers.csv` bazuje na aktualnie notowanych instrumentach. W testach wieloletnich instrumenty wycofane z obrotu lub zlikwidowane w przeszłości nie są uwzględnione w darmowym feedzie Yahoo Finance.
2. **Strefy Czasowe i Godziny Sesji:** Giełdy w Europie zamykają się o 17:30 CET, podczas gdy giełdy w USA handlują do 22:00 CET (16:00 EST). Uruchomienie skanera o 17:35 CET pobiera dla akcji z USA świecę niepełną (intraday); pełna ocena rynku amerykańskiego powinna następować po 22:15 CET.
3. **Płynność Pozagiełdowa:** Dane giełdowe odzwierciedlają jedynie wolumen lit-market (na arkuszu). Rzeczywista płynność europejskich funduszy UCITS jest wyższa z uwagi na handel OTC i kwotowania animatorów rynku.
