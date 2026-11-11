# ETF Confluence Scanner 📈🇪🇺

Zaawansowany, modularny i odporny na błędy skaner rynkowy dla europejskich funduszy ETF UCITS (XETRA `.DE`, LSE `.L`), operujący na interwale dziennym (1D). System po zakończeniu sesji giełdowej w Europie pobiera dane zbiorczo, filtruje instrumenty pod kątem płynności, oblicza wskaźniki techniczne i wysyła powiadomienia przez Telegram wyłącznie w przypadku wykrycia silnego setupu odbicia (minimum 3 z 5 kryteriów konfluencji).

---

## 🏗️ Architektura Projektu

```text
├── .github/
│   └── workflows/
│       └── run_scanner.yml   # Workflow GitHub Actions (Pn-Pt 17:35 CET)
├── .env.example              # Wzór pliku zmiennych środowiskowych (Telegram Bot)
├── config.yaml               # Konfiguracja wskaźników, wag i progów scoringu
├── indicators.py             # Matematyka wskaźników (BB, RSI, MACD, MA, Fibo, PA)
├── main.py                   # Główny orkiestrator CLI
├── notifier.py               # Integracja z API Telegrama (Markdown + retry)
├── requirements.txt          # Zależności Python
├── scanner.py                # Silnik skanera, batch download, filtr płynności, scoring
├── test_scanner.py           # Testy jednostkowe (unittest)
└── tickers.csv               # Baza >50 płynnych europejskich ETF-ów UCITS
```

---

## ⚡ 5 Kryteriów Matrycy Konfluencji

Alert generowany jest wyłącznie wtedy, gdy dany ETF spełnia **co najmniej 3 z poniższych 5 kryteriów**:

1. **Wyprzedanie (Oversold):** `RSI(14) < 35` LUB RSI przecięło poziom 30 od dołu w górę w ciągu ostatnich 2 sesji.
2. **Poziom / Geometria:** Dzisiejsze Low uderzyło w dolną wstęgę Bollingera (BB 20, 2) LUB w strefę wsparcia Fibonacciego (0.500 / 0.618 z marginesem +/- 1%).
3. **Price Action:** Wykryto świecę Młota (Pinbar) LUB formację Objęcia Hossy (Bullish Engulfing).
4. **Wolumen absorpcyjny:** Bieżący wolumen relatywny `RVol >= 1.4x` (w stosunku do 20-sesyjnej średniej wolumenu).
5. **Impet (MACD):** Histogram MACD zmniejsza ujemną wartość (krótszy ujemny słupek = wygaszanie pędu podaży).

---

## 🎯 Moduł Zarządzania Ryzykiem (Risk Management)

Dla każdego zakwalifikowanego setupu skaner automatycznie wyznacza:
- **Cena Wejścia:** Bieżące zamknięcie sesji (`Close`).
- **Twardy Stop Loss:** 0.5% poniżej minimum bieżącej formacji świecowej / wsparcia Fibo.
- **Take Profit 1:** Środkowa wstęga SMA 20 (lub najbliższy opór).
- **Take Profit 2:** Poziom Fibo 0.0 (lokalne maksimum z ostatnich 60 sesji).
- **Stosunek R:R:** Matematyczna relacja Risk/Reward. Jeżeli `R:R do TP1 < 1.0`, raport zawiera wyraźne ostrzeżenie o podwyższonym ryzyku.

---

## 🚀 Szybki Start (Lokalnie)

### 1. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 2. Konfiguracja Telegrama
Skopiuj plik szablonu i wprowadź swój token oraz chat ID:
```bash
cp .env.example .env
nano .env
```
Wprowadź:
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TELEGRAM_CHAT_ID=123456789
```

### 3. Uruchomienie skanera

- **Pełne skanowanie rynkowe:**
  ```bash
  python main.py
  ```

- **Test połączenia z Telegramem (wysłanie próbnego alertu):**
  ```bash
  python main.py --test-alert
  ```

- **Tryb DRY-RUN (podgląd alertów w konsoli bez wysyłania do sieci):**
  ```bash
  python main.py --dry-run
  ```

- **Tryb testowy z danymi syntetycznymi (offline):**
  ```bash
  python main.py --mock-data
  ```

### 4. Uruchomienie testów jednostkowych
```bash
python -m unittest -v test_scanner.py
```

---

## 🤖 Automatyzacja na GitHub Actions

W repozytorium znajduje się plik `.github/workflows/run_scanner.yml`, który bezpłatnie uruchamia skaner w chmurze GitHub od poniedziałku do piątku o **17:35 CET** (po zamknięciu rynków europejskich).

### Konfiguracja Secrets w GitHub:
1. Przejdź do: **Settings -> Secrets and variables -> Actions**.
2. Dodaj dwa klucze Repository Secrets:
   - `TELEGRAM_BOT_TOKEN`: Token bota od @BotFather.
   - `TELEGRAM_CHAT_ID`: Identyfikator Twojego czatu lub kanału.
