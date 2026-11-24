# ETF & Equity Confluence Scanner

System do skanowania rynku i testowania strategii opartych na analizie technicznej. Projekt obejmuje europejskie fundusze ETF UCITS (XETRA `.DE`, LSE `.L`) oraz płynne spółki notowane na rynkach amerykańskich i europejskich.

Skaner pracuje na danych dziennych (1D) i szuka instrumentów, na których kilka niezależnych sygnałów technicznych wskazuje na możliwe odbicie wzrostowe.

Projekt składa się z kilku niezależnych modułów: pobierania i filtrowania danych, obliczania wskaźników, oceny sygnałów, zarządzania ryzykiem, generowania wykresów, audytu AI oraz wysyłania powiadomień przez Telegram.

Osobnym elementem jest `backtest.py`, który odtwarza historyczne działanie strategii z zachowaniem tych samych zasad, które obowiązują w produkcyjnym skanerze.

---

## Struktura projektu

```text
├── .github/
│   └── workflows/
│       └── run_scanner.yml   # Automatyczne uruchamianie skanera (Pn-Pt 17:35 CET)
├── .env.example              # Przykładowa konfiguracja zmiennych środowiskowych
├── ai_auditor.py             # Weryfikacja setupów przy użyciu Gemini Flash
├── backtest.py               # Historyczna symulacja strategii
├── chart_generator.py        # Generowanie wykresów technicznych (mplfinance)
├── config.yaml               # Parametry strategii i progi filtrów
├── indicators.py             # Obliczanie wskaźników technicznych
├── main.py                   # Główny punkt wejścia i obsługa całego procesu
├── notifier.py               # Wysyłanie alertów i wykresów przez Telegram
├── requirements.txt          # Zależności projektu
├── scanner.py                # Skanowanie rynku i ocena sygnałów
├── test_scanner.py           # Testy jednostkowe
└── tickers.csv               # Lista ponad 1100 instrumentów
```

## Jak działa strategia?

Podstawą systemu jest pięciokryterialna ocena konfluencji technicznej. Każdy instrument jest sprawdzany pod kątem pięciu niezależnych warunków.

Sygnał może zostać wygenerowany, jeśli spełnione są przynajmniej 3 z 5 kryteriów.

### 1. RSI – wyprzedanie rynku

Sprawdzane są dwa warianty:

- RSI(14) znajduje się poniżej 35.
- RSI wraca powyżej poziomu 30, po wcześniejszym zejściu poniżej tego poziomu w ciągu ostatnich dwóch sesji.

### 2. Poziomy techniczne

Cena musi znajdować się w pobliżu istotnego poziomu technicznego. Pod uwagę brane są:

- Kontakt dziennego minimum z dolną wstęgą Bollingera (20, 2).
- Test poziomów Fibonacciego 0.500 lub 0.618, wyznaczonych na podstawie ostatnich 60 sesji.

Dopuszczalne odchylenie od poziomu Fibo wynosi 1%.

### 3. Price Action

System szuka dwóch formacji mogących wskazywać na przewagę kupujących:

- Hammer – świeca z dolnym knotem o długości co najmniej dwukrotnie większej od korpusu.
- Bullish Engulfing – objęcie hossy.

### 4. Wolumen

Sprawdzany jest relatywny wolumen w stosunku do średniej z ostatnich 20 sesji.

Warunek zostaje spełniony, gdy:

`RVol >= 1.4`

Wyższy wolumen może wskazywać na zwiększoną aktywność kupujących w pobliżu lokalnego wsparcia.

### 5. MACD – zmiana momentum

Oceniane jest zachowanie histogramu MACD.

Kryterium zostaje spełnione, gdy histogram pozostaje ujemny, ale jego wartość jest wyższa niż podczas poprzedniej sesji:

`Hist[T-1] < 0` oraz `Hist[T] > Hist[T-1]`

Ma to wychwytywać sytuacje, w których presja spadkowa zaczyna słabnąć.

---

## Filtr płynności

Przed analizą techniczną każdy instrument przechodzi filtr płynności. Progi są różne dla ETF-ów i pojedynczych akcji, ponieważ oba rodzaje aktywów charakteryzują się inną strukturą obrotu.

### ETF-y UCITS – XETRA i LSE

Wymagania:

- Średni dzienny wolumen: minimum 3 000 jednostek.
- Średni dzienny obrót: minimum 250 000 EUR.

W przypadku europejskich ETF-ów sam wolumen giełdowy nie pokazuje całej dostępnej płynności. Część obrotu odbywa się poza arkuszem zleceń, m.in. przez animatorów rynku i transakcje OTC.

### Akcje – USA i pozostałe rynki

Wymagania:

- Średni dzienny wolumen: minimum 30 000 akcji.
- Średni dzienny obrót: minimum 1 000 000 USD.

Instrumenty niespełniające odpowiednich progów są odrzucane przed dalszą analizą.

---

## Zarządzanie ryzykiem

Samo uzyskanie odpowiedniej liczby punktów konfluencji nie wystarcza do wygenerowania sygnału. Każdy kandydat musi dodatkowo przejść zestaw filtrów ryzyka.

### Filtry wejścia

**Trend długoterminowy**

Cena zamknięcia musi znajdować się powyżej SMA(200):

`Close > SMA(200)`

Pozwala to ograniczyć liczbę sygnałów pojawiających się podczas silnych trendów spadkowych.

**Ochrona przed gwałtownymi spadkami**

W przypadku pojedynczych akcji odrzucane są instrumenty, których dzienna zmiana ceny wynosi mniej niż -6%.

**Ograniczenie pościgu za ceną**

Dzienne wybicie nie może przekraczać 3,5%. Filtr ma ograniczać wejścia po zbyt dynamicznych świecach, kiedy potencjalny punkt wejścia jest już niekorzystny.

### Stop Loss

Stop Loss wyznaczany jest na podstawie minimum świecy sygnałowej, formacji lub pobliskiego wsparcia Fibonacciego.

Poziom SL znajduje się 0,5% poniżej wybranego poziomu technicznego.

Dodatkowo odległość Stop Loss od ceny wejścia musi mieścić się w przedziale:

- Minimum: 0,8%.
- Maksimum: 6,0%.

Setupy, w których nie można wyznaczyć SL spełniającego te warunki, są odrzucane.

### Take Profit

Pozycja jest dzielona na dwie części.

**TP1 – realizacja 50% pozycji**

Pierwszym celem jest środkowa wstęga Bollingera (SMA 20). Jeżeli cena znajduje się już powyżej średniej, wykorzystywana jest górna wstęga.

Po osiągnięciu TP1 Stop Loss pozostałej części pozycji zostaje przesunięty na poziom Break-Even.

**TP2 – realizacja pozostałych 50%**

Drugi cel wyznaczany jest na podstawie:

- Szczytu z ostatnich 60 sesji (Fibo 0.0).
- Górnej wstęgi Bollingera.

### Minimalny stosunek zysku do ryzyka

Każdy setup musi spełniać oba warunki:

| Cel | Minimalny R:R |
|---|---:|
| TP1 | 0.75 |
| TP2 | 1.60 |

Jeżeli potencjalny zysk nie uzasadnia podejmowanego ryzyka, sygnał nie przechodzi dalej.

---

## Audyt setupów przez AI

Zakwalifikowane instrumenty przechodzą dodatkową weryfikację przy użyciu Gemini Flash.

Moduł `ai_auditor.py` otrzymuje wygenerowany wykres techniczny w formacie PNG i analizuje go z perspektywy dodatkowego filtra ryzyka.

AI nie odpowiada za generowanie sygnałów. Jego zadaniem jest próba znalezienia powodów, dla których dany setup może nie być wart zagrania.

Analiza obejmuje przede wszystkim:

- Potencjalne opory horyzontalne z ostatnich 2–3 miesięcy.
- Strukturę rynku, w tym układ Higher Highs i Higher Lows.
- Świece z długim górnym knotem, mogące wskazywać na odrzucenie wyższych cen.
- Ogólną jakość i czytelność układu technicznego.

Wynik zwracany jest w ustrukturyzowanym formacie JSON i zawiera:

- Decyzję `PASS` lub `REJECT`.
- Ocenę pewności.
- Najważniejsze zastrzeżenia.
- Plan działania w formacie IF → THEN.

Do dalszej dystrybucji trafiają wyłącznie setupy zaakceptowane przez audyt.

---

## Backtesting

Moduł `backtest.py` służy do sprawdzania, jak strategia zachowywałaby się na danych historycznych.

Symulacja działa w trybie Walk-Forward, przetwarzając kolejne sesje chronologicznie. Jej logika jest zgodna z produkcyjnym skanerem, dzięki czemu historyczne sygnały powstają według tych samych kryteriów co sygnały bieżące.

### Brak Lookahead Bias

Wskaźniki techniczne i poziomy Fibonacciego obliczane są wyłącznie na podstawie danych dostępnych w analizowanym dniu.

Poziomy Fibo wykorzystują kroczące okno 60 sesji, bez dostępu do przyszłych notowań.

### Moment wejścia w pozycję

Sygnał powstaje po zamknięciu sesji T.

Transakcja otwierana jest dopiero podczas kolejnej sesji, po cenie:

`Open[T+1]`

Pozwala to uniknąć nierealistycznego założenia, że można było wejść po cenie zamknięcia świecy, na podstawie której dopiero wygenerowano sygnał.

### Obsługa SL i TP

Backtest uwzględnia kilka istotnych scenariuszy:

- Jeżeli podczas jednej sesji cena osiągnie zarówno SL, jak i TP1, przyjmowany jest wariant niekorzystny, czyli realizacja Stop Loss.
- Jeżeli rynek otworzy się poniżej poziomu SL, transakcja rozliczana jest po faktycznej cenie otwarcia.
- Po realizacji TP1 połowa pozycji zostaje zamknięta, a SL pozostałej części przesuwany jest na Break-Even.

### Raportowanie wyników

Wyniki prezentowane są z podziałem na:

- ETF-y i pojedyncze akcje.
- Poszczególne lata.
- Stabilność wyników w czasie.
- Wyniki poszczególnych okresów testowych.

Podział roczny pozwala sprawdzić, czy strategia działa w różnych warunkach rynkowych, a nie tylko w jednym sprzyjającym okresie.

---

## Uruchamianie i automatyzacja

Głównym punktem wejścia do aplikacji jest `main.py`, który odpowiada za uruchomienie skanera i przekazanie wyników do kolejnych modułów.

Proces obejmuje pobranie danych, filtrację instrumentów, obliczenie wskaźników, ocenę konfluencji, weryfikację ryzyka, audyt AI i wysłanie zaakceptowanych alertów.

Powiadomienia zawierają informacje o instrumencie, parametrach transakcji oraz wykres techniczny.

Skaner może być uruchamiany lokalnie lub automatycznie przez GitHub Actions.

Domyślny harmonogram workflow to dni robocze o 17:35 CET.

### Konfiguracja

Przed pierwszym uruchomieniem należy:

1. Zainstalować zależności z `requirements.txt`.
2. Przygotować plik `.env` na podstawie `.env.example`.
3. Uzupełnić klucze API Gemini oraz dane dostępowe do Telegrama.
4. Sprawdzić parametry strategii w `config.yaml`.
5. Uruchomić testy jednostkowe.

---

## Ograniczenia i założenia

Projekt wykorzystuje dane historyczne i bieżące dostępne w publicznych źródłach, dlatego należy uwzględnić kilka ograniczeń.

### Survivorship Bias

Lista instrumentów w `tickers.csv` obejmuje przede wszystkim aktualnie notowane aktywa.

W rezultacie historyczne testy mogą nie uwzględniać spółek i ETF-ów, które zostały wcześniej wycofane z obrotu, zlikwidowane lub przestały być notowane.

Może to prowadzić do zawyżenia historycznych wyników strategii.

### Godziny sesji i strefy czasowe

Rynki europejskie i amerykańskie zamykają się o różnych godzinach.

Przy standardowym harmonogramie uruchamiania o 17:35 CET europejskie sesje są już zakończone, ale notowania amerykańskie nadal trwają.

Dane pobrane dla amerykańskich akcji mogą więc obejmować niepełną świecę dzienną.

Aby analizować kompletne dane z USA, skaner powinien być uruchamiany po zakończeniu tamtejszej sesji, np. około 22:15 CET (z uwzględnieniem zmian czasu).

### Rzeczywista płynność ETF-ów

Dane wolumenowe z Yahoo Finance nie pokazują całego obrotu na europejskich ETF-ach.

Część płynności zapewniają animatorzy rynku, Autoryzowani Uczestnicy oraz transakcje OTC i RFQ.

Oznacza to, że rzeczywista możliwość zawarcia transakcji może być większa, niż wynikałoby to z samego wolumenu giełdowego.

### Charakter wyników backtestu

Wyniki symulacji historycznej nie gwarantują podobnych rezultatów w przyszłości.

---

## Cel projektu

Celem projektu jest stworzenie powtarzalnego procesu selekcji instrumentów, w którym każdy sygnał przechodzi przez te same kryteria techniczne i zasady zarządzania ryzykiem.

Istotne jest nie tylko znalezienie potencjalnych okazji rynkowych, ale również odrzucenie setupów, w których relacja potencjalnego zysku do ryzyka jest niekorzystna.

System ma służyć jako narzędzie wspomagające analizę i selekcję transakcji, a nie jako gwarancja zyskowności czy automatyczny system inwestycyjny.
