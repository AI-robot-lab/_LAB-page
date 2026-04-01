# Wprowadzenie do Prompt Engineering

## Czym jest Prompt Engineering?

**Prompt Engineering** to dziedzina zajmująca się projektowaniem, optymalizacją i strukturyzowaniem zapytań (promptów) kierowanych do dużych modeli językowych (LLM) w celu uzyskania jak najlepszych, najbardziej precyzyjnych i użytecznych odpowiedzi.

W kontekście robotyki humanoidalnej umiejętność formułowania skutecznych promptów jest kluczowa — od sterowania zachowaniem robota przez język naturalny, po integrację LLM z systemami planowania zadań.

## Anatomia Dobrego Promptu

Skuteczny prompt składa się z kilku elementów:

```
┌─────────────────────────────────────┐
│  [ROLA / PERSONA]                   │
│  Kim jest model? Jaki ma kontekst?  │
├─────────────────────────────────────┤
│  [KONTEKST / TŁO]                   │
│  Dane wejściowe, stan środowiska    │
├─────────────────────────────────────┤
│  [ZADANIE / INSTRUKCJA]             │
│  Co dokładnie model ma wykonać?     │
├─────────────────────────────────────┤
│  [FORMAT WYJŚCIA]                   │
│  JSON, lista, kod, narracja?        │
└─────────────────────────────────────┘
```

## Rodzaje Promptów

### Zero-Shot Prompt

Model odpowiada bez żadnych przykładów — bazuje wyłącznie na wiedzy z treningu.

```python
prompt = """
Jesteś asystentem robota humanoidalnego.
Użytkownik powiedział: "Przynieś mi herbatę".
Wygeneruj plan akcji dla robota.
"""
```

### One-Shot Prompt

Jeden przykład wskazujący modelowi oczekiwany format lub styl odpowiedzi.

```python
prompt = """
Przetłumacz polecenie na akcje robota.

Przykład:
Polecenie: "Idź do kuchni"
Akcje: navigate(location="kitchen")

Teraz:
Polecenie: "Przynieś mi herbatę"
Akcje:
"""
```

### Few-Shot Prompt

Kilka przykładów (zwykle 3–10) dla lepszej generalizacji.

```python
prompt = """
Przetłumacz polecenia na akcje robota.

Polecenie: "Idź do salonu"
Akcje: navigate(location="living_room")

Polecenie: "Podnieś kubek"
Akcje: grasp(object="cup", surface="table")

Polecenie: "Powiedz cześć"
Akcje: speak(text="Cześć! Jak mogę pomóc?")

Polecenie: "Przynieś mi herbatę"
Akcje:
"""
```

## Zasady Skutecznego Promptowania

### 1. Precyzja i jednoznaczność

```python
# Słaby prompt
bad_prompt = "Opisz robota"

# Dobry prompt
good_prompt = """
Opisz robota humanoidalnego Unitree G1 w 3 zdaniach,
uwzględniając: liczbę stopni swobody, zastosowanie i
przybliżoną wagę. Pisz po polsku.
"""
```

### 2. Podaj kontekst

```python
context_prompt = """
Kontekst: Jesteś modułem kognitywnym robota humanoidalnego
pracującego w laboratorium badawczym. Robot ma dostęp do
kamery RGB-D, czujników dotyku i mikrofonu.

Aktualny stan: robot stoi przy stole warsztatowym.
Bateria: 75%. Czas pracy: 2h 15min.

Zadanie: {task}
"""
```

### 3. Określ format wyjścia

```python
format_prompt = """
Przeanalizuj sytuację i odpowiedz w formacie JSON:

{{
    "analiza": "krótki opis sytuacji",
    "ryzyko": "niskie|średnie|wysokie",
    "akcje": [
        {{"nazwa": "akcja_1", "priorytet": 1}},
        {{"nazwa": "akcja_2", "priorytet": 2}}
    ],
    "uzasadnienie": "dlaczego te akcje"
}}

Sytuacja: {situation}
"""
```

### 4. Iteracyjne doskonalenie

```python
def refine_prompt(initial_prompt, feedback, iteration=1):
    """
    Iteracyjne ulepszanie promptu na podstawie wyników
    """
    refined = f"""
{initial_prompt}

Poprzednia odpowiedź była niewystarczająca ponieważ: {feedback}
Proszę udziel lepszej odpowiedzi uwzględniając powyższe uwagi.
"""
    return refined
```

## Metryki Jakości Promptów

```python
def evaluate_prompt_quality(prompt, responses, criteria):
    """
    Ocena jakości promptu na podstawie uzyskanych odpowiedzi

    criteria: dict z wagami dla każdego kryterium
    Przykład: {"accuracy": 0.4, "format": 0.3, "completeness": 0.3}
    """
    scores = []

    for response in responses:
        score = 0.0

        if criteria.get("accuracy"):
            # Sprawdź poprawność merytoryczną
            accuracy = check_accuracy(response)
            score += accuracy * criteria["accuracy"]

        if criteria.get("format"):
            # Sprawdź zgodność z formatem
            format_ok = check_format(response)
            score += format_ok * criteria["format"]

        if criteria.get("completeness"):
            # Sprawdź kompletność odpowiedzi
            completeness = check_completeness(response)
            score += completeness * criteria["completeness"]

        scores.append(score)

    return {
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "pass_rate": sum(1 for s in scores if s >= 0.7) / len(scores)
    }
```

## Typowe Błędy w Prompt Engineering

| Błąd | Problem | Rozwiązanie |
|------|---------|-------------|
| Zbyt ogólne polecenie | Niejednoznaczne wyniki | Dodaj kontekst i ograniczenia |
| Brak formatu wyjścia | Trudna parsowanie | Określ strukturę JSON/Markdown |
| Zbyt długi prompt | Gubienie kontekstu | Skróć, zostaw tylko to co istotne |
| Sprzeczne instrukcje | Niezdefiniowane zachowanie | Sprawdź logiczną spójność |
| Brak przykładów | Błędna interpretacja | Dodaj few-shot examples |

## Narzędzia do Testowania Promptów

```python
class PromptTester:
    def __init__(self, llm_client):
        self.client = llm_client
        self.results = []

    def test(self, prompt_template, test_cases, expected_fn):
        """
        Przetestuj prompt na zestawie przypadków testowych

        expected_fn: funkcja(odpowiedź) → bool sprawdzająca poprawność
        """
        passed = 0

        for case in test_cases:
            prompt = prompt_template.format(**case["inputs"])
            response = self.client.generate(prompt)
            ok = expected_fn(response, case.get("expected"))

            self.results.append({
                "case": case["name"],
                "passed": ok,
                "response": response
            })

            if ok:
                passed += 1

        print(f"Wyniki: {passed}/{len(test_cases)} zaliczonych")
        return passed / len(test_cases)

# Przykład użycia
tester = PromptTester(llm_client)

test_cases = [
    {
        "name": "polecenie_nawigacji",
        "inputs": {"command": "Idź do kuchni"},
        "expected": "navigate"
    },
    {
        "name": "polecenie_chwytu",
        "inputs": {"command": "Podnieś kubek"},
        "expected": "grasp"
    }
]

score = tester.test(
    prompt_template="Przetłumacz polecenie na akcję robota.\nPolecenie: {command}\nAkcja:",
    test_cases=test_cases,
    expected_fn=lambda resp, exp: exp.lower() in resp.lower()
)
```

## Powiązane Artykuły

- [Few-Shot Prompting](#wiki-prompt-few-shot)
- [Chain-of-Thought Prompting](#wiki-prompt-chain-of-thought)
- [Prompty Systemowe i Role Prompting](#wiki-prompt-system-roles)
- [RAG i Prompt Engineering](#wiki-prompt-rag)
- [Zaawansowane Techniki Promptowania](#wiki-prompt-advanced)
- [Large Language Models (LLM)](#wiki-llm)

---

*Ostatnia aktualizacja: 2025-04-01*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
