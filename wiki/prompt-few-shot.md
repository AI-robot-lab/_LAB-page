# Few-Shot Prompting

## Wprowadzenie

**Few-Shot Prompting** to technika, w której do promptu dołączamy kilka przykładów (tzw. „shotów") ilustrujących oczekiwane zachowanie modelu. Dzięki temu LLM uczy się wzorca odpowiedzi bez dodatkowego trenowania — wyłącznie na podstawie przykładów w kontekście.

## Jak Działa Few-Shot Learning

```
┌──────────────────────────────────────┐
│  Przykład 1: input → output          │
│  Przykład 2: input → output          │
│  Przykład 3: input → output          │
│  ─────────────────────────────────   │
│  Nowe zapytanie: input → ???         │
└──────────────────────────────────────┘
```

Model rozpoznaje wzorzec z przykładów i stosuje go do nowego zapytania — bez aktualizacji wag.

## Implementacja w Robotyce

### Tłumaczenie Poleceń na Akcje

```python
def build_few_shot_prompt(user_command):
    examples = [
        ("Idź do kuchni",         "navigate(location='kitchen')"),
        ("Podnieś kubek z stołu", "grasp(object='cup', surface='table')"),
        ("Powiedz cześć",         "speak(text='Cześć! Jak mogę pomóc?')"),
        ("Zatrzymaj się",         "stop()"),
        ("Odłóż przedmiot",       "release(surface='table')"),
    ]

    prompt = "Przetłumacz polecenie użytkownika na akcję robota.\n\n"
    for cmd, action in examples:
        prompt += f"Polecenie: {cmd}\nAkcja: {action}\n\n"

    prompt += f"Polecenie: {user_command}\nAkcja:"
    return prompt
```

### Klasyfikacja Intencji

```python
intent_examples = [
    ("Przynieś mi wodę",         "fetch_object"),
    ("Gdzie jest moja torba?",   "find_object"),
    ("Jak się nazywasz?",        "answer_question"),
    ("Zatrzymaj się natychmiast","emergency_stop"),
    ("Otwórz drzwi",             "manipulate_environment"),
]

def classify_intent(utterance, examples):
    prompt = "Sklasyfikuj intencję wypowiedzi użytkownika.\n\n"
    for text, intent in examples:
        prompt += f"Wypowiedź: \"{text}\"\nIntencja: {intent}\n\n"
    prompt += f"Wypowiedź: \"{utterance}\"\nIntencja:"
    return prompt
```

## Dobór Przykładów

### Zasada różnorodności

```python
import random

def select_diverse_examples(example_pool, n=5, query=None):
    """
    Wybierz n różnorodnych przykładów z puli.
    Opcjonalnie preferuj podobne do zapytania (semantic retrieval).
    """
    if query and len(example_pool) > n * 3:
        # Proste dopasowanie słów kluczowych
        scored = []
        query_words = set(query.lower().split())
        for ex in example_pool:
            ex_words = set(ex[0].lower().split())
            overlap = len(query_words & ex_words)
            scored.append((overlap, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Mix: połowa podobnych, połowa losowych
        top = [ex for _, ex in scored[:n // 2]]
        rest = random.sample(
            [ex for _, ex in scored[n // 2:]], n - len(top)
        )
        return top + rest
    return random.sample(example_pool, min(n, len(example_pool)))
```

### Kolejność przykładów

Badania pokazują, że kolejność przykładów wpływa na jakość odpowiedzi:

```python
def order_examples_by_similarity(examples, query, embedder):
    """
    Przykłady najbardziej podobne do zapytania umieść na końcu
    (efekt recency — model lepiej pamięta ostatnie przykłady).
    """
    query_emb = embedder.encode(query)
    scored = []
    for ex in examples:
        ex_emb = embedder.encode(ex[0])
        sim = cosine_similarity(query_emb, ex_emb)
        scored.append((sim, ex))
    scored.sort(key=lambda x: x[0])  # rosnąco — najbardziej podobny na końcu
    return [ex for _, ex in scored]
```

## Few-Shot dla Strukturyzowanych Odpowiedzi

```python
structured_examples = [
    {
        "input": "Użytkownik prosi o herbatę",
        "output": {
            "intent": "fetch_beverage",
            "object": "tea",
            "priority": "normal",
            "actions": ["navigate(kitchen)", "grasp(kettle)", "pour(cup)"]
        }
    },
    {
        "input": "Użytkownik upuścił coś na podłogę",
        "output": {
            "intent": "pick_up",
            "object": "unknown",
            "priority": "high",
            "actions": ["navigate_to_object()", "grasp(floor_object)"]
        }
    }
]

import json

def build_structured_few_shot(scenario, examples):
    prompt = "Przeanalizuj scenariusz i odpowiedz w formacie JSON.\n\n"
    for ex in examples:
        prompt += f"Scenariusz: {ex['input']}\n"
        prompt += f"Odpowiedź: {json.dumps(ex['output'], ensure_ascii=False)}\n\n"
    prompt += f"Scenariusz: {scenario}\nOdpowiedź:"
    return prompt
```

## Powiązane Artykuły

- [Wprowadzenie do Prompt Engineering](#wiki-prompt-engineering-intro)
- [Chain-of-Thought Prompting](#wiki-prompt-chain-of-thought)
- [Zaawansowane Techniki Promptowania](#wiki-prompt-advanced)
- [Large Language Models (LLM)](#wiki-llm)

---

*Ostatnia aktualizacja: 2025-04-01*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
