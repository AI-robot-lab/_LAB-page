# GitHub Webhooks — automatyczne powiadomienia z repozytoriów

GitHub Webhooks to mechanizm umożliwiający automatyczne wysyłanie powiadomień HTTP do zewnętrznego serwera w momencie wystąpienia określonego zdarzenia w repozytorium. Pozwalają integrować repozytoria GitHub z dowolnymi usługami zewnętrznymi — systemami CI/CD, komunikatorami, bazami danych, monitoringiem czy własnymi aplikacjami — bez potrzeby ciągłego odpytywania API.

---

## 1. Czym są webhooks?

**Webhook** to żądanie HTTP POST wysyłane automatycznie przez GitHub do wskazanego adresu URL (tzw. *payload URL*) w odpowiedzi na zdarzenie w repozytorium. Zamiast aplikacja pytała GitHub „czy coś się zmieniło?", GitHub sam informuje aplikację: „właśnie coś się wydarzyło".

Webhooks są często porównywane do odwróconego API:

| Klasyczne API (polling) | Webhook (push) |
|---|---|
| Klient regularnie odpytuje serwer | Serwer sam informuje klienta |
| Opóźnienie zależy od częstotliwości odpytywania | Powiadomienie następuje natychmiast po zdarzeniu |
| Duże zużycie zasobów przy braku zmian | Żądanie tylko wtedy, gdy coś się wydarzy |
| Prostsze do zaimplementowania po stronie klienta | Wymaga publicznego endpointu odbierającego żądania |

### Typowe zastosowania

- uruchamianie pipeline CI/CD po `push` lub `pull_request`,
- wysyłanie powiadomień na Slacka, Discorda lub Teams,
- synchronizacja danych z zewnętrzną bazą danych,
- aktualizowanie dashboardów monitoringu,
- automatyczne wdrożenia (deploy) po scaleniu z gałęzią `main`,
- rejestrowanie zdarzeń w systemach audytu.

---

## 2. Jak działa webhook?

### 2.1. Przebieg zdarzenia

1. W repozytorium następuje zdarzenie (np. ktoś wysyła commit).
2. GitHub tworzy ładunek JSON (*payload*) z opisem zdarzenia.
3. GitHub wysyła żądanie HTTP POST na skonfigurowany *payload URL*.
4. Serwer odbierający żądanie przetwarza dane i wykonuje określone akcje.
5. Serwer odpowiada kodem HTTP `2xx` (np. `200 OK`), potwierdzając odbiór.
6. Jeśli odpowiedź nie nadejdzie w ciągu 10 sekund, GitHub uznaje dostarczenie za nieudane i ponowi próbę.

### 2.2. Mechanizm ponownych prób

GitHub stosuje wykładnicze opóźnienie między kolejnymi próbami. Łącznie podejmuje maksymalnie kilkadziesiąt prób przez kilka dni. Każda nieudana i udana próba jest zapisywana w dzienniku dostaw (*delivery log*) dostępnym w ustawieniach webhooka.

### 2.3. Nagłówki żądania

Każde żądanie webhook zawiera standardowy zestaw nagłówków HTTP:

| Nagłówek | Opis |
|---|---|
| `X-GitHub-Event` | Nazwa zdarzenia, np. `push`, `pull_request` |
| `X-GitHub-Delivery` | Unikatowy identyfikator GUID dostarczenia |
| `X-Hub-Signature-256` | Podpis HMAC-SHA256 treści żądania |
| `Content-Type` | Zazwyczaj `application/json` |
| `User-Agent` | `GitHub-Hookshot/<hash>` |

---

## 3. Tworzenie webhooka w repozytorium

### 3.1. Przez interfejs GitHub

1. Przejdź do repozytorium i otwórz **Settings**.
2. W menu po lewej stronie wybierz **Webhooks**.
3. Kliknij **Add webhook**.
4. Wypełnij formularz:
   - **Payload URL** — adres URL serwera, który będzie odbierał żądania,
   - **Content type** — wybierz `application/json`,
   - **Secret** — opcjonalne hasło do podpisywania ładunku (zalecane),
   - **SSL verification** — pozostaw włączone (wymagane dla adresów HTTPS),
   - **Which events** — wybierz zdarzenia do nasłuchiwania.
5. Kliknij **Add webhook**.

### 3.2. Przez GitHub API

```bash
curl -X POST \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/<OWNER>/<REPO>/hooks \
  -d '{
    "name": "web",
    "active": true,
    "events": ["push", "pull_request"],
    "config": {
      "url": "https://example.com/webhook",
      "content_type": "json",
      "secret": "moje_tajne_haslo",
      "insecure_ssl": "0"
    }
  }'
```

### 3.3. Lokalny serwer deweloperski za pomocą ngrok

Podczas lokalnego developmentu serwer nie ma publicznego adresu IP. Narzędzie **ngrok** tworzy tunel HTTP do lokalnego portu:

```bash
# Instalacja (Linux/macOS)
brew install ngrok          # macOS
sudo snap install ngrok     # Ubuntu

# Uruchomienie tunelu na porcie 3000
ngrok http 3000
```

Po uruchomieniu ngrok wyświetli publiczny adres URL (np. `https://abc123.ngrok.io`), który można wpisać jako *payload URL* webhooka.

---

## 4. Zdarzenia (events)

GitHub obsługuje kilkadziesiąt typów zdarzeń. Poniżej najważniejsze z nich:

| Zdarzenie | Opis |
|---|---|
| `push` | Wysłanie commitów do gałęzi lub tagu |
| `pull_request` | Otwarcie, zamknięcie, scalenie lub aktualizacja PR |
| `issues` | Tworzenie, edycja, zamknięcie zgłoszenia |
| `issue_comment` | Nowy komentarz pod zgłoszeniem lub PR |
| `create` | Utworzenie gałęzi lub tagu |
| `delete` | Usunięcie gałęzi lub tagu |
| `release` | Publikacja, edycja lub usunięcie release'a |
| `workflow_run` | Zakończenie przebiegu GitHub Actions |
| `check_run` | Aktualizacja status check |
| `deployment` | Nowe wdrożenie |
| `deployment_status` | Zmiana statusu wdrożenia |
| `fork` | Sklonowanie (fork) repozytorium |
| `star` | Dodanie lub usunięcie gwiazdki |
| `member` | Zmiana uprawnień współpracownika |
| `repository` | Zmiany w ustawieniach repozytorium |

Aby odbierać wszystkie zdarzenia, wybierz opcję **Send me everything** w ustawieniach webhooka.

---

## 5. Struktura ładunku (payload)

Ładunek webhooka to obiekt JSON. Każde zdarzenie ma własny schemat, ale wszystkie zawierają wspólne pola:

```json
{
  "action": "opened",
  "sender": {
    "login": "jan-kowalski",
    "type": "User"
  },
  "repository": {
    "id": 123456789,
    "name": "my-robot-project",
    "full_name": "ai-robot-lab/my-robot-project",
    "html_url": "https://github.com/ai-robot-lab/my-robot-project",
    "default_branch": "main"
  },
  "organization": {
    "login": "ai-robot-lab"
  }
}
```

### 5.1. Przykład ładunku dla zdarzenia `push`

```json
{
  "ref": "refs/heads/main",
  "before": "abc123...",
  "after": "def456...",
  "commits": [
    {
      "id": "def456...",
      "message": "Add sensor fusion module",
      "author": {
        "name": "Jan Kowalski",
        "email": "jan@example.com"
      },
      "timestamp": "2026-03-20T12:00:00+01:00",
      "added": ["src/sensor_fusion.py"],
      "modified": ["README.md"],
      "removed": []
    }
  ],
  "pusher": {
    "name": "jan-kowalski",
    "email": "jan@example.com"
  },
  "repository": { "..." : "..." }
}
```

### 5.2. Przykład ładunku dla zdarzenia `pull_request`

```json
{
  "action": "opened",
  "number": 42,
  "pull_request": {
    "title": "Feature: add IMU integration",
    "state": "open",
    "head": { "ref": "feature/imu", "sha": "abc..." },
    "base": { "ref": "main", "sha": "def..." },
    "user": { "login": "anna-nowak" },
    "merged": false,
    "draft": false
  },
  "repository": { "..." : "..." }
}
```

---

## 6. Weryfikacja podpisu (bezpieczeństwo)

Każdy webhook powinien być zabezpieczony podpisem HMAC. Bez weryfikacji podpisu dowolny podmiot mógłby wysyłać fałszywe żądania na endpoint.

### 6.1. Jak działa podpis

1. Podczas konfiguracji webhooka ustaw wartość **Secret** (losowy ciąg znaków).
2. GitHub oblicza `HMAC-SHA256(secret, body)` i umieszcza wynik w nagłówku `X-Hub-Signature-256`.
3. Twój serwer samodzielnie oblicza ten sam HMAC z otrzymanego ciała żądania.
4. Porównaj obliczoną wartość z wartością nagłówka — jeśli są zgodne, żądanie pochodzi z GitHub.

### 6.2. Weryfikacja w Pythonie

```python
import hashlib
import hmac
from flask import Flask, request, abort

app = Flask(__name__)

WEBHOOK_SECRET = b"moje_tajne_haslo"  # wartość z konfiguracji webhooka


def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    """Weryfikuje podpis HMAC-SHA256 ładunku webhooka."""
    if not signature_header:
        return False
    hash_algorithm, github_signature = signature_header.split("=", 1)
    if hash_algorithm != "sha256":
        return False
    expected = hmac.new(WEBHOOK_SECRET, payload_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, github_signature)


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(request.data, signature):
        abort(403)

    event = request.headers.get("X-GitHub-Event")
    payload = request.get_json()

    if event == "push":
        branch = payload.get("ref", "").split("/")[-1]
        print(f"Push na gałąź: {branch}")
        # ... obsługa zdarzenia push

    elif event == "pull_request":
        action = payload.get("action")
        pr_title = payload["pull_request"]["title"]
        print(f"Pull request [{action}]: {pr_title}")
        # ... obsługa pull requesta

    return "", 200


if __name__ == "__main__":
    app.run(port=3000)
```

### 6.3. Weryfikacja w Node.js

```javascript
const express = require('express');
const crypto = require('crypto');

const app = express();
const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET;

// Pobieramy surowe ciało żądania przed parsowaniem JSON
app.use(express.raw({ type: 'application/json' }));

function verifySignature(payload, signatureHeader) {
  if (!signatureHeader) return false;
  const [algorithm, githubSignature] = signatureHeader.split('=');
  if (algorithm !== 'sha256') return false;
  const expected = crypto
    .createHmac('sha256', WEBHOOK_SECRET)
    .update(payload)
    .digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(expected),
    Buffer.from(githubSignature)
  );
}

app.post('/webhook', (req, res) => {
  const signature = req.headers['x-hub-signature-256'] || '';
  if (!verifySignature(req.body, signature)) {
    return res.status(403).send('Forbidden');
  }

  const event = req.headers['x-github-event'];
  const payload = JSON.parse(req.body);

  if (event === 'push') {
    const branch = payload.ref.split('/').pop();
    console.log(`Push na gałąź: ${branch}`);
  } else if (event === 'pull_request') {
    const { action } = payload;
    const { title } = payload.pull_request;
    console.log(`Pull request [${action}]: ${title}`);
  }

  res.status(200).send('OK');
});

app.listen(3000, () => console.log('Webhook server running on port 3000'));
```

---

## 7. Praktyczny przykład — powiadomienie na Discordzie

Poniższy skrypt w Pythonie odbiera webhook GitHub i wysyła powiadomienie na kanał Discord przez Discord Webhook URL.

```python
import hashlib
import hmac
import requests
from flask import Flask, request, abort

app = Flask(__name__)

WEBHOOK_SECRET = b"moje_tajne_haslo"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/<ID>/<TOKEN>"


def verify_signature(body: bytes, header: str) -> bool:
    if not header:
        return False
    algo, sig = header.split("=", 1)
    expected = hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


@app.route("/webhook", methods=["POST"])
def webhook():
    if not verify_signature(request.data, request.headers.get("X-Hub-Signature-256", "")):
        abort(403)

    event = request.headers.get("X-GitHub-Event")
    payload = request.get_json()

    message = None

    if event == "push":
        repo = payload["repository"]["full_name"]
        branch = payload["ref"].split("/")[-1]
        pusher = payload["pusher"]["name"]
        commits = len(payload.get("commits", []))
        message = f"🚀 **{pusher}** wysłał {commits} commit(ów) do `{branch}` w `{repo}`"

    elif event == "pull_request" and payload.get("action") == "opened":
        repo = payload["repository"]["full_name"]
        title = payload["pull_request"]["title"]
        author = payload["pull_request"]["user"]["login"]
        url = payload["pull_request"]["html_url"]
        message = f"🔀 Nowy PR w `{repo}` od **{author}**: [{title}]({url})"

    if message:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})

    return "", 200
```

---

## 8. Dobre praktyki

### 8.1. Bezpieczeństwo

- **Zawsze konfiguruj Secret** i weryfikuj podpis `X-Hub-Signature-256` przed przetworzeniem ładunku.
- Używaj **HTTPS** dla *payload URL* — GitHub blokuje możliwość wyłączenia weryfikacji SSL dla produkcyjnych webhooków.
- Przechowuj wartość Secret w zmiennej środowiskowej lub menedżerze sekretów (GitHub Secrets, Vault, AWS Secrets Manager), a nie w kodzie źródłowym.
- Stosuj **`hmac.compare_digest`** (Python) lub **`crypto.timingSafeEqual`** (Node.js) do porównania podpisów — zwykłe porównanie stringów jest podatne na ataki czasowe.

### 8.2. Niezawodność

- Endpoint powinien odpowiadać w ciągu **10 sekund**. Jeśli przetwarzanie trwa dłużej, zwróć `200 OK` natychmiast i przekaż zadanie do kolejki (np. Celery, Bull, RabbitMQ).
- Projektuj obsługę zdarzeń jako **idempotentną** — to samo zdarzenie może dotrzeć więcej niż raz. Używaj `X-GitHub-Delivery` jako klucza deduplikacji.
- Loguj każde odebrane zdarzenie z identyfikatorem dostarczenia, by ułatwić debugowanie.

### 8.3. Konfiguracja

- Subskrybuj tylko te zdarzenia, których faktycznie potrzebujesz — nie używaj opcji *Send me everything*, jeśli obsługujesz tylko kilka typów zdarzeń.
- Regularnie sprawdzaj dziennik dostaw w **Settings → Webhooks** i reaguj na powtarzające się błędy.
- W środowiskach testowych korzystaj z narzędzia **ngrok** lub **smee.io** do tunelowania żądań do lokalnego serwera.

---

## 9. Testowanie i debugowanie

### 9.1. Ponowne dostarczenie (Redeliver)

W ustawieniach webhooka, w zakładce **Recent Deliveries**, możesz ręcznie ponowić dostarczenie każdego wcześniejszego zdarzenia. Przydatne podczas testowania zmian w serwerze.

### 9.2. smee.io — kanał testowy

**smee.io** to bezpłatna usługa do przekazywania żądań webhook do lokalnego serwera:

```bash
# Instalacja klienta smee
npm install --global smee-client

# Przekazywanie zdarzeń z kanału smee na lokalny port 3000
smee --url https://smee.io/<UNIKALNY_ID> --target http://localhost:3000/webhook
```

1. Wejdź na [smee.io](https://smee.io) i utwórz nowy kanał.
2. Skopiuj URL kanału i wpisz go jako *payload URL* webhooka w GitHub.
3. Uruchom klienta smee — wszystkie zdarzenia będą przekazywane do lokalnego serwera.

### 9.3. Logi w GitHub

W sekcji **Settings → Webhooks → Recent Deliveries** możesz dla każdego dostarczenia zobaczyć:

- nagłówki żądania,
- ciało żądania (ładunek JSON),
- odpowiedź serwera (kod HTTP i nagłówki),
- czas trwania żądania.

---

## 10. Webhook na poziomie organizacji

Webhook można skonfigurować nie tylko dla pojedynczego repozytorium, ale dla całej organizacji GitHub. Taki webhook odbiera zdarzenia ze wszystkich repozytoriów należących do organizacji.

```bash
curl -X POST \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/orgs/<ORG>/hooks \
  -d '{
    "name": "web",
    "active": true,
    "events": ["push", "pull_request", "issues"],
    "config": {
      "url": "https://example.com/org-webhook",
      "content_type": "json",
      "secret": "tajne_haslo_org"
    }
  }'
```

Zdarzenie na poziomie organizacji zawiera dodatkowe pole `organization` w ładunku.

---

## 11. Porównanie webhooków z GitHub Actions

Webhooks i GitHub Actions często służą podobnym celom. Warto wiedzieć, kiedy wybrać każde z rozwiązań:

| Kryterium | Webhook | GitHub Actions |
|---|---|---|
| Gdzie jest logika | Na własnym serwerze | W repozytorium (YAML) |
| Dostęp do zasobów | Własna infrastruktura | Maszyny GitHub (lub self-hosted runner) |
| Integracja z zewnętrznym systemem | Naturalna | Przez akcje lub skrypty |
| Wymagana infrastruktura | Publiczny endpoint HTTP | Brak (zarządzane przez GitHub) |
| Koszt | Własny serwer | Bezpłatne minuty w planie Free/Pro |
| Złożoność konfiguracji | Wyższe | Niższe — plik YAML w repo |
| Powiadomienia w czasie rzeczywistym | Tak | Tak (przez Actions) |

Ogólna zasada: jeśli chcesz uruchomić coś w **swoim** systemie, użyj webhooków. Jeśli chcesz wykonać coś w kontekście **repozytorium GitHub**, użyj GitHub Actions.

---

## 12. Podsumowanie

GitHub Webhooks to prosty i wydajny sposób integracji repozytoriów z zewnętrznymi systemami. Umożliwiają natychmiastową reakcję na zdarzenia bez konieczności ciągłego odpytywania API.

Kluczowe zasady do zapamiętania:

- zawsze konfiguruj Secret i weryfikuj podpis `X-Hub-Signature-256`,
- używaj HTTPS dla *payload URL*,
- odpowiadaj szybko (`200 OK`) i przetwarzaj zdarzenia asynchronicznie przy długich operacjach,
- projektuj obsługę zdarzeń jako idempotentną,
- subskrybuj tylko potrzebne zdarzenia,
- korzystaj z dziennika dostaw i narzędzi takich jak ngrok lub smee.io podczas debugowania.

## Powiązane artykuły

- [Git i GitHub — kompleksowy poradnik](#wiki-git-github)
- [Testy w GitHub — profesjonalny przewodnik dla studentów](#wiki-github-tests)
- [GitHub Releases — czym są i jak je tworzyć](#wiki-github-releases)
- [Praca w zespole inżynierskim](#wiki-praca-w-zespole)

## Zasoby

- GitHub Docs — Webhooks documentation
- GitHub Docs — Webhook events and payloads
- smee.io — Webhook payload delivery service
- ngrok — Secure tunnels to localhost

---
*Ostatnia aktualizacja: 2026-03-30*
*Autor: Codex / Zespół Laboratorium*
