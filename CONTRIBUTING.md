# Przewodnik dla Współtwórców

Dziękujemy za zainteresowanie współpracą przy projekcie Laboratorium Robotów Humanoidalnych! 🤖

## 📋 Spis treści

- [Jak mogę pomóc?](#jak-mogę-pomóc)
- [Zgłaszanie błędów](#zgłaszanie-błędów)
- [Proponowanie zmian](#proponowanie-zmian)
- [Proces Pull Request](#proces-pull-request)
- [Standardy kodu](#standardy-kodu)
- [Struktura projektu](#struktura-projektu)

## 🤝 Jak mogę pomóc?

Są różne sposoby, aby przyczynić się do projektu:

1. **Zgłaszanie błędów** - znalazłeś bug? Daj nam znać!
2. **Sugerowanie ulepszeń** - masz pomysł na nową funkcję?
3. **Poprawianie dokumentacji** - zauważyłeś błąd lub brakujące informacje?
4. **Pisanie kodu** - chcesz dodać nową funkcjonalność?
5. **Design** - masz pomysły na ulepszenie UI/UX?

## 🐛 Zgłaszanie błędów

Przed zgłoszeniem błędu:
1. Sprawdź [Issues](https://github.com/AI-robot-lab/ai-robot-lab.github.io/issues), czy problem nie został już zgłoszony
2. Upewnij się, że używasz najnowszej wersji strony

Przy zgłaszaniu błędu podaj:
- **Tytuł** - krótki, opisowy tytuł
- **Opis** - szczegółowy opis problemu
- **Kroki reprodukcji** - jak odtworzyć błąd?
- **Oczekiwane zachowanie** - jak powinno działać?
- **Aktualne zachowanie** - co się dzieje?
- **Screenshoty** - jeśli to możliwe
- **Środowisko**:
  - Przeglądarka (Chrome, Firefox, Safari, etc.)
  - Wersja przeglądarki
  - System operacyjny (Windows, macOS, Linux, iOS, Android)
  - Rozmiar ekranu/urządzenie

### Szablon zgłoszenia błędu

```markdown
## Opis błędu
[Jasny i zwięzły opis problemu]

## Kroki reprodukcji
1. Przejdź do '...'
2. Kliknij na '...'
3. Przewiń do '...'
4. Zobacz błąd

## Oczekiwane zachowanie
[Co powinno się stać]

## Aktualne zachowanie
[Co się dzieje]

## Screenshoty
[Jeśli dotyczy]

## Środowisko
- Przeglądarka: [np. Chrome 120]
- OS: [np. Windows 11]
- Urządzenie: [np. Desktop, iPhone 12]
```

## 💡 Proponowanie zmian

Masz pomysł na ulepszenie? Świetnie!

1. Otwórz [Issue](https://github.com/AI-robot-lab/ai-robot-lab.github.io/issues/new)
2. Użyj tagu `enhancement`
3. Opisz:
   - **Jaki problem rozwiązuje** ta zmiana?
   - **Jak to powinno działać?**
   - **Czy są alternatywy?**

## 🔄 Proces Pull Request

### 1. Fork repozytorium
```bash
# Kliknij "Fork" na GitHub, następnie:
git clone https://github.com/TWOJE_KONTO/ai-robot-lab.github.io.git
cd ai-robot-lab.github.io
```

### 2. Utwórz branch
```bash
# Dla nowej funkcji:
git checkout -b feature/nazwa-funkcji

# Dla poprawki błędu:
git checkout -b fix/nazwa-bledu

# Dla dokumentacji:
git checkout -b docs/opis-zmiany
```

### 3. Wprowadź zmiany
- Pisz czytelny kod
- Trzymaj się konwencji projektu
- Testuj swoje zmiany
- Dodaj komentarze gdzie potrzebne

### 4. Commit zmian
```bash
git add .
git commit -m "feat: dodano nową sekcję publikacji"

# lub
git commit -m "fix: poprawiono responsywność menu"

# lub
git commit -m "docs: zaktualizowano README"
```

#### Konwencja commitów
Stosujemy [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - nowa funkcja
- `fix:` - poprawka błędu
- `docs:` - zmiany w dokumentacji
- `style:` - formatowanie, brakujące średniki, etc.
- `refactor:` - refaktoryzacja kodu
- `test:` - dodawanie testów
- `chore:` - zmiany w build process, tools, etc.

### 5. Push do swojego forka
```bash
git push origin feature/nazwa-funkcji
```

### 6. Otwórz Pull Request
1. Przejdź do swojego forka na GitHub
2. Kliknij "New Pull Request"
3. Wybierz swój branch
4. Wypełnij szablon PR (patrz niżej)
5. Kliknij "Create Pull Request"

### Szablon Pull Request

```markdown
## Opis
[Krótki opis zmian]

## Typ zmiany
- [ ] Bug fix (zmiana niepowodująca awarii)
- [ ] New feature (zmiana dodająca funkcjonalność)
- [ ] Breaking change (fix lub feature powodujący niedziałanie istniejącej funkcjonalności)
- [ ] Dokumentacja

## Jak przetestowano?
[Opisz przeprowadzone testy]

## Checklist
- [ ] Kod jest zgodny z konwencjami projektu
- [ ] Przeprowadziłem self-review
- [ ] Zaktualizowałem dokumentację
- [ ] Przetestowałem na różnych przeglądarkach
- [ ] Przetestowałem responsywność
- [ ] Sprawdziłem dostępność (WCAG)

## Screenshots
[Jeśli dotyczy]
```

## 📝 Standardy kodu

### HTML
- Używaj semantycznego HTML5
- Dodawaj ARIA labels dla dostępności
- Alt texts dla wszystkich obrazów
- Poprawna hierarchia nagłówków (h1-h6)

```html
<!-- ✅ Dobrze -->
<section id="teams" aria-labelledby="teams-title">
    <h2 id="teams-title">Zespoły Badawcze</h2>
    <img src="team.jpg" alt="Zespół robotyki przy pracy">
</section>

<!-- ❌ Źle -->
<div id="teams">
    <div class="title">Zespoły Badawcze</div>
    <img src="team.jpg">
</div>
```

### CSS
- Używaj CSS Variables dla kolorów
- Mobile-first approach
- Organizuj style w logiczne sekcje
- Komentuj złożone style

```css
/* ✅ Dobrze */
:root {
    --prz-blue: #003366;
}

.button {
    background: var(--prz-blue);
    padding: 12px 24px;
    transition: all 0.3s ease;
}

/* ❌ Źle */
.button {
    background: #003366;
    padding: 12px 24px;
}
```

### JavaScript
- Używaj ES6+ syntax
- Dodawaj komentarze JSDoc
- Obsługuj błędy (try-catch)
- Używaj `'use strict'`

```javascript
// ✅ Dobrze
/**
 * Toggle mobile menu
 * @param {Event} e - Click event
 */
function toggleMenu(e) {
    try {
        const menu = document.querySelector('.nav-flex');
        menu.classList.toggle('active');
    } catch (error) {
        console.error('Error toggling menu:', error);
    }
}

// ❌ Źle
function toggleMenu() {
    document.querySelector('.nav-flex').classList.toggle('active');
}
```

### Dostępność (WCAG 2.1)
- Kontrast co najmniej 4.5:1 dla tekstu
- Wszystkie elementy interaktywne dostępne z klawiatury
- ARIA labels dla elementów bez tekstu
- Focus indicators
- Skip links

### Responsywność
- Mobile-first design
- Testuj na:
  - Mobile (≤767px)
  - Tablet (768px-1023px)
  - Desktop (≥1024px)

## 📁 Struktura projektu

```
├── index.html          # Strona główna
├── styles.css          # Style CSS
├── script.js           # JavaScript
├── README.md           # Dokumentacja
├── CONTRIBUTING.md     # Ten plik
├── CHANGELOG.md        # Historia zmian
├── .gitignore          # Ignorowane pliki
├── .nojekyll           # GitHub Pages config
├── robots.txt          # SEO
├── sitemap.xml         # SEO
└── assets/             # Obrazy, fonty (przyszłe)
    ├── images/
    ├── fonts/
    └── icons/
```

## 🧪 Testowanie

Przed wysłaniem PR, przetestuj:

### Przeglądarki
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

### Urządzenia
- [ ] Desktop (1920x1080)
- [ ] Laptop (1366x768)
- [ ] Tablet (768x1024)
- [ ] Mobile (375x667)

### Dostępność
- [ ] Keyboard navigation
- [ ] Screen reader (NVDA/JAWS/VoiceOver)
- [ ] Kontrast kolorów
- [ ] Focus indicators

### Narzędzia
- [W3C HTML Validator](https://validator.w3.org/)
- [W3C CSS Validator](https://jigsaw.w3.org/css-validator/)
- [WAVE Accessibility Tool](https://wave.webaim.org/)
- [Lighthouse](https://developers.google.com/web/tools/lighthouse)

## 💬 Komunikacja

- **GitHub Issues** - dla bugów i propozycji
- **Pull Requests** - dla zmian w kodzie
- **Email** - dla prywatnych spraw: [kontakt@prz.edu.pl]

## 📜 Kod postępowania

- Bądź uprzejmy i szanuj innych
- Przyjmuj konstruktywną krytykę
- Koncentruj się na tym, co najlepsze dla projektu
- Pokaż empatię wobec innych członków społeczności

## ❓ Pytania?

Jeśli masz pytania dotyczące współpracy:
1. Sprawdź [FAQ w README.md](README.md)
2. Przeszukaj istniejące Issues
3. Otwórz nowy Issue z tagiem `question`

---

**Dziękujemy za Twój wkład!** 🙏

Każdy pull request i issue są ważne dla rozwoju projektu.

---
*Ostatnia aktualizacja: 2025-02-10*
