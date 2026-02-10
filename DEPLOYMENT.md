# Przewodnik Wdrożenia na GitHub Pages

Szczegółowe instrukcje jak wdrożyć stronę Laboratorium Robotów Humanoidalnych na GitHub Pages.

## 📋 Wymagania wstępne

- Konto GitHub
- Git zainstalowany lokalnie
- Edytor tekstu (VS Code, Sublime Text, etc.)
- Podstawowa znajomość Git

## 🚀 Krok po kroku

### 1. Przygotowanie repozytorium

#### Opcja A: Nowe repozytorium

```bash
# Utwórz nowe repozytorium na GitHub o nazwie:
# ai-robot-lab.github.io
# (format: username.github.io)

# Sklonuj repozytorium lokalnie
git clone https://github.com/AI-robot-lab/ai-robot-lab.github.io.git
cd ai-robot-lab.github.io

# Skopiuj wszystkie pliki projektu do tego folderu
# (index.html, styles.css, script.js, README.md, etc.)
```

#### Opcja B: Istniejące repozytorium

```bash
# Jeśli masz już projekt lokalnie
cd twoj-projekt

# Dodaj remote
git remote add origin https://github.com/AI-robot-lab/ai-robot-lab.github.io.git
```

### 2. Dodanie plików

```bash
# Dodaj wszystkie pliki
git add .

# Commit
git commit -m "Initial commit: Humanoid Robotics Lab website v2.3.0"

# Push do GitHub
git branch -M main
git push -u origin main
```

### 3. Aktywacja GitHub Pages

1. Przejdź do repozytorium na GitHub
2. Kliknij **Settings** (⚙️)
3. W menu bocznym kliknij **Pages**
4. W sekcji "Source":
   - **Branch**: wybierz `main`
   - **Folder**: wybierz `/ (root)`
5. Kliknij **Save**

### 4. Weryfikacja

Po kilku minutach strona będzie dostępna pod adresem:
```
https://ai-robot-lab.github.io/
```

## 🔧 Konfiguracja

### Custom Domain (Opcjonalnie)

Jeśli chcesz użyć własnej domeny (np. robotlab.prz.edu.pl):

1. **W ustawieniach DNS domeny** dodaj rekord:
   ```
   Type: CNAME
   Name: www (lub subdomena)
   Value: ai-robot-lab.github.io
   ```

2. **W GitHub Pages Settings**:
   - W polu "Custom domain" wpisz: `robotlab.prz.edu.pl`
   - Kliknij Save
   - Zaznacz "Enforce HTTPS" (po propagacji DNS)

3. **Utwórz plik CNAME** w głównym katalogu:
   ```bash
   echo "robotlab.prz.edu.pl" > CNAME
   git add CNAME
   git commit -m "Add custom domain"
   git push
   ```

### Wymuszenie HTTPS

1. W GitHub Pages Settings
2. Zaznacz checkbox "Enforce HTTPS"
3. Poczekaj na wystawienie certyfikatu (może trwać do 24h)

## 📝 Struktura plików dla GitHub Pages

```
ai-robot-lab.github.io/
├── index.html              # Wymagane - strona główna
├── styles.css
├── script.js
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── DEPLOYMENT.md          # Ten plik
├── .nojekyll              # Ważne! Wyłącza Jekyll
├── .gitignore
├── robots.txt
├── sitemap.xml
├── favicon.ico            # Do dodania
├── CNAME                  # Jeśli używasz custom domain
└── assets/                # Opcjonalne - dla obrazów, etc.
    ├── images/
    └── icons/
```

## 🔍 Rozwiązywanie problemów

### Problem: Strona nie działa po 10 minutach

**Rozwiązanie:**
1. Sprawdź czy GitHub Pages jest włączone w Settings
2. Sprawdź czy branch to `main` a folder `/ (root)`
3. Sprawdź Actions w GitHub - czy build się powiódł
4. Oczyść cache przeglądarki (Ctrl+F5)

### Problem: Strona wyświetla się bez CSS

**Rozwiązanie:**
1. Sprawdź ścieżki w index.html - powinny być relatywne:
   ```html
   <!-- ✅ Dobrze -->
   <link href="styles.css" rel="stylesheet">
   
   <!-- ❌ Źle -->
   <link href="/styles.css" rel="stylesheet">
   <link href="./styles.css" rel="stylesheet">
   ```

2. Upewnij się że plik `.nojekyll` istnieje w głównym katalogu

### Problem: 404 na podstronach

**Rozwiązanie:**
- GitHub Pages obsługuje tylko statyczne strony
- Wszystkie linki powinny prowadzić do #sekcji lub do innych plików .html
- Sprawdź czy używasz poprawnych anchor links (#hero, #teams, etc.)

### Problem: Obrazy nie ładują się

**Rozwiązanie:**
1. Sprawdź ścieżki obrazów - powinny być relatywne lub absolutne URL
2. Jeśli używasz zewnętrznych obrazów, upewnij się że są publicznie dostępne
3. Dodaj `loading="lazy"` dla optymalizacji

```html
<!-- Lokalne -->
<img src="assets/images/robot.jpg" alt="Robot" loading="lazy">

<!-- Zewnętrzne -->
<img src="https://example.com/image.jpg" alt="Robot" loading="lazy">
```

## 📊 Monitoring i Analityka

### Google Analytics (Opcjonalnie)

1. Załóż konto w [Google Analytics](https://analytics.google.com/)
2. Utwórz właściwość dla swojej strony
3. Skopiuj "Measurement ID" (format: G-XXXXXXXXXX)
4. Dodaj do `<head>` w index.html:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

### Google Search Console

1. Przejdź do [Google Search Console](https://search.google.com/search-console)
2. Dodaj właściwość (URL prefix)
3. Zweryfikuj własność (HTML tag lub DNS)
4. Prześlij sitemap: `https://ai-robot-lab.github.io/sitemap.xml`

## 🔄 Aktualizacje

### Aktualizacja treści

```bash
# 1. Wprowadź zmiany w plikach
# 2. Commit
git add .
git commit -m "feat: dodano sekcję publikacji"

# 3. Push
git push origin main

# Strona zaktualizuje się automatycznie w ciągu ~1-3 minut
```

### Rollback (cofnij zmiany)

```bash
# Cofnij do poprzedniego commita
git revert HEAD
git push origin main

# Lub przywróć konkretny commit
git log  # znajdź hash commita
git checkout <commit-hash> .
git commit -m "Revert to previous version"
git push origin main
```

## 🎯 Best Practices

### Performance
- ✅ Zminifikuj CSS i JS (opcjonalnie)
- ✅ Optymalizuj obrazy (WebP, kompresja)
- ✅ Użyj lazy loading dla obrazów
- ✅ Włącz HTTPS
- ✅ Dodaj preconnect dla zewnętrznych zasobów

### SEO
- ✅ Dodaj robots.txt
- ✅ Dodaj sitemap.xml
- ✅ Użyj semantic HTML
- ✅ Dodaj meta description
- ✅ Dodaj Open Graph tags
- ✅ Użyj structured data (Schema.org)

### Dostępność
- ✅ ARIA labels
- ✅ Alt texts dla obrazów
- ✅ Keyboard navigation
- ✅ Skip links
- ✅ Odpowiedni kontrast kolorów

### Bezpieczeństwo
- ✅ Włącz HTTPS
- ✅ Dodaj Content Security Policy (CSP)
- ✅ Użyj rel="noopener noreferrer" dla zewnętrznych linków

## 📱 Testowanie po wdrożeniu

### Checklist
- [ ] Strona ładuje się poprawnie
- [ ] CSS i JavaScript działają
- [ ] Wszystkie linki działają
- [ ] Obrazy się ładują
- [ ] Responsywność (mobile, tablet, desktop)
- [ ] Nawigacja działa
- [ ] Formularze działają (jeśli są)
- [ ] Meta tags są poprawne (View Source)
- [ ] Favicon jest widoczny
- [ ] HTTPS jest aktywne

### Narzędzia testowe
- [PageSpeed Insights](https://pagespeed.web.dev/)
- [GTmetrix](https://gtmetrix.com/)
- [W3C Validator](https://validator.w3.org/)
- [WAVE Accessibility](https://wave.webaim.org/)
- [Mobile-Friendly Test](https://search.google.com/test/mobile-friendly)

## 🆘 Wsparcie

### Dokumentacja
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Git Documentation](https://git-scm.com/doc)
- [HTML MDN](https://developer.mozilla.org/en-US/docs/Web/HTML)
- [CSS MDN](https://developer.mozilla.org/en-US/docs/Web/CSS)

### Community
- [GitHub Community](https://github.community/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/github-pages)

## ✅ Checklist przed produkcją

- [ ] Wszystkie linki działają
- [ ] Obrazy mają alt text
- [ ] Meta tagi są poprawne
- [ ] robots.txt i sitemap.xml są dodane
- [ ] .nojekyll jest w repo
- [ ] README.md jest zaktualizowany
- [ ] HTTPS jest włączone
- [ ] Custom domain jest skonfigurowana (jeśli używana)
- [ ] Google Analytics jest dodane (opcjonalnie)
- [ ] Strona jest przetestowana na różnych urządzeniach
- [ ] Dostępność jest sprawdzona (WCAG)
- [ ] SEO jest zoptymalizowane
- [ ] Performance jest OK (PageSpeed > 90)

---

**Powodzenia z wdrożeniem!** 🚀

*Ostatnia aktualizacja: 2025-02-10*
