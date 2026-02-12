# Laboratorium Robotów Humanoidalnych | Politechnika Rzeszowska

[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue)](https://ai-robot-lab.github.io/)
[![PRz](https://img.shields.io/badge/PRz-Politechnika_Rzeszowska-003366)](https://www.prz.edu.pl/)

Oficjalna strona internetowa Laboratorium Robotów Humanoidalnych działającego w ramach Katedry Informatyki i Automatyki Politechniki Rzeszowskiej im. Ignacego Łukasiewicza.

## 🤖 O naszym Laboratorium

Laboratorium skupia się na:
- **Framework PCA** (Perception-Cognition-Action) - metodyka autonomicznych systemów humanoidalnych
- **Robotyka humanoidalna** - badania z wykorzystaniem Unitree G1 U6 EDU
- **Sztuczna inteligencja** - modele VLM, LLM, uczenie przez wzmacnianie
- **Rehabilitacja wspomagana** - zastosowania w terapii neurologicznej i poznawczej

## 🚀 Technologie

### Frontend
- **HTML5** - semantyczny markup
- **CSS3** - responsywny design, CSS Grid, Flexbox
- **JavaScript ES6+** - interakcje, smooth scrolling
- **Font Awesome 6** - ikony
- **Google Fonts** - typografia (Roboto, Montserrat, Playfair Display)

### Ekosystem Robotyczny
- ROS2 Humble
- NVIDIA Isaac Lab
- PyTorch
- Moveit2
- MediaPipe
- DeepFace

## 📦 Struktura Projektu

```
├── index.html          # Główny plik HTML
├── styles.css          # Arkusz stylów
├── script.js           # Interakcje JavaScript
├── README.md           # Dokumentacja
├── .nojekyll           # Wyłączenie Jekyll (GitHub Pages)
├── robots.txt          # Instrukcje dla robotów
├── sitemap.xml         # Mapa strony (SEO)
└── favicon.ico         # Ikona strony
```

## 🌐 Hostowanie na GitHub Pages

### Krok 1: Utworzenie repozytorium
```bash
# Inicjalizacja repo
git init
git add .
git commit -m "Initial commit: Humanoid Robotics Lab website"

# Dodanie remote
git remote add origin https://github.com/AI-robot-lab/ai-robot-lab.github.io.git
git branch -M main
git push -u origin main
```

### Krok 2: Konfiguracja GitHub Pages
1. Przejdź do Settings → Pages
2. Source: Deploy from a branch
3. Branch: `main` / `root`
4. Save

### Krok 3: Dostęp do strony
Strona będzie dostępna pod adresem:
```
https://ai-robot-lab.github.io/
```

## 🛠️ Rozwój Lokalny

### Wymagania
- Przeglądarka internetowa (Chrome, Firefox, Safari, Edge)
- Edytor kodu (VS Code, Sublime Text)
- Opcjonalnie: Python (dla lokalnego serwera)

### Uruchomienie lokalnie
```bash
# Metoda 1: Python 3
python -m http.server 8000

# Metoda 2: Python 2
python -m SimpleHTTPServer 8000

# Metoda 3: VS Code Live Server extension
# Kliknij prawym → "Open with Live Server"
```

Otwórz przeglądarkę: `http://localhost:8000`

## ✅ Checklist przed wdrożeniem

- [x] Walidacja HTML5 (https://validator.w3.org/)
- [x] Walidacja CSS3 (https://jigsaw.w3.org/css-validator/)
- [x] Test responsywności (mobile, tablet, desktop)
- [x] Test dostępności WCAG 2.1 (Level AA)
- [x] Optymalizacja obrazów
- [x] Meta tagi SEO
- [x] Open Graph tags
- [x] Schema.org structured data
- [x] Lazy loading dla obrazów
- [x] Smooth scrolling
- [x] Cross-browser testing
- [ ] Favicon (do dodania)
- [ ] robots.txt (opcjonalnie)
- [ ] sitemap.xml (opcjonalnie)

## 📱 Responsywność

Strona jest w pełni responsywna i obsługuje:
- **Desktop**: ≥1200px
- **Laptop**: 1024px - 1199px
- **Tablet**: 768px - 1023px
- **Mobile**: ≤767px
- **Small mobile**: ≤480px

## ♿ Dostępność

Strona spełnia standardy WCAG 2.1 Level AA:
- Semantyczny HTML5
- ARIA labels
- Skip to content link
- Focus indicators
- Kontrast kolorów
- Keyboard navigation
- Screen reader friendly

## 🔍 SEO

Zaimplementowane praktyki SEO:
- Meta description
- Open Graph tags
- Twitter Card
- Structured data (Schema.org)
- Semantic HTML
- Alt texts dla obrazów
- Sitemap.xml (do dodania)
- robots.txt (do dodania)

## 📊 Analytics (Opcjonalnie)

Aby dodać Google Analytics, dodaj w `<head>`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## 🐛 Zgłaszanie błędów

Jeśli znajdziesz błąd:
1. Sprawdź [Issues](https://github.com/AI-robot-lab/ai-robot-lab.github.io/issues)
2. Utwórz nowy Issue z opisem problemu
3. Dodaj screenshoty jeśli możliwe

## 🤝 Współpraca

1. Fork repozytorium
2. Utwórz branch (`git checkout -b feature/AmazingFeature`)
3. Commit zmian (`git commit -m 'Add some AmazingFeature'`)
4. Push do brancha (`git push origin feature/AmazingFeature`)
5. Otwórz Pull Request

## 📄 Licencja

Copyright © 2026 Politechnika Rzeszowska im. Ignacego Łukasiewicza

## 📞 Kontakt

**Laboratorium Robotów Humanoidalnych**
- **Adres**: Al. Powstańców Warszawy 12, 35-959 Rzeszów
- **GitHub**: [@AI-robot-lab](https://github.com/AI-robot-lab)
- **Kierownik organizacyjny**: dr inż. Mateusz Pomianek

## 🔗 Linki

- [Politechnika Rzeszowska](https://www.prz.edu.pl/)
- [Katedra Informatyki i Automatyki](https://kia.prz.edu.pl/)
- [ROS2 Documentation](https://docs.ros.org/)
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

---

**System Version**: 26.2.12 
**Last Updated**: 2026-02-12 
**Built with**: ❤️ by Humanoid Robotics Lab Team
