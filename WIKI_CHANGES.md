# WIKI - Znaczące Ulepszenia Graficzne i Funkcjonalne

## 🎨 Główne Zmiany Graficzne

### 1. **Hero Section**
- ✨ **Gradient Background**: Ciemnoniebieski gradient (#003366 → #004d99)
- 🌟 **Animowana Ikona**: Floating animation dla ikony książki
- 📊 **Statystyki**: 3 karty ze statystykami (35+ artykułów, 6 kategorii, 12 gotowych)
- 🎭 **Backdrop Effects**: Subtelne radial gradienty w tle
- 💫 **Hover Effects**: Karty reagują na hover z animacją

### 2. **Sidebar (Menu)**
- 📌 **Sticky Positioning**: Menu przyklejone podczas scrollowania
- 🔍 **Ulepszony Search**: Lepsze style z focus states
- 🎯 **Kategorie z Ikonami**: Każda kategoria ma dedykowaną ikonę
- ✨ **Animowane Linki**: 
  - Efekt podkreślenia z lewej strony
  - Smooth transitions
  - Active state z gradienten
- 📜 **Custom Scrollbar**: Stylowany scrollbar w kolorach PRz

### 3. **Quick Start Cards**
- 🎴 **Card Layout**: 6 kart w siatce 2x3
- 🎨 **Ikony w Kolorze**: Każda karta ma unikalną ikonę
- 💡 **Hover Animations**: 
  - Podniesienie karty
  - Rotacja ikony
  - Zmiana koloru tytułu
- 📝 **Opis**: Każda karta ma tytuł i krótki opis

### 4. **Info Box**
- 💙 **Niebieski Gradient**: Przyjemny gradient background
- ℹ️ **Ikona Info**: Duża ikona informacyjna
- 📌 **Border**: Wyraźny niebieski border
- 📱 **Responsive**: Zmienia layout na mobile

### 5. **Breadcrumbs (Okruszki)**
- 🏠 **Nawigacja**: Home → Kategoria → Artykuł
- 🎨 **Subtelny Background**: Szary background
- 🔗 **Aktywne Linki**: Linki z hover effects

## 🔧 Ulepszenia Techniczne

### wiki.js
```javascript
// ✅ Poprawne ładowanie plików z folderu wiki/
// ✅ Obsługa hash navigation (#article-id)
// ✅ Search z debounce (300ms)
// ✅ Active states dla linków
// ✅ Breadcrumbs update
// ✅ Internal links (#wiki-article-id)
// ✅ Back/forward browser navigation
```

### styles.css
```css
/* Nowe Style */
.wiki-hero-icon        /* Animowana ikona */
.wiki-hero-stats       /* Flex container dla statystyk */
.hero-stat             /* Pojedyncza statystyka */
.sidebar-sticky        /* Sticky positioning */
.quick-links-grid      /* Grid 2x3 dla cards */
.quick-link-card       /* Pojedyncza karta */
.quick-link-icon       /* Ikona w karcie */
.quick-link-content    /* Treść karty */
.wiki-info-box         /* Info box */
.breadcrumbs           /* Nawigacja breadcrumbs */
```

## 📱 Responsive Design

### Desktop (> 1024px)
- Grid 320px sidebar + reszta content
- Sticky sidebar
- Cards 2 kolumny

### Tablet (768px - 1024px)
- Stack layout (sidebar na górze)
- Cards 1 kolumna
- Mniejsze fonty

### Mobile (< 768px)
- Full width wszystko
- Stack layout
- Statystyki pionowo
- Cards 1 kolumna
- Mniejsze paddingi

## 🎯 Kluczowe Cechy

### 1. **Konsystentne Kolory**
- PRz Blue: #003366
- PRz Gold: #c5a059
- Białe karty na szarym tle
- Gradienty dla depth

### 2. **Smooth Animations**
- Wszystkie transitions 0.3s ease
- Hover effects na wszystkich elementach
- Float animation dla ikony
- Scale i rotate dla ikon w kartach

### 3. **Professional Typography**
- Playfair Display dla nagłówków
- Roboto dla tekstu
- Montserrat dla kategorii
- Różne wagi dla hierarchii

### 4. **Accessibility**
- ARIA labels
- Keyboard navigation
- Skip links
- Semantic HTML

## 📂 Struktura Plików

```
├── wiki.html           # Główna strona WIKI (ulepszona)
├── wiki.js             # JavaScript (przepisany)
├── styles.css          # CSS (znacznie rozszerzony)
└── wiki/               # Folder z artykułami .md
    ├── ros2.md
    ├── isaac-lab.md
    ├── computer-vision.md
    ├── llm.md
    ├── pytorch.md
    ├── opencv.md
    └── ... (35+ artykułów)
```

## 🚀 Jak Używać

1. **Otwórz** `wiki.html` w przeglądarce
2. **Kliknij** artykuł z menu po lewej
3. **Artykuł** załaduje się z animacją
4. **Breadcrumbs** pokażą ścieżkę
5. **Wyszukaj** używając search bara

## ✨ Highlights

### Hero z Animacjami
```html
<div class="wiki-hero-icon">
    <i class="fa-solid fa-book-open"></i>
</div>
```
- Floating animation
- 4rem font size
- Golden color

### Quick Links jako Cards
```html
<a class="quick-link-card">
    <div class="quick-link-icon">🤖</div>
    <div class="quick-link-content">
        <strong>Tytuł</strong>
        <span>Opis</span>
    </div>
</a>
```
- Gradient background dla ikony
- Hover: scale + rotate ikony
- Border highlight na hover

### Search z Auto-filter
```javascript
// Debounce 300ms
searchInput.addEventListener('input', ...)
```
- Filtruje kategorie
- Ukrywa puste kategorie
- Highlight aktywnych linków

## 🎨 Paleta Kolorów

| Element | Kolor | Użycie |
|---------|-------|--------|
| **Primary** | #003366 | Hero, linki, borders |
| **Gold** | #c5a059 | Akcenty, ikony, hover |
| **White** | #ffffff | Karty, tło content |
| **Gray** | #f8f9fa | Tło sekcji |
| **Dark** | #2c3e50 | Tekst główny |
| **Light** | #6c757d | Tekst drugorzędny |

## 📊 Statystyki

- **35+** artykułów (zdefiniowanych)
- **12** kompletnych artykułów
- **6** kategorii
- **~500** linii nowego CSS
- **~300** linii nowego JS

## 🔮 Gotowe na Produkcję

✅ Wszystkie pliki w `/mnt/user-data/outputs/`  
✅ Gotowe do wdrożenia na GitHub Pages  
✅ Responsive na wszystkich urządzeniach  
✅ Accessibility compliant  
✅ SEO optimized  

---

*WIKI System v2.0 - Laboratorium Robotów Humanoidalnych PRz*
