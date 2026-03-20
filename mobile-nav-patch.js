/**
 * Navigation & Typography JS Patch
 * Laboratorium Robotów Humanoidalnych — Patch v2.0
 *
 * Dodaj na końcu script.js lub jako osobny plik po script.js
 * <script src="mobile-nav-patch.js" defer></script>
 */

(function () {
    'use strict';

    // ====================================================
    // 1. HAMBURGER — płynna animacja + label "Menu/Zamknij"
    // ====================================================
    function initHamburgerMenu() {
        const toggle  = document.querySelector('.mobile-menu-toggle');
        const navFlex = document.querySelector('.nav-flex');
        if (!toggle || !navFlex) return;

        // Wstaw label i spacer do istniejącego przycisku
        // (tylko jeśli nie zostały już wstawione)
        if (!toggle.querySelector('.menu-label')) {
            const label   = document.createElement('span');
            label.className = 'menu-label';
            label.textContent = 'Menu';
            label.setAttribute('aria-hidden', 'true');

            const spacer = document.createElement('span');
            spacer.className = 'menu-spacer';

            // Wstaw po istniejących span (hamburger linie)
            toggle.appendChild(spacer);
            toggle.appendChild(label);
        }

        const label = toggle.querySelector('.menu-label');

        // Klonuj przycisk, żeby usunąć stare listenery
        const freshToggle = toggle.cloneNode(true);
        toggle.parentNode.replaceChild(freshToggle, toggle);

        function openMenu() {
            navFlex.classList.add('active');
            freshToggle.classList.add('active');
            freshToggle.setAttribute('aria-expanded', 'true');
            const lbl = freshToggle.querySelector('.menu-label');
            if (lbl) lbl.textContent = 'Zamknij';
            // Zablokuj scroll strony gdy menu otwarte (opcjonalne)
            // document.body.style.overflow = 'hidden';
        }

        function closeMenu() {
            navFlex.classList.remove('active');
            freshToggle.classList.remove('active');
            freshToggle.setAttribute('aria-expanded', 'false');
            const lbl = freshToggle.querySelector('.menu-label');
            if (lbl) lbl.textContent = 'Menu';
            // document.body.style.overflow = '';
        }

        freshToggle.addEventListener('click', function () {
            const isOpen = navFlex.classList.contains('active');
            isOpen ? closeMenu() : openMenu();
        });

        // Zamknij klikając link
        navFlex.querySelectorAll('a').forEach(function (link) {
            link.addEventListener('click', function () {
                if (window.innerWidth <= 768) {
                    closeMenu();
                }
            });
        });

        // Zamknij klawiszem Escape
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && navFlex.classList.contains('active')) {
                closeMenu();
                freshToggle.focus();
            }
        });

        // Zamknij klikając poza menu
        document.addEventListener('click', function (e) {
            const nav = document.querySelector('nav');
            if (nav && !nav.contains(e.target) && navFlex.classList.contains('active')) {
                closeMenu();
            }
        });

        // Na resize > 768px zawsze pokaż nav (bez klas active)
        window.addEventListener('resize', function () {
            if (window.innerWidth > 768) {
                navFlex.classList.remove('active');
                freshToggle.classList.remove('active');
                freshToggle.setAttribute('aria-expanded', 'false');
            }
        });
    }


    // ====================================================
    // 2. NAV SHADOW — dodaj cień po przewinięciu
    // ====================================================
    function initNavScroll() {
        const nav = document.querySelector('nav');
        if (!nav) return;

        let ticking = false;

        function updateNavShadow() {
            if (window.scrollY > 10) {
                nav.classList.add('scrolled');
            } else {
                nav.classList.remove('scrolled');
            }
            ticking = false;
        }

        window.addEventListener('scroll', function () {
            if (!ticking) {
                requestAnimationFrame(updateNavShadow);
                ticking = true;
            }
        }, { passive: true });
    }


    // ====================================================
    // 3. ACTIVE LINK — poprawna detekcja aktywnej strony
    //    (dla linków prowadzących do stron, nie #kotew)
    // ====================================================
    function initActivePageLink() {
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = document.querySelectorAll('nav a[href]');

        navLinks.forEach(function (link) {
            const href = link.getAttribute('href');
            // Tylko linki do stron (nie #anchory, nie zewnętrzne)
            if (!href || href.startsWith('#') || href.startsWith('http')) return;

            const linkPage = href.split('/').pop().split('#')[0];
            if (linkPage === currentPage || (currentPage === '' && linkPage === 'index.html')) {
                link.classList.add('active');
            }
        });
    }


    // ====================================================
    // 4. SCROLL PROGRESS — subtelny pasek postępu
    // ====================================================
    function initScrollProgress() {
        // Sprawdź czy istnieje (mógł być dodany przez inne skrypty)
        if (document.querySelector('.scroll-progress')) return;

        const bar = document.createElement('div');
        bar.className = 'scroll-progress';
        bar.setAttribute('role', 'progressbar');
        bar.setAttribute('aria-label', 'Postęp czytania');
        bar.setAttribute('aria-valuemin', '0');
        bar.setAttribute('aria-valuemax', '100');
        document.body.prepend(bar);

        let ticking = false;

        function updateProgress() {
            const scrollable = document.documentElement.scrollHeight - window.innerHeight;
            const progress   = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
            bar.style.width  = Math.round(progress) + '%';
            bar.setAttribute('aria-valuenow', Math.round(progress));
            ticking = false;
        }

        window.addEventListener('scroll', function () {
            if (!ticking) {
                requestAnimationFrame(updateProgress);
                ticking = true;
            }
        }, { passive: true });
    }


    // ====================================================
    // 5. FLUID FONT SIZE — iOS viewport bug fix
    // ====================================================
    function fixViewportHeight() {
        function setVH() {
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', vh + 'px');
        }
        setVH();
        window.addEventListener('resize', setVH, { passive: true });
        window.addEventListener('orientationchange', function () {
            setTimeout(setVH, 150);
        });
    }


    // ====================================================
    // 6. PREVENT IOS ZOOM na inputach
    // ====================================================
    function preventIOSInputZoom() {
        if (!/iPhone|iPad|iPod/.test(navigator.userAgent)) return;

        const style = document.createElement('style');
        style.textContent = [
            'input[type="text"], input[type="email"], input[type="number"],',
            'input[type="tel"], input[type="search"], select, textarea {',
            '  font-size: max(16px, 1rem) !important;',
            '}'
        ].join(' ');
        document.head.appendChild(style);
    }


    // ====================================================
    // 7. WIKI SEARCH — podświetl znaleziony fragment
    // ====================================================
    function enhanceWikiSearch() {
        const searchInput = document.getElementById('wikiSearch');
        if (!searchInput) return;

        // Już jest obsługiwane przez wiki.js — tylko dodajemy ARIA
        searchInput.setAttribute('role', 'searchbox');
        searchInput.setAttribute('aria-label', 'Szukaj artykułów w bazie wiedzy');
        searchInput.setAttribute('autocomplete', 'off');
        searchInput.setAttribute('autocorrect', 'off');
        searchInput.setAttribute('autocapitalize', 'off');
        searchInput.setAttribute('spellcheck', 'false');

        // Wyczyść szukanie klawiszem Escape
        searchInput.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                this.value = '';
                this.dispatchEvent(new Event('input'));
                this.blur();
            }
        });
    }


    // ====================================================
    // INIT
    // ====================================================
    function init() {
        initHamburgerMenu();
        initNavScroll();
        initActivePageLink();
        initScrollProgress();
        fixViewportHeight();
        preventIOSInputZoom();
        enhanceWikiSearch();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
