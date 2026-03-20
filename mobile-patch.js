/**
 * Mobile JS Patch
 * Laboratorium Robotów Humanoidalnych
 * Dodaj ten skrypt na KOŃCU script.js (lub jako osobny plik)
 */

(function () {
    'use strict';

    // ====================================
    // MOBILNE MENU — Poprawka animacji
    // Zastępuje display:none/flex → max-height animation
    // ====================================
    function patchMobileMenu() {
        const toggle = document.querySelector('.mobile-menu-toggle');
        const navFlex = document.querySelector('.nav-flex');
        if (!toggle || !navFlex) return;

        // Usuń stare listenery przez klonowanie
        const newToggle = toggle.cloneNode(true);
        toggle.parentNode.replaceChild(newToggle, toggle);

        newToggle.addEventListener('click', function () {
            const isOpen = navFlex.classList.contains('active');

            if (isOpen) {
                navFlex.classList.remove('active');
                this.setAttribute('aria-expanded', 'false');
                this.classList.remove('active');
            } else {
                navFlex.classList.add('active');
                this.setAttribute('aria-expanded', 'true');
                this.classList.add('active');
            }
        });

        // Zamknij klikając link na mobile
        navFlex.querySelectorAll('a').forEach(function (link) {
            link.addEventListener('click', function () {
                if (window.innerWidth <= 768) {
                    navFlex.classList.remove('active');
                    newToggle.setAttribute('aria-expanded', 'false');
                    newToggle.classList.remove('active');
                }
            });
        });
    }

    // ====================================
    // POZYCJA DARK MODE TOGGLE
    // Przesuń powyżej bottom nav na mobile
    // ====================================
    function positionDarkModeToggle() {
        const toggle = document.getElementById('darkModeToggle');
        if (!toggle) return;

        function updatePosition() {
            const isMobile = window.innerWidth <= 768;
            if (isMobile) {
                // Oblicz offset: bottom nav (64px) + safe area + margin
                const safeArea = parseInt(
                    getComputedStyle(document.documentElement)
                        .getPropertyValue('--sat') || '0'
                ) || 0;
                toggle.style.bottom = (64 + safeArea + 16) + 'px';
                toggle.style.top = 'auto';
                toggle.style.right = '16px';
            } else {
                toggle.style.bottom = '';
                toggle.style.top = '80px';
                toggle.style.right = '20px';
            }
        }

        updatePosition();
        window.addEventListener('resize', updatePosition);
    }

    // ====================================
    // IOS VIEWPORT HEIGHT FIX
    // Naprawia 100vh na iOS Safari
    // ====================================
    function fixIOSViewport() {
        function setVH() {
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', vh + 'px');
        }
        setVH();
        window.addEventListener('resize', setVH);
        window.addEventListener('orientationchange', function () {
            setTimeout(setVH, 100);
        });
    }

    // ====================================
    // SMOOTH SCROLL — Fix dla starych iOS
    // ====================================
    function fixSmoothScroll() {
        if ('scrollBehavior' in document.documentElement.style) return;

        // Polyfill dla starszych przeglądarek
        document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
            anchor.addEventListener('click', function (e) {
                const target = document.querySelector(this.getAttribute('href'));
                if (!target) return;
                e.preventDefault();
                const headerOffset = 70;
                const top = target.getBoundingClientRect().top + window.pageYOffset - headerOffset;
                window.scrollTo({ top: top, behavior: 'smooth' });
            });
        });
    }

    // ====================================
    // ZAPOBIEGAJ ZOOMOWI NA IOS przy focus
    // Ustawia font-size min 16px dla inputów
    // ====================================
    function preventIOSZoom() {
        if (!/iPad|iPhone|iPod/.test(navigator.userAgent)) return;

        const inputs = document.querySelectorAll('input, select, textarea');
        inputs.forEach(function (input) {
            const computedSize = parseFloat(getComputedStyle(input).fontSize);
            if (computedSize < 16) {
                input.style.fontSize = '16px';
            }
        });
    }

    // ====================================
    // NOTIFICATION PANEL — Poprawka
    // Zamknij na swipe down (mobile)
    // ====================================
    function addSwipeToCloseNotifPanel() {
        document.addEventListener('click', function (e) {
            const panel = document.querySelector('.notif-panel');
            if (panel && e.target === panel) {
                panel.remove();
            }
        });

        // Obserwuj dynamicznie dodawane panele
        const observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (mutation) {
                mutation.addedNodes.forEach(function (node) {
                    if (node.classList && node.classList.contains('notif-panel')) {
                        addTouchDismiss(node);
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true });
    }

    function addTouchDismiss(panel) {
        let startY = 0;
        panel.addEventListener('touchstart', function (e) {
            startY = e.touches[0].clientY;
        }, { passive: true });

        panel.addEventListener('touchmove', function (e) {
            const deltaY = e.touches[0].clientY - startY;
            if (deltaY > 80) {
                panel.remove();
            }
        }, { passive: true });
    }

    // ====================================
    // LAZY LOAD POPRAWKA
    // Na starszych urządzeniach
    // ====================================
    function initLazyLoad() {
        if ('IntersectionObserver' in window) {
            const lazyImages = document.querySelectorAll('img[loading="lazy"]');
            const imageObserver = new IntersectionObserver(function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                        }
                        imageObserver.unobserve(img);
                    }
                });
            }, { rootMargin: '50px' });

            lazyImages.forEach(function (img) {
                imageObserver.observe(img);
            });
        }
    }

    // ====================================
    // INIT
    // ====================================
    function init() {
        patchMobileMenu();
        positionDarkModeToggle();
        fixIOSViewport();
        fixSmoothScroll();
        preventIOSZoom();
        addSwipeToCloseNotifPanel();
        initLazyLoad();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
