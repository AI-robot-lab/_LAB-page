/**
 * Main JavaScript for Humanoid Robotics Lab Website
 * Politechnika Rzeszowska
 */

'use strict';

// ====================================
// Mobile Menu Toggle
// ====================================
document.addEventListener('DOMContentLoaded', function() {
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const navFlex = document.querySelector('.nav-flex');
    
    if (mobileMenuToggle && navFlex) {
        mobileMenuToggle.addEventListener('click', function() {
            const isExpanded = this.getAttribute('aria-expanded') === 'true';
            
            // Toggle menu
            navFlex.classList.toggle('active');
            
            // Update ARIA attribute
            this.setAttribute('aria-expanded', !isExpanded);
            
            // Animate hamburger icon
            this.classList.toggle('active');
        });
        
        // Close menu when clicking on a link
        const navLinks = navFlex.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    navFlex.classList.remove('active');
                    mobileMenuToggle.setAttribute('aria-expanded', 'false');
                    mobileMenuToggle.classList.remove('active');
                }
            });
        });
    }

    initDarkMode();
});

// ====================================
// Dark Mode
// ====================================

const DEFAULT_THEME = 'light';
const THEME_TRANSITION_MS = 200;

// Apply saved theme immediately to prevent flash of wrong theme
(function() {
    const savedTheme = localStorage.getItem('theme') || DEFAULT_THEME;
    document.documentElement.setAttribute('data-theme', savedTheme);
})();

function initDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');

    const savedTheme = localStorage.getItem('theme') || DEFAULT_THEME;
    updateDarkModeIcon(savedTheme);

    if (darkModeToggle && !darkModeToggle.dataset.darkModeInit) {
        darkModeToggle.dataset.darkModeInit = '1';
        darkModeToggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.documentElement.style.transition = 'background-color 0.2s ease, color 0.2s ease';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            updateDarkModeIcon(newTheme);

            setTimeout(function() {
                document.documentElement.style.transition = '';
            }, THEME_TRANSITION_MS);
        });
    }
}

function updateDarkModeIcon(theme) {
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (!darkModeToggle) return;

    const icon = darkModeToggle.querySelector('i');
    if (icon) {
        icon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    }
}

// ====================================
// Smooth Scrolling Enhancement
// ====================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        
        // Skip empty anchors and wiki article links
        if (href === '#') {
            if (!this.dataset.article) {
                e.preventDefault();
            }
            return;
        }
        
        const targetElement = document.querySelector(href);
        
        if (targetElement) {
            e.preventDefault();
            
            const headerOffset = 80; // Height of sticky nav
            const elementPosition = targetElement.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
            
            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// ====================================
// Active Navigation Link
// ====================================
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    
    let currentSection = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (pageYOffset >= sectionTop - 100) {
            currentSection = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + currentSection) {
            link.classList.add('active');
        }
    });
});

// ====================================
// Image Lazy Loading Fallback
// ====================================
if ('loading' in HTMLImageElement.prototype) {
    // Browser supports lazy loading
    const images = document.querySelectorAll('img[loading="lazy"]');
    images.forEach(img => {
        img.src = img.src;
    });
} else {
    // Fallback for browsers that don't support lazy loading
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/lazysizes/5.3.2/lazysizes.min.js';
    document.body.appendChild(script);
}

// ====================================
// Add animation on scroll
// ====================================
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
        }
    });
}, observerOptions);

// Observe elements
document.addEventListener('DOMContentLoaded', function() {
    const animateElements = document.querySelectorAll('.team-card, .soft-item, .rehab-box');
    animateElements.forEach(el => {
        observer.observe(el);
    });
});

// Add CSS for fade-in animation
const style = document.createElement('style');
style.textContent = `
    .team-card,
    .soft-item,
    .rehab-box {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.6s ease, transform 0.6s ease;
    }
    
    .fade-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);

// ====================================
// External Links - Open in New Tab
// ====================================
document.querySelectorAll('a[href^="http"]').forEach(link => {
    if (!link.getAttribute('target')) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
    }
});

// ====================================
// Performance: Preload Important Resources
// ====================================
function preloadResource(href, as) {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    document.head.appendChild(link);
}

// ====================================
// Console Info
// ====================================
console.log('%cLaboratorium Robotów Humanoidalnych', 'color: #003366; font-size: 20px; font-weight: bold;');
console.log('%cPolitechnika Rzeszowska', 'color: #c5a059; font-size: 14px;');
console.log('System Version: 2.3.0-stable');
console.log('GitHub: https://github.com/AI-robot-lab');

// ====================================
// Keyboard Navigation Enhancement
// ====================================
document.addEventListener('keydown', function(e) {
    // Press 'Esc' to close mobile menu
    if (e.key === 'Escape') {
        const navFlex = document.querySelector('.nav-flex');
        const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
        
        if (navFlex && navFlex.classList.contains('active')) {
            navFlex.classList.remove('active');
            if (mobileMenuToggle) {
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                mobileMenuToggle.classList.remove('active');
            }
        }
    }
});

// ====================================
// Print Optimization
// ====================================
window.addEventListener('beforeprint', function() {
    // Expand all collapsed sections before printing
    document.querySelectorAll('.nav-flex').forEach(nav => {
        nav.style.display = 'none';
    });
});

window.addEventListener('afterprint', function() {
    // Restore original state after printing
    document.querySelectorAll('.nav-flex').forEach(nav => {
        nav.style.display = '';
    });
});

// ====================================
// Error Handling for Images
// ====================================
document.querySelectorAll('img').forEach(img => {
    img.addEventListener('error', function() {
        this.style.display = 'none';
        console.warn('Failed to load image:', this.src);
    });
});

// ====================================
// Service Worker Registration
// ====================================
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('./sw.js').then(function(registration) {
            console.log('ServiceWorker registered:', registration);
        }).catch(function(error) {
            console.log('ServiceWorker registration failed:', error);
        });
    });
}

// ====================================
// PWA: Bottom Navigation Bar
// ====================================
(function() {
    function createBottomNav() {
        if (document.querySelector('.pwa-bottom-nav')) return;

        var nav = document.createElement('nav');
        nav.className = 'pwa-bottom-nav';
        nav.setAttribute('aria-label', 'Nawigacja mobilna');

        var pathname = window.location.pathname;
        var currentPage = pathname.split('/').pop() || 'index.html';
        if (currentPage === '' || currentPage === '/' || !currentPage.includes('.')) {
            currentPage = 'index.html';
        }

        var items = [
            { href: 'index.html', icon: 'fa-solid fa-house', label: 'Strona główna', id: 'index.html' },
            { href: 'wiki.html', icon: 'fa-solid fa-book', label: 'Wiki', id: 'wiki.html' },
            { href: 'pdf.html', icon: 'fa-solid fa-file-pdf', label: 'PDF', id: 'pdf.html' },
            { href: 'contact.html', icon: 'fa-solid fa-envelope', label: 'Kontakt', id: 'contact.html' }
        ];

        items.forEach(function(item) {
            var a = document.createElement('a');
            a.href = item.href;
            a.className = 'pwa-bottom-nav-item';
            if (currentPage === item.id) {
                a.classList.add('active');
            }
            a.setAttribute('aria-label', item.label);
            a.innerHTML = '<i class="' + item.icon + '" aria-hidden="true"></i><span>' + item.label + '</span>';
            nav.appendChild(a);
        });

        document.body.appendChild(nav);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createBottomNav);
    } else {
        createBottomNav();
    }
})();

// ====================================
// PWA: Install Prompt
// ====================================
(function() {
    var deferredPrompt = null;

    window.addEventListener('beforeinstallprompt', function(e) {
        e.preventDefault();
        deferredPrompt = e;
        showInstallBanner();
    });

    function showInstallBanner() {
        if (document.querySelector('.pwa-install-banner')) return;
        if (window.matchMedia('(display-mode: standalone)').matches) return;
        if (window.navigator.standalone === true) return;

        var banner = document.createElement('div');
        banner.className = 'pwa-install-banner';
        banner.innerHTML =
            '<div class="pwa-install-content">' +
                '<img src="assets/icons/icon-192x192.png" alt="RobotLab" class="pwa-install-icon">' +
                '<div class="pwa-install-text">' +
                    '<strong>RobotLab PRz</strong>' +
                    '<span>Zainstaluj aplikację na urządzeniu</span>' +
                '</div>' +
            '</div>' +
            '<div class="pwa-install-actions">' +
                '<button class="pwa-install-btn" aria-label="Zainstaluj aplikację">Instaluj</button>' +
                '<button class="pwa-install-close" aria-label="Zamknij">&times;</button>' +
            '</div>';

        document.body.appendChild(banner);

        banner.querySelector('.pwa-install-btn').addEventListener('click', function() {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then(function() {
                    deferredPrompt = null;
                }).finally(function() {
                    banner.remove();
                });
            }
        });

        banner.querySelector('.pwa-install-close').addEventListener('click', function() {
            banner.remove();
        });
    }
})();

// ====================================
// PWA: Standalone Mode Detection
// ====================================
(function() {
    function checkStandaloneMode() {
        var isStandalone = window.matchMedia('(display-mode: standalone)').matches ||
                           window.navigator.standalone === true;
        if (isStandalone) {
            document.documentElement.classList.add('pwa-standalone');
        }
    }
    checkStandaloneMode();
    window.matchMedia('(display-mode: standalone)').addEventListener('change', checkStandaloneMode);
})();
