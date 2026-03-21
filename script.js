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
        const closeMobileMenu = function() {
            navFlex.classList.remove('active');
            mobileMenuToggle.setAttribute('aria-expanded', 'false');
            mobileMenuToggle.classList.remove('active');
        };

        mobileMenuToggle.addEventListener('click', function() {
            const isExpanded = this.getAttribute('aria-expanded') === 'true';

            navFlex.classList.toggle('active', !isExpanded);
            this.setAttribute('aria-expanded', String(!isExpanded));
            this.classList.toggle('active', !isExpanded);
        });
        
        // Close menu when clicking on a link
        const navLinks = navFlex.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    closeMobileMenu();
                }
            });
        });

        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && navFlex.classList.contains('active')) {
                closeMobileMenu();
                mobileMenuToggle.focus();
            }
        });

        document.addEventListener('click', function(event) {
            if (window.innerWidth > 768 || !navFlex.classList.contains('active')) return;
            if (!event.target.closest('nav')) {
                closeMobileMenu();
            }
        });

        window.addEventListener('resize', function() {
            if (window.innerWidth > 768) {
                closeMobileMenu();
            }
        });
    }

    initDarkMode();

    // iOS viewport height fix (--vh variable for 100vh workaround)
    (function fixViewportHeight() {
        function setVH() {
            document.documentElement.style.setProperty('--vh', (window.innerHeight * 0.01) + 'px');
        }
        setVH();
        window.addEventListener('resize', setVH, { passive: true });
        window.addEventListener('orientationchange', function() {
            setTimeout(setVH, 150);
        });
    })();
});

// ====================================
// Dark Mode
// ====================================

const DEFAULT_THEME = 'light';
const THEME_TRANSITION_MS = 200;

function getStoredTheme() {
    try {
        return localStorage.getItem('theme') || DEFAULT_THEME;
    } catch (error) {
        console.warn('Theme preference unavailable:', error);
        return DEFAULT_THEME;
    }
}

function setStoredTheme(theme) {
    try {
        localStorage.setItem('theme', theme);
    } catch (error) {
        console.warn('Failed to persist theme preference:', error);
    }
}

// Apply saved theme immediately to prevent flash of wrong theme
(function() {
    const savedTheme = getStoredTheme();
    document.documentElement.setAttribute('data-theme', savedTheme);
})();

function initDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');

    const savedTheme = getStoredTheme();
    updateDarkModeIcon(savedTheme);

    if (darkModeToggle && !darkModeToggle.dataset.darkModeInit) {
        darkModeToggle.dataset.darkModeInit = '1';
        darkModeToggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.documentElement.style.transition = 'background-color 0.2s ease, color 0.2s ease';
            document.documentElement.setAttribute('data-theme', newTheme);
            setStoredTheme(newTheme);

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
// Nav Scroll Shadow
// ====================================
(function initNavScroll() {
    const nav = document.querySelector('nav');
    if (!nav) return;
    let ticking = false;
    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(function() {
                nav.classList.toggle('scrolled', window.scrollY > 10);
                ticking = false;
            });
            ticking = true;
        }
    }, { passive: true });
})();

// ====================================
// Scroll Progress Bar
// ====================================
(function initScrollProgress() {
    if (document.querySelector('.scroll-progress')) return;
    const bar = document.createElement('div');
    bar.className = 'scroll-progress';
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-label', 'Postęp czytania');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    document.body.prepend(bar);
    let ticking = false;
    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(function() {
                const scrollable = document.documentElement.scrollHeight - window.innerHeight;
                const progress = scrollable > 0 ? Math.round(window.scrollY / scrollable * 100) : 0;
                bar.style.width = progress + '%';
                bar.setAttribute('aria-valuenow', progress);
                ticking = false;
            });
            ticking = true;
        }
    }, { passive: true });
})();

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

const observer = 'IntersectionObserver' in window
    ? new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions)
    : null;

// Observe elements
document.addEventListener('DOMContentLoaded', function() {
    const animateElements = document.querySelectorAll('.team-card, .soft-item, .rehab-box');

    if (!observer) {
        animateElements.forEach(el => el.classList.add('fade-in'));
        return;
    }

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
        const inlineHandler = this.getAttribute('onerror');
        const currentSrc = this.currentSrc || this.src;

        if (inlineHandler && !this.dataset.fallbackAttempted) {
            this.dataset.fallbackAttempted = '1';
            console.warn('Primary image failed, attempting fallback:', currentSrc);
            return;
        }

        this.style.display = 'none';
        console.warn('Failed to load image:', currentSrc);
    });

    img.addEventListener('load', function() {
        this.style.display = '';
    });
});
