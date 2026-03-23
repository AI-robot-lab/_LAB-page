/**
 * Main JavaScript for Humanoid Robotics Lab Website
 * Politechnika Rzeszowska
 */

'use strict';

const MOBILE_BREAKPOINT = 768;
const DEFAULT_THEME = 'light';
const THEME_TRANSITION_MS = 200;
const DARK_MODE_LABELS = {
    dark: 'Włącz jasny motyw',
    light: 'Włącz ciemny motyw'
};

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

function getHashTarget(hash) {
    if (!hash || hash === '#') return null;

    const normalizedHash = hash.startsWith('#') ? hash.slice(1) : hash;
    const decodedHash = decodeURIComponent(normalizedHash);

    if (!decodedHash) return null;

    return document.getElementById(decodedHash)
        || document.querySelector(`[name="${CSS.escape(decodedHash)}"]`);
}

function syncDarkModeControl(theme) {
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (!darkModeToggle) return;

    const icon = darkModeToggle.querySelector('i');
    if (icon) {
        icon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    }

    darkModeToggle.setAttribute('aria-pressed', String(theme === 'dark'));
    darkModeToggle.setAttribute('aria-label', DARK_MODE_LABELS[theme] || DARK_MODE_LABELS.light);
    darkModeToggle.title = DARK_MODE_LABELS[theme] || DARK_MODE_LABELS.light;
}

// Apply saved theme immediately to prevent flash of wrong theme
(function() {
    const savedTheme = getStoredTheme();
    document.documentElement.setAttribute('data-theme', savedTheme);
})();

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
                if (window.innerWidth <= MOBILE_BREAKPOINT) {
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
            if (window.innerWidth > MOBILE_BREAKPOINT || !navFlex.classList.contains('active')) return;
            if (!event.target.closest('nav')) {
                closeMobileMenu();
            }
        });

        window.addEventListener('resize', function() {
            if (window.innerWidth > MOBILE_BREAKPOINT) {
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
function initDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const savedTheme = getStoredTheme();

    syncDarkModeControl(savedTheme);

    if (darkModeToggle && !darkModeToggle.dataset.darkModeInit) {
        darkModeToggle.dataset.darkModeInit = '1';
        darkModeToggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.documentElement.style.transition = 'background-color 0.2s ease, color 0.2s ease';
            document.documentElement.setAttribute('data-theme', newTheme);
            setStoredTheme(newTheme);
            syncDarkModeControl(newTheme);

            setTimeout(function() {
                document.documentElement.style.transition = '';
            }, THEME_TRANSITION_MS);
        });
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

        const targetElement = getHashTarget(href);

        if (targetElement) {
            e.preventDefault();

            const headerOffset = 80; // Height of sticky nav
            const elementPosition = targetElement.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.scrollY - headerOffset;

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
(function initActiveNavLink() {
    const updateActiveNavigation = function() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('nav a[href^="#"]');
        let currentSection = '';

        sections.forEach(section => {
            if (window.scrollY >= section.offsetTop - 100) {
                currentSection = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${currentSection}`);
        });
    };

    window.addEventListener('scroll', updateActiveNavigation, { passive: true });
    window.addEventListener('load', updateActiveNavigation);
    document.addEventListener('DOMContentLoaded', updateActiveNavigation);
})();

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

    const updateScrollProgress = function() {
        const scrollable = document.documentElement.scrollHeight - window.innerHeight;
        const progress = scrollable > 0 ? Math.round(window.scrollY / scrollable * 100) : 0;
        bar.style.width = progress + '%';
        bar.setAttribute('aria-valuenow', String(progress));
    };

    let ticking = false;
    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(function() {
                updateScrollProgress();
                ticking = false;
            });
            ticking = true;
        }
    }, { passive: true });

    window.addEventListener('resize', updateScrollProgress, { passive: true });
    document.addEventListener('DOMContentLoaded', updateScrollProgress);
    window.addEventListener('load', updateScrollProgress);
})();

// ====================================
// Image Lazy Loading Fallback
// ====================================
if ('loading' in HTMLImageElement.prototype) {
    const images = document.querySelectorAll('img[loading="lazy"]');
    images.forEach(img => {
        img.src = img.src;
    });
} else {
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

document.addEventListener('DOMContentLoaded', function() {
    const animateElements = document.querySelectorAll('.team-card, .soft-item, .rehab-box, .experience-card, .spotlight-panel, .roadmap-step, .metric-pill');

    if (!observer) {
        animateElements.forEach(el => el.classList.add('fade-in'));
        return;
    }

    animateElements.forEach(el => {
        observer.observe(el);
    });
});

const style = document.createElement('style');
style.textContent = `
    .team-card,
    .soft-item,
    .rehab-box,
    .experience-card,
    .spotlight-panel,
    .roadmap-step,
    .metric-pill {
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
// Keyboard Navigation Enhancement
// ====================================
document.addEventListener('keydown', function(e) {
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
    document.querySelectorAll('.nav-flex').forEach(nav => {
        nav.style.display = 'none';
    });
});

window.addEventListener('afterprint', function() {
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


// ====================================
// Interactive Showcase
// ====================================
document.addEventListener('DOMContentLoaded', function() {
    const scenarios = {
        rehab: [
            ['sense', 'Analiza postawy pacjenta i zakresu ruchu kończyn.'],
            ['infer', 'Silnik kognicji dopasowuje poziom assist-as-needed.'],
            ['act', 'Robot prowadzi sesję z informacją zwrotną w czasie rzeczywistym.']
        ],
        inspection: [
            ['scan', 'Moduł obserwacji mapuje stanowisko i wykrywa odchylenia.'],
            ['reason', 'VLM porównuje scenę z checklistą jakości.'],
            ['respond', 'Robot raportuje ryzyka i planuje sekwencję kontroli.']
        ],
        hri: [
            ['listen', 'System analizuje mowę, gesty i sygnały afektywne użytkownika.'],
            ['align', 'LLM buduje kontekst dialogu i intencję zadania.'],
            ['engage', 'Awatar ruchowy dopasowuje gesty oraz odpowiedzi robota.']
        ]
    };

    const commandFeed = document.getElementById('commandFeed');
    const chips = document.querySelectorAll('.scenario-chip');

    function renderScenario(name) {
        if (!commandFeed || !scenarios[name]) return;
        commandFeed.innerHTML = scenarios[name].map(([label, text]) => `
            <div class="feed-line">
                <span>${label}</span>
                <p>${text}</p>
            </div>
        `).join('');

        chips.forEach(chip => {
            const active = chip.dataset.scenario === name;
            chip.classList.toggle('active', active);
            chip.setAttribute('aria-selected', String(active));
        });
    }

    chips.forEach(chip => {
        chip.addEventListener('click', function() {
            renderScenario(this.dataset.scenario);
        });
    });

    const spotlightContent = {
        perception: {
            title: 'Perception Stack',
            text: 'Warstwa percepcji agreguje dane przestrzenne, wizualne i behawioralne w jedno źródło prawdy dla robota.',
            items: [
                'LiDAR 3D, kamery RGB-D, IMU i mikrofony kierunkowe',
                'Rozumienie sceny i obiektów z uwzględnieniem proxemics',
                'Wykrywanie sygnałów użytkownika ważnych dla bezpieczeństwa i HRI'
            ]
        },
        cognition: {
            title: 'Cognition Engine',
            text: 'Kognicja łączy modele świata, semantykę i planowanie zadań, aby robot potrafił podjąć właściwą decyzję.',
            items: [
                'Integracja VLM/LLM z planowaniem i politykami sterowania',
                'Predykcja intencji oraz adaptacja zachowania do użytkownika',
                'Pipeline eksperymentalny gotowy do iteracji sim-to-real'
            ]
        },
        action: {
            title: 'Action & Embodiment',
            text: 'Warstwa wykonawcza realizuje ruch, manipulację i teleoperację z naciskiem na płynność oraz precyzję.',
            items: [
                'Manipulacja dłońmi Dex3-1 i planowanie chwytu',
                'Teleoperacja VR/WebRTC oraz nadzór operatora',
                'Stabilizacja ruchu i walidacja trajektorii w pętli zamkniętej'
            ]
        }
    };

    const spotlightTitle = document.getElementById('spotlightTitle');
    const spotlightText = document.getElementById('spotlightText');
    const spotlightList = document.getElementById('spotlightList');
    const spotlightCards = document.querySelectorAll('.spotlight-card');
    const spotlightButtons = document.querySelectorAll('[data-spotlight-target]');

    function renderSpotlight(key) {
        const content = spotlightContent[key];
        if (!content || !spotlightTitle || !spotlightText || !spotlightList) return;

        spotlightTitle.textContent = content.title;
        spotlightText.textContent = content.text;
        spotlightList.innerHTML = content.items.map(item => `<li>${item}</li>`).join('');
        spotlightCards.forEach(card => card.classList.toggle('is-active', card.dataset.spotlight === key));
    }

    spotlightButtons.forEach(button => {
        button.addEventListener('click', function() {
            renderSpotlight(this.dataset.spotlightTarget);
        });
    });

    const counters = document.querySelectorAll('[data-counter]');
    const counterObserver = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = Number(el.dataset.counter || 0);
            let current = 0;
            const step = Math.max(1, Math.ceil(target / 24));
            const timer = setInterval(() => {
                current += step;
                if (current >= target) {
                    el.textContent = String(target);
                    clearInterval(timer);
                    return;
                }
                el.textContent = String(current);
            }, 36);
            counterObserver.unobserve(el);
        });
    }, { threshold: 0.4 }) : null;

    counters.forEach(counter => {
        if (counterObserver) {
            counterObserver.observe(counter);
        } else {
            counter.textContent = counter.dataset.counter;
        }
    });

    renderScenario('rehab');
    renderSpotlight('perception');
});

// ====================================
// Task Reminders — Weekly Task Panel
// ====================================
(function () {
    'use strict';

    function escapeHTML(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    const MS_PER_DAY = 86400000;

    function getISOWeekString(date) {
        const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
        const day = d.getUTCDay() || 7;
        d.setUTCDate(d.getUTCDate() + 4 - day);
        const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
        const week = Math.ceil(((d - yearStart) / MS_PER_DAY + 1) / 7);
        return `${d.getUTCFullYear()}-W${String(week).padStart(2, '0')}`;
    }

    function createPWABottomNav() {
        if (document.querySelector('.pwa-bottom-nav')) return;

        const currentPage = window.location.pathname.split('/').pop() || 'index.html';

        const links = [
            { href: 'index.html', icon: 'fa-house',    label: 'Start',   caption: 'Strona główna' },
            { href: 'wiki.html',  icon: 'fa-book',     label: 'WIKI',    caption: 'Baza wiedzy'   },
            { href: 'pdf.html',   icon: 'fa-file-pdf', label: 'PDF',     caption: 'Materiały'     },
            { href: 'contact.html', icon: 'fa-envelope', label: 'Kontakt', caption: 'Napisz do nas' },
        ];

        const linksHTML = links.map(link => {
            const active = link.href === currentPage ? ' active' : '';
            return `<a href="${link.href}" class="pwa-bottom-nav-item${active}" aria-label="${link.caption}">` +
                `<span class="pwa-bottom-nav-icon"><i class="fa-solid ${link.icon}" aria-hidden="true"></i></span>` +
                `<span class="pwa-bottom-nav-meta">` +
                `<span class="pwa-bottom-nav-label">${link.label}</span>` +
                `<span class="pwa-bottom-nav-caption">${link.caption}</span>` +
                `</span></a>`;
        }).join('');

        const nav = document.createElement('nav');
        nav.className = 'pwa-bottom-nav';
        nav.setAttribute('aria-label', 'Szybka nawigacja');
        nav.innerHTML =
            `<div class="pwa-bottom-nav-shell">` +
            linksHTML +
            `<button class="pwa-bottom-nav-item pwa-notif-btn" type="button" aria-label="Zadania tygodniowe">` +
            `<span class="pwa-bottom-nav-icon"><i class="fa-solid fa-bell" aria-hidden="true"></i></span>` +
            `<span class="pwa-bottom-nav-meta">` +
            `<span class="pwa-bottom-nav-label">Zadania</span>` +
            `<span class="pwa-bottom-nav-caption">Ten tydzień</span>` +
            `</span>` +
            `</button>` +
            `</div>`;

        document.body.appendChild(nav);

        nav.querySelector('.pwa-notif-btn').addEventListener('click', function () {
            const existing = document.querySelector('.notif-panel');
            if (existing) {
                existing.remove();
                this.classList.remove('notif-active');
                return;
            }
            const bellBtn = this;
            bellBtn.classList.add('notif-active');
            fetch('./task.json')
                .then(r => r.json())
                .then(data => renderNotifPanel(data, bellBtn))
                .catch(() => renderNotifPanel(null, bellBtn));
        });
    }

    function renderNotifPanel(data, bellBtn) {
        const currentWeek = getISOWeekString(new Date());
        const panel = document.createElement('div');
        panel.className = 'notif-panel';
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('aria-label', 'Zadania tygodniowe');

        let bodyHTML;
        if (data && data.weeks) {
            const relevantWeeks = Object.entries(data.weeks)
                .filter(([week]) => week >= currentWeek);

            if (relevantWeeks.length === 0) {
                bodyHTML = '<p class="notif-no-tasks">Brak zadań na ten i kolejne tygodnie.</p>';
            } else {
                bodyHTML = relevantWeeks.map(([week, teams]) => {
                    const label = week === currentWeek
                        ? `Bieżący tydzień (${escapeHTML(week)})`
                        : escapeHTML(week);
                    const teamsHTML = Object.entries(teams).map(([key, tasks]) => {
                        const name = escapeHTML((data.teams && data.teams[key]) || key);
                        const items = tasks.map(t => `<li>${escapeHTML(t)}</li>`).join('');
                        return `<div class="notif-team"><h4>${name}</h4><ol>${items}</ol></div>`;
                    }).join('');
                    return `<section><strong>${label}</strong>${teamsHTML}</section>`;
                }).join('');
            }
        } else {
            bodyHTML = '<p class="notif-no-tasks">Nie udało się załadować zadań.</p>';
        }

        panel.innerHTML =
            `<div class="notif-panel-header">` +
            `<h3><i class="fa-solid fa-bell" aria-hidden="true"></i> Zadania <small>${escapeHTML(currentWeek)}</small></h3>` +
            `<button class="notif-panel-close" aria-label="Zamknij">\u00D7</button>` +
            `</div>` +
            `<div class="notif-panel-body">${bodyHTML}</div>`;

        function closePanel() {
            panel.remove();
            if (bellBtn) bellBtn.classList.remove('notif-active');
        }

        panel.querySelector('.notif-panel-close').addEventListener('click', closePanel);
        panel.addEventListener('click', function (e) {
            if (e.target === panel) closePanel();
        });

        document.body.appendChild(panel);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createPWABottomNav);
    } else {
        createPWABottomNav();
    }
})();
