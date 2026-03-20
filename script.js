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
            // After registration, check if we should auto-notify for the current week.
            if (Notification.permission === 'granted') {
                notifyCurrentWeekIfNeeded(registration);
            }
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
            { href: 'index.html', icon: 'fa-solid fa-house', label: 'Główna', id: 'index.html' },
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

        // Notification bell button
        var bellBtn = document.createElement('button');
        bellBtn.type = 'button';
        bellBtn.className = 'pwa-bottom-nav-item pwa-notif-btn';
        bellBtn.setAttribute('aria-label', 'Powiadomienia o zadaniach');
        bellBtn.innerHTML = '<i class="fa-solid fa-bell" aria-hidden="true"></i><span>Zadania</span>';
        if (Notification.permission === 'granted') {
            bellBtn.classList.add('notif-active');
        }
        bellBtn.addEventListener('click', function() {
            openNotificationPanel();
        });
        nav.appendChild(bellBtn);

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

// ====================================
// PWA: Push Notifications – Weekly Tasks
// ====================================
(function() {
    var TASKS_URL = './tasks.json';
    var PERIODIC_SYNC_TAG = 'weekly-tasks-sync';
    var LS_NOTIF_WEEK = 'robotlab-notified-week';

    /**
     * Returns the ISO week key for a given date, e.g. "2026-W12".
     */
    function getISOWeekKey(date) {
        var d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
        var day = d.getUTCDay() || 7;
        d.setUTCDate(d.getUTCDate() + 4 - day);
        var yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
        var weekNo = Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
        return d.getUTCFullYear() + '-W' + String(weekNo).padStart(2, '0');
    }

    /**
     * Fetch tasks.json and return the entry for the current week, or null.
     */
    function fetchCurrentWeekTasks() {
        return fetch(TASKS_URL)
            .then(function(res) {
                if (!res.ok) return null;
                return res.json();
            })
            .then(function(data) {
                if (!data) return null;
                var weekKey = getISOWeekKey(new Date());
                if (data.weeks && data.weeks[weekKey]) {
                    return { weekKey: weekKey, teams: data.teams, tasks: data.weeks[weekKey] };
                }
                return null;
            })
            .catch(function() { return null; });
    }

    /**
     * Register Periodic Background Sync so the SW can notify once a week
     * even when the app is closed (Chrome/Android).
     */
    function registerPeriodicSync(registration) {
        if (!('periodicSync' in registration)) return;
        registration.periodicSync.register(PERIODIC_SYNC_TAG, {
            minInterval: 7 * 24 * 60 * 60 * 1000 // one week in ms
        }).catch(function() { /* permission may be denied – silently ignore */ });
    }

    /**
     * Ask the SW to show notifications for the current week's tasks via postMessage.
     * Saves the notified week to localStorage to avoid repeating on every page load.
     */
    function notifyViaServiceWorker(registration, weekKey, teams, tasks) {
        if (!registration.active) return;
        registration.active.postMessage({
            type: 'SHOW_WEEKLY_TASKS',
            weekKey: weekKey,
            teams: teams,
            tasks: tasks
        });
        localStorage.setItem(LS_NOTIF_WEEK, weekKey);
    }

    /**
     * Called after SW registration (and when user manually enables notifications).
     * Shows notifications if the current week hasn't been notified yet.
     */
    window.notifyCurrentWeekIfNeeded = function(registration) {
        var currentWeek = getISOWeekKey(new Date());
        var lastNotifiedWeek = localStorage.getItem(LS_NOTIF_WEEK);
        if (lastNotifiedWeek === currentWeek) return; // already notified this week

        fetchCurrentWeekTasks().then(function(result) {
            if (!result) return;
            notifyViaServiceWorker(registration, result.weekKey, result.teams, result.tasks);
            registerPeriodicSync(registration);
        });
    };

    /**
     * Open the notification management panel (permission request + current week tasks preview).
     */
    window.openNotificationPanel = function() {
        if (document.querySelector('.notif-panel')) return;

        fetchCurrentWeekTasks().then(function(result) {
            var panel = document.createElement('div');
            panel.className = 'notif-panel';
            panel.setAttribute('role', 'dialog');
            panel.setAttribute('aria-modal', 'true');
            panel.setAttribute('aria-labelledby', 'notif-panel-title');

            var weekLabel = result ? result.weekKey.replace('-W', ', tydzień ') : '';

            var tasksHtml = '';
            if (result) {
                for (var teamKey in result.teams) {
                    if (!Object.prototype.hasOwnProperty.call(result.teams, teamKey)) continue;
                    var teamTasks = result.tasks[teamKey];
                    if (!teamTasks || teamTasks.length === 0) continue;
                    tasksHtml += '<div class="notif-team">' +
                        '<h4>' + result.teams[teamKey] + '</h4><ol>';
                    teamTasks.forEach(function(t) {
                        tasksHtml += '<li>' + t + '</li>';
                    });
                    tasksHtml += '</ol></div>';
                }
            } else {
                tasksHtml = '<p class="notif-no-tasks">Brak zadań dla bieżącego tygodnia.</p>';
            }

            var notifGranted = Notification.permission === 'granted';
            var notifDenied  = Notification.permission === 'denied';

            panel.innerHTML =
                '<div class="notif-panel-header">' +
                    '<h3 id="notif-panel-title"><i class="fa-solid fa-bell" aria-hidden="true"></i> Zadania tygodnia' +
                        (weekLabel ? ' <small>(' + weekLabel + ')</small>' : '') +
                    '</h3>' +
                    '<button class="notif-panel-close" aria-label="Zamknij">&times;</button>' +
                '</div>' +
                '<div class="notif-panel-body">' +
                    tasksHtml +
                '</div>' +
                '<div class="notif-panel-footer">' +
                    (!notifDenied
                        ? '<button class="notif-enable-btn" ' + (notifGranted ? 'disabled' : '') + '>' +
                            (notifGranted
                                ? '<i class="fa-solid fa-check" aria-hidden="true"></i> Powiadomienia włączone'
                                : '<i class="fa-solid fa-bell" aria-hidden="true"></i> Włącz powiadomienia push') +
                          '</button>'
                        : '<p class="notif-denied-msg">Powiadomienia zablokowane w ustawieniach przeglądarki.</p>'
                    ) +
                    (notifGranted && result
                        ? '<button class="notif-test-btn"><i class="fa-solid fa-paper-plane" aria-hidden="true"></i> Wyślij powiadomienie teraz</button>'
                        : '') +
                '</div>';

            document.body.appendChild(panel);

            // Close button
            panel.querySelector('.notif-panel-close').addEventListener('click', function() {
                panel.remove();
            });

            // Enable notifications button
            var enableBtn = panel.querySelector('.notif-enable-btn');
            if (enableBtn && !notifGranted) {
                enableBtn.addEventListener('click', function() {
                    requestNotificationPermission();
                    panel.remove();
                });
            }

            // Test notification button
            var testBtn = panel.querySelector('.notif-test-btn');
            if (testBtn) {
                testBtn.addEventListener('click', function() {
                    if ('serviceWorker' in navigator) {
                        navigator.serviceWorker.ready.then(function(reg) {
                            if (result) {
                                // Clear the stored week so the notification fires immediately
                                // even if this week was already notified (test/preview mode).
                                localStorage.removeItem(LS_NOTIF_WEEK);
                                notifyViaServiceWorker(reg, result.weekKey, result.teams, result.tasks);
                            }
                        });
                    }
                    panel.remove();
                });
            }

            // Close on backdrop click
            panel.addEventListener('click', function(e) {
                if (e.target === panel) panel.remove();
            });
        });
    };

    /**
     * Request notification permission and, if granted, register sync and notify.
     */
    function requestNotificationPermission() {
        if (!('Notification' in window)) return;

        Notification.requestPermission().then(function(permission) {
            // Update bell button state
            var bellBtn = document.querySelector('.pwa-notif-btn');
            if (bellBtn) {
                if (permission === 'granted') {
                    bellBtn.classList.add('notif-active');
                } else {
                    bellBtn.classList.remove('notif-active');
                }
            }

            if (permission === 'granted' && 'serviceWorker' in navigator) {
                navigator.serviceWorker.ready.then(function(registration) {
                    localStorage.removeItem(LS_NOTIF_WEEK); // force notify immediately
                    notifyCurrentWeekIfNeeded(registration);
                    registerPeriodicSync(registration);
                });
            }
        });
    }

    /**
     * Show a first-time notification prompt banner (only once, only on mobile).
     */
    function maybeShowNotifPrompt() {
        if (!('Notification' in window)) return;
        if (Notification.permission !== 'default') return;
        if (localStorage.getItem('robotlab-notif-prompt-dismissed')) return;
        if (window.innerWidth > 768) return; // mobile only

        var banner = document.createElement('div');
        banner.className = 'notif-prompt-banner';
        banner.setAttribute('role', 'alertdialog');
        banner.setAttribute('aria-live', 'polite');
        banner.innerHTML =
            '<div class="notif-prompt-content">' +
                '<i class="fa-solid fa-bell notif-prompt-icon" aria-hidden="true"></i>' +
                '<div class="notif-prompt-text">' +
                    '<strong>Zadania tygodnia</strong>' +
                    '<span>Włącz powiadomienia push, aby otrzymywać zadania dla swojego zespołu każdego tygodnia.</span>' +
                '</div>' +
            '</div>' +
            '<div class="notif-prompt-actions">' +
                '<button class="notif-prompt-enable" aria-label="Włącz powiadomienia">Włącz</button>' +
                '<button class="notif-prompt-close" aria-label="Nie teraz">Nie teraz</button>' +
            '</div>';

        document.body.appendChild(banner);

        banner.querySelector('.notif-prompt-enable').addEventListener('click', function() {
            banner.remove();
            requestNotificationPermission();
        });

        banner.querySelector('.notif-prompt-close').addEventListener('click', function() {
            banner.remove();
            localStorage.setItem('robotlab-notif-prompt-dismissed', '1');
        });
    }

    // Initialise on DOMContentLoaded.
    function init() {
        // Show the first-visit prompt after a short delay so it doesn't clash with install banner.
        setTimeout(maybeShowNotifPrompt, 3000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
