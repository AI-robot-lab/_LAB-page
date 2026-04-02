/**
 * Theme Utilities — shared across all pages
 * Politechnika Rzeszowska
 */

'use strict';

const DEFAULT_THEME = 'light';
const THEME_TRANSITION_MS = 300;
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

function initDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const savedTheme = getStoredTheme();
    document.documentElement.setAttribute('data-theme', savedTheme);
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

// Apply saved theme immediately to prevent flash of wrong theme
(function() {
    const savedTheme = getStoredTheme();
    document.documentElement.setAttribute('data-theme', savedTheme);
})();
