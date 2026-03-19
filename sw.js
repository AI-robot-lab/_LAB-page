const CACHE_NAME = 'robotlab-prz-v2';
const ASSETS_TO_CACHE = [
    './',
    './index.html',
    './contact.html',
    './wiki.html',
    './pdf.html',
    './styles.css',
    './script.js',
    './wiki.js',
    './contact-form.js',
    './manifest.json',
    './favicon.svg',
    './tasks.json',
    './assets/icons/favicon.ico',
    './assets/icons/icon-192x192.png',
    './assets/icons/icon-512x512.png'
];

const TASKS_URL = './tasks.json';
const PERIODIC_SYNC_TAG = 'weekly-tasks-sync';

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(ASSETS_TO_CACHE);
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((name) => name !== CACHE_NAME)
                    .map((name) => caches.delete(name))
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    if (event.request.method !== 'GET') return;

    const url = new URL(event.request.url);

    // Skip cross-origin requests (fonts, CDN, external images) — let browser handle them
    if (url.origin !== self.location.origin) return;

    // Cache-first for static assets, network-first for HTML
    const isHTML = event.request.headers.get('accept') &&
                   event.request.headers.get('accept').includes('text/html');

    if (isHTML) {
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                    return response;
                })
                .catch(() => caches.match(event.request))
        );
    } else {
        event.respondWith(
            caches.match(event.request).then((cached) => {
                if (cached) return cached;
                return fetch(event.request).then((response) => {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                    return response;
                });
            })
        );
    }
});

// ====================================
// Push Notifications
// ====================================

/**
 * Returns the ISO week key for a given date, e.g. "2026-W12".
 */
function getISOWeekKey(date) {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayOfWeek = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayOfWeek);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    const weekNo = Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
    return d.getUTCFullYear() + '-W' + String(weekNo).padStart(2, '0');
}

/**
 * Fetch tasks.json and return the entry for the current week, or null.
 */
async function fetchCurrentWeekTasks() {
    try {
        const response = await fetch(TASKS_URL);
        if (!response.ok) return null;
        const data = await response.json();
        const weekKey = getISOWeekKey(new Date());
        return data.weeks && data.weeks[weekKey] ? { weekKey, teams: data.teams, tasks: data.weeks[weekKey] } : null;
    } catch {
        return null;
    }
}

/**
 * Show a notification for each team's tasks for the current week.
 */
async function showWeeklyTaskNotifications(weekKey, teams, tasks) {
    for (const [teamKey, teamName] of Object.entries(teams)) {
        const teamTasks = tasks[teamKey];
        if (!teamTasks || teamTasks.length === 0) continue;

        const body = teamTasks.map((t, i) => (i + 1) + '. ' + t).join('\n');

        await self.registration.showNotification('RobotLab PRz \u2013 ' + teamName, {
            body,
            icon: './assets/icons/icon-192x192.png',
            badge: './assets/icons/icon-192x192.png',
            tag: 'weekly-tasks-' + weekKey + '-' + teamKey,
            renotify: false,
            data: { url: './index.html#teams', weekKey, teamKey }
        });
    }
}

// Handle push events sent by a push server (VAPID-based Web Push).
self.addEventListener('push', (event) => {
    let payload = null;

    if (event.data) {
        try {
            payload = event.data.json();
        } catch {
            payload = { title: 'RobotLab PRz', body: event.data.text() };
        }
    }

    if (payload && payload.weekKey && payload.teams && payload.tasks) {
        event.waitUntil(
            showWeeklyTaskNotifications(payload.weekKey, payload.teams, payload.tasks)
        );
        return;
    }

    // Generic push: fetch and display current week's tasks.
    event.waitUntil(
        fetchCurrentWeekTasks().then((result) => {
            if (!result) return Promise.resolve();
            return showWeeklyTaskNotifications(result.weekKey, result.teams, result.tasks);
        })
    );
});

// Handle Periodic Background Sync – fire once per week to remind teams of their tasks.
self.addEventListener('periodicsync', (event) => {
    if (event.tag === PERIODIC_SYNC_TAG) {
        event.waitUntil(
            fetchCurrentWeekTasks().then((result) => {
                if (!result) return Promise.resolve();
                return showWeeklyTaskNotifications(result.weekKey, result.teams, result.tasks);
            })
        );
    }
});

// Handle message from the main thread to trigger an immediate task notification.
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SHOW_WEEKLY_TASKS') {
        const { weekKey, teams, tasks } = event.data;
        if (weekKey && teams && tasks) {
            event.waitUntil(showWeeklyTaskNotifications(weekKey, teams, tasks));
        } else {
            event.waitUntil(
                fetchCurrentWeekTasks().then((result) => {
                    if (!result) return;
                    return showWeeklyTaskNotifications(result.weekKey, result.teams, result.tasks);
                })
            );
        }
    }
});

// Open the relevant page when the user taps a notification.
self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    const targetUrl = (event.notification.data && event.notification.data.url)
        ? event.notification.data.url
        : './index.html';

    event.waitUntil(
        self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
            for (const client of clientList) {
                if (client.url.includes(self.location.origin) && 'focus' in client) {
                    client.navigate(targetUrl);
                    return client.focus();
                }
            }
            if (self.clients.openWindow) {
                return self.clients.openWindow(targetUrl);
            }
        })
    );
});
