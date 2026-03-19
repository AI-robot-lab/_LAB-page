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
    './assets/icons/favicon.ico',
    './assets/icons/icon-192x192.png',
    './assets/icons/icon-512x512.png'
];

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
