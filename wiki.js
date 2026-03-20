/**
 * WIKI System - Markdown Article Loader and Renderer
 * Laboratorium Robotów Humanoidalnych
 * Version: 3.0
 */

'use strict';

// Article database - maps article IDs to markdown files
// Wszystkie pliki są w folderze wiki/
const ARTICLES = {
    // Robotyka
    'ros2': 'wiki/ros2.md',
    'isaac-lab': 'wiki/isaac-lab.md',
    'unitree-g1': 'wiki/unitree-g1.md',
    'pca-framework': 'wiki/pca-framework.md',
    'slam': 'wiki/slam.md',
    'slam_marker_map': 'wiki/slam_marker_map.md',
    'imu': 'wiki/imu.md',
    'sensor-fusion': 'wiki/sensor-fusion.md',

    // Percepcja
    'computer-vision': 'wiki/computer-vision.md',
    'camera-spatial-orientation': 'wiki/camera-spatial-orientation.md',
    'kalibracja_kamery': 'wiki/kalibracja_kamery.md',
    'markery_wizyjne_nawigacja': 'wiki/markery_wizyjne_nawigacja.md',
    'affective-computing': 'wiki/affective-computing.md',
    'emotion-recognition': 'wiki/emotion-recognition.md',
    'face-detection': 'wiki/face-detection.md',
    'lidar': 'wiki/lidar.md',
    'object-detection': 'wiki/object-detection.md',
    'pose-estimation': 'wiki/pose-estimation.md',

    // Kognicja
    'llm': 'wiki/llm.md',
    'vlm': 'wiki/vlm.md',
    'reinforcement-learning': 'wiki/reinforcement-learning.md',
    'deep-learning': 'wiki/deep-learning.md',
    'neural-networks': 'wiki/neural-networks.md',
    'transformers': 'wiki/transformers.md',
    'transfer-learning': 'wiki/transfer-learning.md',

    // Akcja
    'motion-planning': 'wiki/motion-planning.md',
    'manipulation': 'wiki/manipulation.md',
    'sim-to-real': 'wiki/sim-to-real.md',
    'control-theory': 'wiki/control-theory.md',
    'kinematics': 'wiki/kinematics.md',
    'trajectory-optimization': 'wiki/trajectory-optimization.md',

    // Technologie
    'pytorch': 'wiki/pytorch.md',
    'opencv': 'wiki/opencv.md',
    'mediapipe': 'wiki/mediapipe.md',
    'deepface': 'wiki/deepface.md',
    'moveit2': 'wiki/moveit2.md',
    'docker': 'wiki/docker.md',

    // Praca w zespole
    'praca-w-zespole': 'wiki/praca-w-zespole.md',
    'bash-ssh-jetson': 'wiki/bash-ssh-jetson.md',
    'zakladanie-repozytorium-zadaniowego': 'wiki/zakladanie-repozytorium-zadaniowego.md',
    'github-releases': 'wiki/github-releases.md',
    'github_releases_packages': 'wiki/github_releases_packages.md',
    'git-github': 'wiki/git-github.md',
    'github-tests': 'wiki/github-tests.md',

    // Inne
    'hri': 'wiki/hri.md',
    'safety': 'wiki/safety.md',
    'ethics': 'wiki/ethics.md'
};

// Article metadata
const METADATA = {
    // Robotyka
    'ros2': { category: 'Robotyka', title: 'ROS2 - Robot Operating System' },
    'isaac-lab': { category: 'Robotyka', title: 'NVIDIA Isaac Lab' },
    'unitree-g1': { category: 'Robotyka', title: 'Unitree G1 - Specyfikacja' },
    'pca-framework': { category: 'Robotyka', title: 'Framework PCA' },
    'slam': { category: 'Robotyka', title: 'SLAM - Lokalizacja i Mapowanie' },
    'slam_marker_map': { category: 'Robotyka', title: 'SLAM - Mapa Markerów' },
    'imu': { category: 'Robotyka', title: 'IMU - Inertial Measurement Unit' },
    'sensor-fusion': { category: 'Robotyka', title: 'Fuzja Sensoryczna' },

    // Percepcja
    'computer-vision': { category: 'Percepcja', title: 'Computer Vision' },
    'camera-spatial-orientation': { category: 'Percepcja', title: 'Kamera w Orientacji Przestrzennej' },
    'kalibracja_kamery': { category: 'Percepcja', title: 'Kalibracja Kamery Robota' },
    'markery_wizyjne_nawigacja': { category: 'Percepcja', title: 'Markery Wizyjne w Nawigacji' },
    'affective-computing': { category: 'Percepcja', title: 'Informatyka Afektywna' },
    'emotion-recognition': { category: 'Percepcja', title: 'Rozpoznawanie Emocji' },
    'face-detection': { category: 'Percepcja', title: 'Detekcja Twarzy' },
    'lidar': { category: 'Percepcja', title: 'LiDAR 3D' },
    'object-detection': { category: 'Percepcja', title: 'Detekcja Obiektów' },
    'pose-estimation': { category: 'Percepcja', title: 'Estymacja Pozy' },

    // Kognicja
    'llm': { category: 'Kognicja', title: 'Large Language Models (LLM)' },
    'vlm': { category: 'Kognicja', title: 'Vision-Language Models (VLM)' },
    'reinforcement-learning': { category: 'Kognicja', title: 'Uczenie przez Wzmacnianie' },
    'deep-learning': { category: 'Kognicja', title: 'Deep Learning' },
    'neural-networks': { category: 'Kognicja', title: 'Sieci Neuronowe' },
    'transformers': { category: 'Kognicja', title: 'Architektury Transformer' },
    'transfer-learning': { category: 'Kognicja', title: 'Transfer Learning' },

    // Akcja
    'motion-planning': { category: 'Akcja', title: 'Planowanie Ruchu' },
    'manipulation': { category: 'Akcja', title: 'Manipulacja Robotyczna' },
    'sim-to-real': { category: 'Akcja', title: 'Transfer Sim-to-Real' },
    'control-theory': { category: 'Akcja', title: 'Teoria Sterowania' },
    'kinematics': { category: 'Akcja', title: 'Kinematyka Robotów' },
    'trajectory-optimization': { category: 'Akcja', title: 'Optymalizacja Trajektorii' },

    // Technologie
    'pytorch': { category: 'Technologie', title: 'PyTorch' },
    'opencv': { category: 'Technologie', title: 'OpenCV' },
    'mediapipe': { category: 'Technologie', title: 'MediaPipe' },
    'deepface': { category: 'Technologie', title: 'DeepFace' },
    'moveit2': { category: 'Technologie', title: 'MoveIt2' },
    'docker': { category: 'Technologie', title: 'Docker dla Robotyki' },

    // Praca w zespole
    'praca-w-zespole': { category: 'Praca w zespole', title: 'Praca w zespole inżynierskim' },
    'bash-ssh-jetson': { category: 'Praca w zespole', title: 'Bash i dostęp do robota (Jetson) po SSH' },
    'zakladanie-repozytorium-zadaniowego': { category: 'Praca w zespole', title: 'Tworzenie nowego repozytorium zadaniowego' },
    'github-releases': { category: 'Praca w zespole', title: 'GitHub Releases — czym są i jak je tworzyć' },
    'github_releases_packages': { category: 'Praca w zespole', title: 'GitHub Releases i Packages' },
    'git-github': { category: 'Praca w zespole', title: 'Git i GitHub — kompleksowy poradnik' },
    'github-tests': { category: 'Praca w zespole', title: 'Testy w GitHub — profesjonalny przewodnik dla studentów' },

    // Inne
    'hri': { category: 'Inne', title: 'Interakcja Człowiek-Robot' },
    'safety': { category: 'Inne', title: 'Bezpieczeństwo Robotów' },
    'ethics': { category: 'Inne', title: 'Etyka w Robotyce' }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Starting WIKI initialization...');
    initWiki();
    initDarkMode();
    initScrollProgress();
});

function initDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');

    // Check saved preference
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Update icon
    updateDarkModeIcon(savedTheme);

    // Toggle functionality
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            // Smooth transition
            document.documentElement.style.transition = 'background-color 0.2s ease, color 0.2s ease';

            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            updateDarkModeIcon(newTheme);

            // Remove transition after animation
            setTimeout(() => {
                document.documentElement.style.transition = '';
            }, 300);
        });
    }
}

function updateDarkModeIcon(theme) {
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (!darkModeToggle) return;

    const icon = darkModeToggle.querySelector('i');
    if (theme === 'dark') {
        icon.className = 'fa-solid fa-sun';
    } else {
        icon.className = 'fa-solid fa-moon';
    }
}

function initScrollProgress() {
    // Create scroll progress bar
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    document.body.appendChild(progressBar);

    // Update on scroll
    window.addEventListener('scroll', function() {
        const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (window.scrollY / windowHeight) * 100;
        progressBar.style.width = scrolled + '%';
    });
}

function initWiki() {
    // Initialize marked.js for markdown rendering
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true
        });
    }

    // Add smooth scroll to html
    document.documentElement.style.scrollBehavior = 'smooth';

    // Set up event listeners
    setupArticleLinks();
    setupSearch();

    // Load article from URL hash if present
    const hash = window.location.hash.substring(1);
    if (hash) {
        loadArticle(hash);
    }

    // Add fade-in animation to sidebar
    const sidebar = document.querySelector('.wiki-sidebar');
    if (sidebar) {
        sidebar.style.opacity = '0';
        sidebar.style.transform = 'translateX(-20px)';
        setTimeout(() => {
            sidebar.style.transition = 'all 0.5s ease';
            sidebar.style.opacity = '1';
            sidebar.style.transform = 'translateX(0)';
        }, 100);
    }

    // Setup collapsible categories
    setupCollapsibleCategories();

    // Add stagger animation to category links
    const categories = document.querySelectorAll('.wiki-category');
    categories.forEach((category, index) => {
        category.style.opacity = '0';
        category.style.transform = 'translateY(10px)';
        setTimeout(() => {
            category.style.transition = 'all 0.4s ease';
            category.style.opacity = '1';
            category.style.transform = 'translateY(0)';
        }, 150 + (index * 50));
    });
}

function setupCollapsibleCategories() {
    const categories = document.querySelectorAll('.wiki-category');

    categories.forEach(category => {
        const header = category.querySelector('h4');
        const list = category.querySelector('ul');
        if (!header || !list) return;

        // Add article count badge
        const articleCount = list.querySelectorAll('li').length;
        const badge = document.createElement('span');
        badge.className = 'category-count';
        badge.textContent = articleCount;
        header.appendChild(badge);

        // Add chevron icon
        const chevron = document.createElement('i');
        chevron.className = 'fa-solid fa-chevron-down category-chevron';
        header.appendChild(chevron);

        // Start collapsed
        category.classList.add('collapsed');

        // Toggle on click
        header.addEventListener('click', function(e) {
            e.preventDefault();
            category.classList.toggle('collapsed');
        });
    });

    // If there's a hash, expand the category containing that article
    const hash = window.location.hash.substring(1);
    if (hash) {
        expandCategoryForArticle(hash);
    }
}

function expandCategoryForArticle(articleId) {
    const link = document.querySelector('[data-article="' + CSS.escape(articleId) + '"]');
    if (link) {
        const category = link.closest('.wiki-category');
        if (category) {
            category.classList.remove('collapsed');
        }
    }
}

function setupArticleLinks() {
    // Get all article links
    const articleLinks = document.querySelectorAll('[data-article]');

    articleLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            // Remove active class from all links
            articleLinks.forEach(l => l.classList.remove('active'));

            // Add active class to clicked link
            this.classList.add('active');

            // Get article ID (remove 'wiki/' prefix if present)
            let articleId = this.dataset.article;
            articleId = articleId.replace('wiki/', '');

            // Update URL hash
            window.location.hash = articleId;

            // Expand parent category if collapsed
            expandCategoryForArticle(articleId);

            // Load article
            loadArticle(articleId);

            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
}

function setupSearch() {
    const searchInput = document.getElementById('wikiSearch');
    if (!searchInput) return;

    let searchTimeout;

    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);

        searchTimeout = setTimeout(() => {
            const query = this.value.toLowerCase().trim();
            filterArticles(query);
        }, 300);
    });
}

function filterArticles(query) {
    const categories = document.querySelectorAll('.wiki-category');

    if (!query) {
        // Show all, restore collapsed state
        categories.forEach(cat => {
            cat.style.display = '';
            cat.classList.add('collapsed');
            const links = cat.querySelectorAll('li');
            links.forEach(li => li.style.display = '');
        });
        // Re-expand category of active article if any
        const activeLink = document.querySelector('.wiki-category a.active');
        if (activeLink) {
            const activeCat = activeLink.closest('.wiki-category');
            if (activeCat) activeCat.classList.remove('collapsed');
        }
        return;
    }

    categories.forEach(cat => {
        const links = cat.querySelectorAll('li');
        let hasVisibleLinks = false;

        links.forEach(li => {
            const link = li.querySelector('a');
            const text = link.textContent.toLowerCase();
            const articleId = link.dataset.article.replace('wiki/', '');

            if (text.includes(query) || articleId.includes(query)) {
                li.style.display = '';
                hasVisibleLinks = true;
            } else {
                li.style.display = 'none';
            }
        });

        // Hide category if no visible links, expand matching ones
        if (hasVisibleLinks) {
            cat.style.display = '';
            cat.classList.remove('collapsed');
        } else {
            cat.style.display = 'none';
        }
    });
}

async function loadArticle(articleId) {
    const articleContainer = document.getElementById('wikiArticle');
    const breadcrumbs = document.getElementById('breadcrumbs');

    if (!articleContainer) {
        console.error('Article container not found');
        return;
    }

    // Show loading with progress bar
    articleContainer.innerHTML = `
        <div class="loading">
            <i class="fa-solid fa-spinner fa-spin"></i>
            <div class="loading-text">Ładowanie artykułu...</div>
            <div class="loading-progress">
                <div class="loading-progress-bar"></div>
            </div>
        </div>
    `;

    // Check if article exists
    if (!ARTICLES[articleId]) {
        console.error('Article not found:', articleId);
        showError('Artykuł nie został znaleziony');
        return;
    }

    try {
        const articlePath = ARTICLES[articleId];
        console.log('Loading article from:', articlePath);

        // Fetch markdown file
        const response = await fetch(articlePath);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const markdown = await response.text();
        console.log('Markdown loaded, length:', markdown.length);

        // Wait for marked if not yet available (up to 3 s), then fall back to built-in parser
        if (typeof marked === 'undefined') {
            await new Promise((resolve) => {
                let attempts = 0;
                const interval = setInterval(() => {
                    attempts++;
                    if (typeof marked !== 'undefined' || attempts >= 12) {
                        clearInterval(interval);
                        resolve();
                    }
                }, 250);
            });
        }

        // Render markdown using marked.js when available, otherwise use built-in parser
        const html = typeof marked !== 'undefined'
            ? marked.parse(markdown)
            : parseMarkdownFallback(markdown);

        // Fade in animation
        articleContainer.style.opacity = '0';
        articleContainer.innerHTML = html;

        // Smooth fade in
        setTimeout(() => {
            articleContainer.style.transition = 'opacity 0.3s ease';
            articleContainer.style.opacity = '1';
        }, 50);

        // Highlight code blocks
        if (typeof hljs !== 'undefined') {
            articleContainer.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }

        // Add copy buttons to code blocks
        addCopyButtons(articleContainer);

        // Add reading time estimate
        addReadingTime(articleContainer);

        // Generate table of contents
        generateTableOfContents(articleContainer);

        // Update breadcrumbs
        updateBreadcrumbs(articleId);

        // Process internal links
        processInternalLinks(articleContainer);

        // Add smooth scroll behavior
        articleContainer.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

    } catch (error) {
        console.error('Error loading article:', error);
        showError('Błąd podczas ładowania artykułu');
    }
}

function updateBreadcrumbs(articleId) {
    const breadcrumbs = document.getElementById('breadcrumbs');
    if (!breadcrumbs || !METADATA[articleId]) return;

    const metadata = METADATA[articleId];

    document.getElementById('currentCategory').textContent = metadata.category;
    document.getElementById('currentArticle').textContent = metadata.title;

    breadcrumbs.style.display = 'flex';
}

function processInternalLinks(container) {
    // Find all links that start with #wiki-
    const links = container.querySelectorAll('a[href^="#wiki-"]');

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const articleId = this.getAttribute('href').replace('#wiki-', '');
            window.location.hash = articleId;
            loadArticle(articleId);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
}

function showError(message) {
    const articleContainer = document.getElementById('wikiArticle');

    articleContainer.innerHTML = `
        <div class="wiki-error">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <h3>${message}</h3>
            <p>Wybierz artykuł z menu po lewej stronie lub użyj wyszukiwarki.</p>
            <div style="margin-top: 25px;">
                <a href="#" onclick="window.location.reload()" style="
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px 24px;
                    background: linear-gradient(135deg, var(--prz-blue) 0%, #004d99 100%);
                    color: white;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    transition: all 0.3s ease;
                ">
                    <i class="fa-solid fa-rotate-right"></i>
                    Odśwież stronę
                </a>
            </div>
        </div>
    `;

    const breadcrumbs = document.getElementById('breadcrumbs');
    if (breadcrumbs) {
        breadcrumbs.style.display = 'none';
    }
}

// Handle back/forward navigation
window.addEventListener('hashchange', function() {
    const hash = window.location.hash.substring(1);
    if (hash) {
        loadArticle(hash);

        // Update active link
        const articleLinks = document.querySelectorAll('[data-article]');
        articleLinks.forEach(link => {
            const linkId = link.dataset.article.replace('wiki/', '');
            if (linkId === hash) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
});

function addCopyButtons(container) {
    /**
     * Dodaj przyciski kopiowania do wszystkich bloków kodu
     */
    const codeBlocks = container.querySelectorAll('pre');

    codeBlocks.forEach((pre, index) => {
        // Wrap in container
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        // Create copy button
        const button = document.createElement('button');
        button.className = 'copy-code-btn';
        button.innerHTML = '<i class="fa-solid fa-copy"></i> Copy';

        button.addEventListener('click', async function() {
            const code = pre.querySelector('code').textContent;

            try {
                await navigator.clipboard.writeText(code);

                // Visual feedback
                button.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
                button.classList.add('copied');

                setTimeout(() => {
                    button.innerHTML = '<i class="fa-solid fa-copy"></i> Copy';
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
                button.innerHTML = '<i class="fa-solid fa-xmark"></i> Failed';
            }
        });

        wrapper.appendChild(button);
    });
}

function addReadingTime(container) {
    /**
     * Oblicz i wyświetl szacowany czas czytania
     */
    const text = container.textContent;
    const wordsPerMinute = 200;
    const words = text.trim().split(/\s+/).length;
    const minutes = Math.ceil(words / wordsPerMinute);

    // Create reading time badge
    const badge = document.createElement('div');
    badge.className = 'reading-time';
    badge.innerHTML = `
        <i class="fa-solid fa-clock"></i>
        <span>${minutes} min czytania</span>
    `;

    // Insert after h1
    const h1 = container.querySelector('h1');
    if (h1 && h1.nextSibling) {
        h1.parentNode.insertBefore(badge, h1.nextSibling);
    }
}

function generateTableOfContents(container) {
    /**
     * Generuj spis treści z nagłówków
     */
    const headings = container.querySelectorAll('h2, h3');

    if (headings.length < 3) return; // Don't show TOC for short articles

    const toc = document.createElement('div');
    toc.className = 'article-toc';
    toc.innerHTML = '<h3><i class="fa-solid fa-list"></i> Spis Treści</h3><ul></ul>';

    const ul = toc.querySelector('ul');

    headings.forEach((heading, index) => {
        // Add ID for linking
        const id = `heading-${index}`;
        heading.id = id;

        // Create TOC item
        const li = document.createElement('li');
        const level = heading.tagName === 'H2' ? 0 : 20;
        li.style.paddingLeft = level + 'px';

        const link = document.createElement('a');
        link.href = `#${id}`;
        link.textContent = heading.textContent;
        link.addEventListener('click', function(e) {
            e.preventDefault();
            heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });

        li.appendChild(link);
        ul.appendChild(li);
    });

    // Insert after reading time or h1
    const readingTime = container.querySelector('.reading-time');
    const h1 = container.querySelector('h1');

    if (readingTime) {
        readingTime.parentNode.insertBefore(toc, readingTime.nextSibling);
    } else if (h1) {
        h1.parentNode.insertBefore(toc, h1.nextSibling);
    }
}

/**
 * Built-in fallback markdown parser.
 * Used when the marked.js CDN library is not available.
 * Supports: headings, bold, italic, inline code, fenced code blocks,
 *           unordered/ordered lists, blockquotes, horizontal rules,
 *           links, images, and paragraphs.
 */
function parseMarkdownFallback(markdown) {
    function escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function sanitizeUrl(url) {
        const trimmed = url.trim();
        // Allow only http, https, mailto, and relative URLs
        if (/^(https?:|mailto:|\/|\.\/|\.\.\/|#)/.test(trimmed)) {
            return trimmed;
        }
        return '#';
    }

    function inlineFormat(escapedText) {
        // Input is already HTML-escaped; only process markdown markers
        // Bold + italic
        escapedText = escapedText.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        // Bold
        escapedText = escapedText.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        escapedText = escapedText.replace(/__(.+?)__/g, '<strong>$1</strong>');
        // Italic
        escapedText = escapedText.replace(/\*(.+?)\*/g, '<em>$1</em>');
        escapedText = escapedText.replace(/_(.+?)_/g, '<em>$1</em>');
        // Inline code — content is already HTML-escaped; no extra escaping needed
        escapedText = escapedText.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Images — validate src URL; alt text is already escaped via escapeHtml
        escapedText = escapedText.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
            const safeSrc = sanitizeUrl(src);
            return `<img src="${safeSrc}" alt="${alt}" style="max-width:100%">`;
        });
        // Links — validate href URL; link text is already escaped via escapeHtml
        escapedText = escapedText.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, href) => {
            const safeHref = sanitizeUrl(href);
            return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${text}</a>`;
        });
        return escapedText;
    }

    const lines = markdown.split('\n');
    const html = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // Fenced code block
        const fenceMatch = line.match(/^```(\w*)/);
        if (fenceMatch) {
            const lang = fenceMatch[1] || '';
            const codeLines = [];
            i++;
            while (i < lines.length && !lines[i].startsWith('```')) {
                codeLines.push(escapeHtml(lines[i]));
                i++;
            }
            i++; // skip closing ```
            html.push(`<pre><code class="language-${lang}">${codeLines.join('\n')}</code></pre>`);
            continue;
        }

        // Headings
        const headingMatch = line.match(/^(#{1,6})\s+(.*)/);
        if (headingMatch) {
            const level = headingMatch[1].length;
            html.push(`<h${level}>${inlineFormat(escapeHtml(headingMatch[2]))}</h${level}>`);
            i++;
            continue;
        }

        // Horizontal rule
        if (/^[-*_]{3,}\s*$/.test(line)) {
            html.push('<hr>');
            i++;
            continue;
        }

        // Blockquote
        if (line.startsWith('> ')) {
            const quoteLines = [];
            while (i < lines.length && lines[i].startsWith('> ')) {
                quoteLines.push(inlineFormat(escapeHtml(lines[i].slice(2))));
                i++;
            }
            html.push('<blockquote>' + quoteLines.join('<br>') + '</blockquote>');
            continue;
        }

        // Unordered list
        if (/^[\*\-]\s+/.test(line)) {
            const items = [];
            while (i < lines.length && /^[\*\-]\s+/.test(lines[i])) {
                items.push('<li>' + inlineFormat(escapeHtml(lines[i].replace(/^[\*\-]\s+/, ''))) + '</li>');
                i++;
            }
            html.push('<ul>' + items.join('') + '</ul>');
            continue;
        }

        // Ordered list
        if (/^\d+\.\s+/.test(line)) {
            const items = [];
            while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
                items.push('<li>' + inlineFormat(escapeHtml(lines[i].replace(/^\d+\.\s+/, ''))) + '</li>');
                i++;
            }
            html.push('<ol>' + items.join('') + '</ol>');
            continue;
        }

        // Empty line — paragraph break
        if (line.trim() === '') {
            i++;
            continue;
        }

        // Paragraph: collect consecutive non-empty, non-special lines
        const paraLines = [];
        while (
            i < lines.length &&
            lines[i].trim() !== '' &&
            !/^#{1,6}\s/.test(lines[i]) &&
            !/^[\*\-]\s/.test(lines[i]) &&
            !/^\d+\.\s/.test(lines[i]) &&
            !/^```/.test(lines[i]) &&
            !/^> /.test(lines[i]) &&
            !/^[-*_]{3,}\s*$/.test(lines[i])
        ) {
            paraLines.push(inlineFormat(escapeHtml(lines[i])));
            i++;
        }
        if (paraLines.length > 0) {
            html.push('<p>' + paraLines.join(' ') + '</p>');
        }
    }

    return html.join('\n');
}
