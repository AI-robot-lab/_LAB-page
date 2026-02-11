/**
 * WIKI System - Markdown Article Loader and Renderer
 * Laboratorium Robotów Humanoidalnych
 * Version: 2.0
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
    'imu': 'wiki/imu.md',
    'sensor-fusion': 'wiki/sensor-fusion.md',
    
    // Percepcja
    'computer-vision': 'wiki/computer-vision.md',
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
    'imu': { category: 'Robotyka', title: 'IMU - Inertial Measurement Unit' },
    'sensor-fusion': { category: 'Robotyka', title: 'Fuzja Sensoryczna' },
    
    // Percepcja
    'computer-vision': { category: 'Percepcja', title: 'Computer Vision' },
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
    
    // Inne
    'hri': { category: 'Inne', title: 'Interakcja Człowiek-Robot' },
    'safety': { category: 'Inne', title: 'Bezpieczeństwo Robotów' },
    'ethics': { category: 'Inne', title: 'Etyka w Robotyce' }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initWiki();
});

function initWiki() {
    // Initialize marked.js for markdown rendering
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return code;
            },
            breaks: true,
            gfm: true
        });
    }

    // Set up event listeners
    setupArticleLinks();
    setupSearch();
    
    // Load article from URL hash if present
    const hash = window.location.hash.substring(1);
    if (hash) {
        loadArticle(hash);
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
        // Show all
        categories.forEach(cat => {
            cat.style.display = 'block';
            const links = cat.querySelectorAll('li');
            links.forEach(li => li.style.display = 'block');
        });
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
                li.style.display = 'block';
                hasVisibleLinks = true;
            } else {
                li.style.display = 'none';
            }
        });
        
        // Hide category if no visible links
        cat.style.display = hasVisibleLinks ? 'block' : 'none';
    });
}

async function loadArticle(articleId) {
    const articleContainer = document.getElementById('wikiArticle');
    const breadcrumbs = document.getElementById('breadcrumbs');
    
    if (!articleContainer) return;
    
    // Show loading
    articleContainer.innerHTML = '<div class="loading"><i class="fa-solid fa-spinner fa-spin"></i> Ładowanie...</div>';
    
    // Check if article exists
    if (!ARTICLES[articleId]) {
        showError('Artykuł nie został znaleziony');
        return;
    }
    
    try {
        // Fetch markdown file
        const response = await fetch(ARTICLES[articleId]);
        
        if (!response.ok) {
            throw new Error('Nie można załadować artykułu');
        }
        
        const markdown = await response.text();
        
        // Render markdown
        const html = marked.parse(markdown);
        articleContainer.innerHTML = html;
        
        // Highlight code blocks
        if (typeof hljs !== 'undefined') {
            articleContainer.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
        
        // Update breadcrumbs
        updateBreadcrumbs(articleId);
        
        // Process internal links
        processInternalLinks(articleContainer);
        
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
            <p>Wybierz artykuł z menu po lewej stronie.</p>
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
