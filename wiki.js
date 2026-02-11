/**
 * WIKI System - Markdown Article Loader and Renderer
 * Laboratorium Robotów Humanoidalnych
 */

'use strict';

// Article database - maps article IDs to markdown files
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
    
    // Percepcja
    'computer-vision': { category: 'Percepcja', title: 'Computer Vision' },
    'affective-computing': { category: 'Percepcja', title: 'Informatyka Afektywna' },
    'emotion-recognition': { category: 'Percepcja', title: 'Rozpoznawanie Emocji' },
    'face-detection': { category: 'Percepcja', title: 'Detekcja Twarzy' },
    'lidar': { category: 'Percepcja', title: 'LiDAR 3D' },
    
    // Kognicja
    'llm': { category: 'Kognicja', title: 'Large Language Models (LLM)' },
    'vlm': { category: 'Kognicja', title: 'Vision-Language Models (VLM)' },
    'reinforcement-learning': { category: 'Kognicja', title: 'Uczenie przez Wzmacnianie' },
    'deep-learning': { category: 'Kognicja', title: 'Deep Learning' },
    
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
    'ethics': { category: 'Etyka', title: 'Etyka w Robotyce' },
    
    // Percepcja - dodatkowe
    'object-detection': { category: 'Percepcja', title: 'Detekcja Obiektów' },
    'pose-estimation': { category: 'Percepcja', title: 'Estymacja Pozy' },
    
    // Kognicja - dodatkowe
    'neural-networks': { category: 'Kognicja', title: 'Sieci Neuronowe' },
    'transformers': { category: 'Kognicja', title: 'Architektury Transformer' },
    'transfer-learning': { category: 'Kognicja', title: 'Transfer Learning' },
    
    // Robotyka - dodatkowe
    'slam': { category: 'Robotyka', title: 'SLAM - Lokalizacja i Mapowanie' },
    'imu': { category: 'Robotyka', title: 'IMU - Inertial Measurement Unit' },
    'sensor-fusion': { category: 'Robotyka', title: 'Fuzja Sensoryczna' }
};

document.addEventListener('DOMContentLoaded', function() {
    initWiki();
});

function initWiki() {
    // Initialize marked.js
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

    // Handle article links
    const articleLinks = document.querySelectorAll('[data-article]');
    articleLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const articleId = this.getAttribute('data-article');
            loadArticle(articleId);
            
            // Update active state
            articleLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Scroll to top on mobile
            if (window.innerWidth <= 768) {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    });

    // Search functionality
    const searchInput = document.getElementById('wikiSearch');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(function() {
            searchArticles(this.value);
        }, 300));
    }

    // Load article from URL hash
    const hash = window.location.hash.slice(1);
    if (hash && ARTICLES[hash]) {
        loadArticle(hash);
    }
}

/**
 * Load and render markdown article
 */
async function loadArticle(articleId) {
    const articleContainer = document.getElementById('wikiArticle');
    const breadcrumbs = document.getElementById('breadcrumbs');
    const welcomeContent = document.querySelector('.wiki-welcome');
    
    if (!ARTICLES[articleId]) {
        showError('Artykuł nie został znaleziony');
        return;
    }

    // Show loading state
    articleContainer.innerHTML = '<div class="loading"><i class="fa-solid fa-spinner fa-spin"></i> Ładowanie...</div>';
    
    // Hide welcome content
    if (welcomeContent) {
        welcomeContent.style.display = 'none';
    }

    try {
        const response = await fetch(ARTICLES[articleId]);
        
        if (!response.ok) {
            throw new Error('Nie udało się załadować artykułu');
        }

        const markdown = await response.text();
        const html = typeof marked !== 'undefined' ? marked.parse(markdown) : markdown;
        
        // Render article
        articleContainer.innerHTML = html;
        
        // Highlight code blocks
        if (typeof hljs !== 'undefined') {
            articleContainer.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }

        // Update breadcrumbs
        if (breadcrumbs && METADATA[articleId]) {
            const meta = METADATA[articleId];
            document.getElementById('currentCategory').textContent = meta.category;
            document.getElementById('currentArticle').textContent = meta.title;
            breadcrumbs.style.display = 'block';
        }

        // Update URL hash
        window.location.hash = articleId;

        // Add "Edit on GitHub" link
        addEditLink(articleId);

    } catch (error) {
        console.error('Error loading article:', error);
        showError('Nie udało się załadować artykułu. Spróbuj ponownie później.');
    }
}

/**
 * Show error message
 */
function showError(message) {
    const articleContainer = document.getElementById('wikiArticle');
    articleContainer.innerHTML = `
        <div class="wiki-error">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <h3>Błąd</h3>
            <p>${message}</p>
            <a href="wiki.html" class="btn-secondary">Powrót do strony głównej</a>
        </div>
    `;
}

/**
 * Add "Edit on GitHub" link
 */
function addEditLink(articleId) {
    const articleContainer = document.getElementById('wikiArticle');
    const editLink = document.createElement('div');
    editLink.className = 'wiki-edit-link';
    editLink.innerHTML = `
        <a href="https://github.com/AI-robot-lab/ai-robot-lab.github.io/edit/main/${ARTICLES[articleId]}" 
           target="_blank" 
           rel="noopener noreferrer">
            <i class="fa-brands fa-github"></i> Edytuj na GitHub
        </a>
    `;
    articleContainer.appendChild(editLink);
}

/**
 * Search articles
 */
function searchArticles(query) {
    const links = document.querySelectorAll('[data-article]');
    const lowerQuery = query.toLowerCase();

    links.forEach(link => {
        const text = link.textContent.toLowerCase();
        const category = link.closest('.wiki-category');
        
        if (text.includes(lowerQuery)) {
            link.style.display = 'block';
            if (category) {
                category.style.display = 'block';
            }
        } else {
            link.style.display = 'none';
        }
    });

    // Hide empty categories
    document.querySelectorAll('.wiki-category').forEach(category => {
        const visibleLinks = category.querySelectorAll('[data-article][style*="display: block"], [data-article]:not([style*="display: none"])');
        if (query && visibleLinks.length === 0) {
            category.style.display = 'none';
        } else {
            category.style.display = 'block';
        }
    });
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Handle internal wiki links
 */
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#wiki-"]')) {
        e.preventDefault();
        const articleId = e.target.getAttribute('href').replace('#wiki-', '');
        loadArticle(articleId);
    }
});

// Handle browser back/forward
window.addEventListener('hashchange', function() {
    const hash = window.location.hash.slice(1);
    if (hash && ARTICLES[hash]) {
        loadArticle(hash);
    }
});
