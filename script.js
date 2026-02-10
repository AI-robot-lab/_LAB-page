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
});

// ====================================
// Smooth Scrolling Enhancement
// ====================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        
        // Skip empty anchors
        if (href === '#') {
            e.preventDefault();
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
// Service Worker Registration (Optional)
// ====================================
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Uncomment when you create a service worker
        // navigator.serviceWorker.register('/sw.js').then(function(registration) {
        //     console.log('ServiceWorker registered:', registration);
        // }).catch(function(error) {
        //     console.log('ServiceWorker registration failed:', error);
        // });
    });
}
