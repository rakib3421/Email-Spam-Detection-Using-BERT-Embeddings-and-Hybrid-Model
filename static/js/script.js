// Global variables
let isAnalyzing = false;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize the application
function initializeApp() {
    // Setup event listeners
    setupFormSubmission();
    setupNavigation();
    setupAnimations();
    
    console.log('SpamGuard application initialized');
}

// Form submission handling
function setupFormSubmission() {
    const form = document.getElementById('spam-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    if (form && analyzeBtn) {
        form.addEventListener('submit', handleFormSubmit);
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    if (isAnalyzing) return;
    
    const subject = document.getElementById('subject').value.trim();
    const message = document.getElementById('message').value.trim();
    
    // Validation
    if (!subject && !message) {
        showError('Please enter either a subject or message to analyze.');
        return;
    }
    
    // Start analysis
    startAnalysis();
    
    try {
        const result = await analyzeEmail(subject, message);
        displayResults(result);
    } catch (error) {
        console.error('Analysis error:', error);
        showError('An error occurred during analysis. Please try again.');
    } finally {
        stopAnalysis();
    }
}

// Start analysis (show loading state)
function startAnalysis() {
    isAnalyzing = true;
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    
    // Clear previous results
    clearResults();
    
    // Show analyzing message
    showAnalyzingState();
}

// Stop analysis (hide loading state)
function stopAnalysis() {
    isAnalyzing = false;
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.classList.remove('loading');
    analyzeBtn.disabled = false;
}

// Show analyzing state
function showAnalyzingState() {
    const resultContent = document.getElementById('result-content');
    const resultIcon = document.getElementById('result-icon');
    
    resultIcon.innerHTML = '<i class="fas fa-brain"></i>';
    resultIcon.style.background = 'linear-gradient(45deg, #6366f1, #8b5cf6)';
    
    resultContent.innerHTML = `
        <div class="analyzing-state">
            <div class="analysis-spinner"></div>
            <h3>Analyzing Email...</h3>
            <p>Processing with BERT transformers and ML ensemble</p>
            <div class="analysis-steps">
                <div class="step active">
                    <i class="fas fa-search"></i>
                    <span>Preprocessing text</span>
                </div>
                <div class="step">
                    <i class="fas fa-brain"></i>
                    <span>BERT encoding</span>
                </div>
                <div class="step">
                    <i class="fas fa-cogs"></i>
                    <span>Feature extraction</span>
                </div>
                <div class="step">
                    <i class="fas fa-check-circle"></i>
                    <span>Final prediction</span>
                </div>
            </div>
        </div>
    `;
    
    // Add analyzing styles
    const style = document.createElement('style');
    style.textContent = `
        .analyzing-state {
            text-align: center;
            width: 100%;
        }
        
        .analysis-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #e5e7eb;
            border-top: 4px solid #6366f1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 2rem;
        }
        
        .analyzing-state h3 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #374151;
            margin-bottom: 0.5rem;
        }
        
        .analyzing-state p {
            color: #6b7280;
            margin-bottom: 2rem;
        }
        
        .analysis-steps {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            text-align: left;
            max-width: 300px;
            margin: 0 auto;
        }
        
        .step {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem;
            border-radius: 0.5rem;
            background: #f9fafb;
            transition: all 0.3s ease;
        }
        
        .step.active {
            background: linear-gradient(45deg, #6366f1, #8b5cf6);
            color: white;
            transform: scale(1.02);
        }
        
        .step i {
            width: 20px;
            text-align: center;
        }
    `;
    document.head.appendChild(style);
    
    // Animate steps
    animateAnalysisSteps();
}

// Animate analysis steps
function animateAnalysisSteps() {
    const steps = document.querySelectorAll('.step');
    let currentStep = 0;
    
    const interval = setInterval(() => {
        if (currentStep > 0) {
            steps[currentStep - 1].classList.remove('active');
        }
        
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(interval);
        }
    }, 800);
}

// Analyze email via API
async function analyzeEmail(subject, message) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            subject: subject,
            message: message
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Analysis failed');
    }
    
    return await response.json();
}

// Display analysis results
function displayResults(result) {
    const resultContent = document.getElementById('result-content');
    const resultIcon = document.getElementById('result-icon');
    
    // Update icon based on result
    if (result.prediction === 'spam') {
        resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        resultIcon.style.background = 'linear-gradient(45deg, #ef4444, #fca5a5)';
    } else {
        resultIcon.innerHTML = '<i class="fas fa-shield-check"></i>';
        resultIcon.style.background = 'linear-gradient(45deg, #10b981, #86efac)';
    }
    
    // Display results
    resultContent.innerHTML = generateResultsHTML(result);
    
    // Add fade-in animation
    resultContent.classList.add('fade-in');
    
    // Auto-scroll to results
    document.getElementById('results-section').scrollIntoView({
        behavior: 'smooth',
        block: 'center'
    });
}

// Generate results HTML
function generateResultsHTML(result) {
    const isSpam = result.prediction === 'spam';
    const confidence = (result.confidence * 100).toFixed(1);
    const spamProb = (result.spam_probability * 100).toFixed(1);
    const hamProb = (result.ham_probability * 100).toFixed(1);
    
    return `
        <div class="prediction-result">
            <div class="prediction-badge ${result.prediction}">
                ${isSpam ? '🚨 SPAM DETECTED' : '✅ LEGITIMATE EMAIL'}
            </div>
            
            <div class="confidence-section">
                <h4 style="margin-bottom: 1rem; color: #374151;">Confidence Analysis</h4>
                <div class="confidence-bars">
                    <div class="confidence-item">
                        <div class="confidence-label" style="color: #ef4444;">Spam Probability</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill spam" style="width: ${spamProb}%"></div>
                        </div>
                        <div class="confidence-value" style="color: #ef4444;">${spamProb}%</div>
                    </div>
                    <div class="confidence-item">
                        <div class="confidence-label" style="color: #10b981;">Ham Probability</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill ham" style="width: ${hamProb}%"></div>
                        </div>
                        <div class="confidence-value" style="color: #10b981;">${hamProb}%</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 1rem;">
                    <strong>Overall Confidence: ${confidence}%</strong>
                </div>
            </div>
            
            ${result.features ? generateDetailsHTML(result.features) : ''}
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: #f9fafb; border-radius: 0.5rem; text-align: left;">
                <h5 style="margin-bottom: 0.5rem; color: #374151;">Analysis Summary:</h5>
                <p style="color: #6b7280; font-size: 0.875rem;">
                    This email was analyzed using BERT transformers for semantic understanding, 
                    TF-IDF for text features, and an ensemble of machine learning models. 
                    The prediction is based on ${result.features ? Object.keys(result.features).length : 'multiple'} 
                    features including text patterns, spam indicators, and statistical analysis.
                </p>
                ${result.analysis_time ? `<p style="color: #9ca3af; font-size: 0.75rem; margin-top: 0.5rem;">Analysis completed at: ${result.analysis_time}</p>` : ''}
            </div>
        </div>
    `;
}

// Generate details HTML
function generateDetailsHTML(features) {
    const keyFeatures = [
        { key: 'word_count', label: 'Word Count', format: (v) => v },
        { key: 'char_count', label: 'Character Count', format: (v) => v },
        { key: 'exclamation_count', label: 'Exclamation Marks', format: (v) => v },
        { key: 'caps_ratio', label: 'Caps Ratio', format: (v) => (v * 100).toFixed(1) + '%' },
        { key: 'spam_word_count', label: 'Spam Keywords', format: (v) => v },
        { key: 'has_url', label: 'Contains URL', format: (v) => v ? 'Yes' : 'No' },
        { key: 'has_email', label: 'Contains Email', format: (v) => v ? 'Yes' : 'No' },
        { key: 'unique_word_ratio', label: 'Word Diversity', format: (v) => (v * 100).toFixed(1) + '%' }
    ];
    
    const availableFeatures = keyFeatures.filter(f => features.hasOwnProperty(f.key));
    
    if (availableFeatures.length === 0) return '';
    
    return `
        <div class="details-section">
            <h5 style="margin-bottom: 1rem; color: #374151;">Key Features Analyzed:</h5>
            <div class="details-grid">
                ${availableFeatures.map(feature => `
                    <div class="detail-item">
                        <span class="detail-label">${feature.label}:</span>
                        <span class="detail-value">${feature.format(features[feature.key])}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// Clear results
function clearResults() {
    const resultContent = document.getElementById('result-content');
    resultContent.innerHTML = `
        <div class="placeholder-content">
            <i class="fas fa-envelope"></i>
            <p>Enter email content and click "Analyze Email" to see results</p>
        </div>
    `;
    
    const resultIcon = document.getElementById('result-icon');
    resultIcon.innerHTML = '<i class="fas fa-search"></i>';
    resultIcon.style.background = 'linear-gradient(45deg, #6b7280, #4b5563)';
}

// Show error message
function showError(message) {
    const resultContent = document.getElementById('result-content');
    const resultIcon = document.getElementById('result-icon');
    
    resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
    resultIcon.style.background = 'linear-gradient(45deg, #ef4444, #fca5a5)';
    
    resultContent.innerHTML = `
        <div class="error-state">
            <div class="error-icon">
                <i class="fas fa-times-circle"></i>
            </div>
            <h3>Analysis Error</h3>
            <p>${message}</p>
            <button onclick="clearResults()" class="retry-btn">
                <i class="fas fa-refresh"></i>
                Try Again
            </button>
        </div>
    `;
    
    // Add error styles
    const style = document.createElement('style');
    style.textContent = `
        .error-state {
            text-align: center;
            width: 100%;
            color: #ef4444;
        }
        
        .error-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .error-state h3 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .error-state p {
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        
        .retry-btn {
            background: linear-gradient(45deg, #ef4444, #fca5a5);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: transform 0.2s ease;
        }
        
        .retry-btn:hover {
            transform: translateY(-1px);
        }
    `;
    document.head.appendChild(style);
}

// Load spam example
function loadSpamExample() {
    document.getElementById('subject').value = 'URGENT! Claim Your $1000 Cash Prize NOW!';
    document.getElementById('message').value = `Congratulations! You've won $1000 CASH! Click here immediately to claim your prize before it expires. This is a limited time offer - don't miss out! 

Free money is waiting for you! No purchase necessary. Call now: 1-800-WINNER or visit our website to claim your guaranteed cash prize!

URGENT ACTION REQUIRED! Offer expires in 24 hours!`;
}

// Load ham example
function loadHamExample() {
    document.getElementById('subject').value = 'Project Meeting Tomorrow at 2 PM';
    document.getElementById('message').value = `Hi team,

I hope this email finds you well. I wanted to remind everyone about our project meeting scheduled for tomorrow at 2 PM in the conference room.

We'll be discussing the current project status, upcoming milestones, and addressing any questions or concerns you might have.

Please bring your laptops and any relevant documents. If you can't attend, please let me know in advance.

Looking forward to seeing everyone there.

Best regards,
Project Manager`;
}

// Navigation handling
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update active nav link
                    navLinks.forEach(l => l.classList.remove('active'));
                    this.classList.add('active');
                }
            }
        });
    });
}

// Scroll to analyzer
function scrollToAnalyzer() {
    document.getElementById('analyzer').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Setup animations
function setupAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.feature-card, .stat-card, .step');
    animateElements.forEach(el => observer.observe(el));
}

// Utility function to format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Add smooth scrolling for navbar on scroll
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 4px 6px -1px rgb(0 0 0 / 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!isAnalyzing) {
            const form = document.getElementById('spam-form');
            if (form) {
                form.dispatchEvent(new Event('submit'));
            }
        }
    }
    
    // Escape to clear form
    if (e.key === 'Escape') {
        if (!isAnalyzing) {
            document.getElementById('subject').value = '';
            document.getElementById('message').value = '';
            clearResults();
        }
    }
});

// Export functions for global access
window.scrollToAnalyzer = scrollToAnalyzer;
window.loadSpamExample = loadSpamExample;
window.loadHamExample = loadHamExample;
window.clearResults = clearResults;
