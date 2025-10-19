// Global state
let currentChatId = null;
let messages = [];
let isLoading = false;
let planningMode = false;
let pendingPlan = null; // Track the last plan waiting for approval

console.log('🎯 Requiem UI Script Loaded - Version 15.28.0 - Railway Backend Integration');
console.log('🔗 Environment Variable VITE_API_BASE_URL:', import.meta.env.VITE_API_BASE_URL);
console.log('🔗 All Environment Variables:', import.meta.env);

// Test Chart.js availability
if (typeof Chart !== 'undefined') {
    console.log('✅ Chart.js is loaded and available');
} else {
    console.error('❌ Chart.js is NOT loaded');
}

// DOM elements
const sidebarToggle = document.getElementById('sidebarToggle');
const newChatBtn = document.getElementById('newChatBtn');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeSection = document.querySelector('.welcome-section');
const loadingOverlay = document.getElementById('loadingOverlay');
const chatHistory = document.getElementById('chatHistory');
const planningModeBtn = document.getElementById('planningModeBtn');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeInputHandling();
    loadChatHistory();
    loadToolsFromStorage(); // Initialize tools data
    loadToolsFromAPI(); // Preload tools from API
});

// Event Listeners
function initializeEventListeners() {
    // Sidebar toggle
    sidebarToggle.addEventListener('click', toggleSidebar);
    
    // New chat
    newChatBtn.addEventListener('click', startNewChat);
    
    // Send message
    sendBtn.addEventListener('click', sendMessage);
    
    // Planning mode toggle
    planningModeBtn.addEventListener('click', togglePlanningMode);
    
    // Quick action cards
    document.querySelectorAll('.action-card').forEach(card => {
        card.addEventListener('click', function() {
            const prompt = this.getAttribute('data-prompt');
            if (prompt) {
                messageInput.value = prompt;
                sendMessage();
            }
        });
    });
    
    // Chat history items
    document.querySelectorAll('.chat-item').forEach(item => {
        item.addEventListener('click', function() {
            selectChatItem(this);
        });
    });
}

// Input handling
function initializeInputHandling() {
    console.log('🔧 initializeInputHandling called');
    console.log('📝 messageInput element:', messageInput);
    
    // Set initial height to match the CSS
    messageInput.style.height = '56px';
    
    messageInput.addEventListener('input', function() {
        // Auto-resize textarea (ChatGPT style)
        this.style.height = 'auto';
        this.style.height = Math.max(56, Math.min(this.scrollHeight, 200)) + 'px';
        
        // Enable/disable send button
        sendBtn.disabled = this.value.trim() === '' || isLoading;
        
        // Handle ticker suggestions
        handleTickerSuggestions(this.value, this);
    });
    
    // Focus management
    messageInput.addEventListener('focus', function() {
        this.parentElement.style.transform = 'scale(1.02)';
        this.parentElement.style.transition = 'transform 0.2s ease';
    });
    
    messageInput.addEventListener('blur', function() {
        this.parentElement.style.transform = 'scale(1)';
    });
    
    console.log('🎯 Attaching keydown listener to messageInput...');
    
    messageInput.addEventListener('keydown', function(e) {
        console.log('🔑 Key pressed:', e.key, 'Shift:', e.shiftKey, 'Ticker suggestions:', tickerSuggestions.length, 'isLoading:', isLoading);
        
        // Handle ticker suggestions first if they're visible
        if (tickerSuggestions.length > 0) {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                navigateTickerSuggestions(1);
                return;
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                navigateTickerSuggestions(-1);
                return;
            } else if (e.key === 'Enter' && selectedTickerIndex >= 0) {
                e.preventDefault();
                selectTickerSuggestion();
                return;
            } else if (e.key === 'Escape') {
                hideTickerSuggestions();
                return;
            }
        }
        
        // Handle Enter key for sending messages
        if (e.key === 'Enter' && !e.shiftKey) {
            console.log('✅ Enter pressed! isLoading:', isLoading, 'hasText:', messageInput.value.trim() !== '');
            e.preventDefault();
            if (!isLoading && messageInput.value.trim() !== '') {
                console.log('🚀 Calling sendMessage()...');
                sendMessage();
            } else {
                console.log('❌ Cannot send - isLoading:', isLoading, 'empty:', messageInput.value.trim() === '');
            }
        }
    });
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.ticker-suggestions') && !e.target.closest('.message-input')) {
            hideTickerSuggestions();
        }
    });
    
    // Focus input on load
    messageInput.focus();
}

// Sidebar functions
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('open');
}

// Chat management
function startNewChat() {
    currentChatId = null;
    messages = [];
    showWelcomeScreen();
    messageInput.focus();
}

function selectChatItem(item) {
    // Remove active class from all items
    document.querySelectorAll('.chat-item').forEach(i => i.classList.remove('active'));
    
    // Add active class to selected item
    item.classList.add('active');
    
    // Load chat (placeholder - in real app, this would load from storage)
    loadChat(item);
}

function loadChat(chatItem) {
    const chatTitle = chatItem.querySelector('.chat-title').textContent;
    
    // Hide welcome screen
    welcomeSection.style.display = 'none';
    messagesContainer.style.display = 'block';
    
    // Clear messages
    messagesContainer.innerHTML = '';
    
    // Add sample messages for demo
    if (chatTitle === 'SPY Momentum Strategy') {
        addSampleMessages();
    } else {
        addMessage('assistant', 'Chat history loading... (This would load the actual chat from storage)');
    }
    
    messageInput.focus();
}

function addSampleMessages() {
    // User message
    addMessage('user', 'Backtest a 12-month momentum strategy on SPY with monthly rebalancing');
    
    // Assistant response with strategy results
    setTimeout(() => {
        addStrategyResult();
    }, 1000);
}

// Message handling
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;
    
    // Hide welcome screen if showing
    if (welcomeSection.style.display !== 'none') {
        welcomeSection.style.display = 'none';
        messagesContainer.style.display = 'block';
    }
    
    // Add user message
    addMessage('user', message);
    
    // Check if this is a response to a pending plan
    if (pendingPlan && isApprovalResponse(message)) {
        handlePlanApproval(message);
        return;
    }
    
    // Show loading with initial stage
    showLoading(true, 'Processing query...');
    updateLoadingStage('thinking');
    
    try {
        // Stage 1: Understanding query
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Call Requiem API
        const response = planningMode 
            ? await callRequiemPlanningAPI(message)
            : await callRequiemAPI(message);
        
        console.log('📡 Full API Response:', response);
        
        // Add assistant response based on intent
        console.log('Response success status:', response.success);
        console.log('Response intent:', response.intent);
        
        // Handle statistical analysis, technical analysis, risk metrics, and mathematical calculations regardless of success status
        if (response.intent === 'statistical_analysis' || response.intent === 'analysis' || 
            response.intent === 'risk_metrics' || response.intent === 'mathematical_calculation') {
            console.log('🎯 Calling addStatisticalAnalysisResult with data:', response.data);
            addStatisticalAnalysisResult(response.data, response.message);
            return; // Exit early to avoid other processing
        }
        
    if (response.success) {
        console.log('Processing response with intent:', response.intent);
        if (response.intent === 'backtest') {
            addStrategyResult(response.data);
        } else if (response.intent === 'tool_backtest') {
            console.log('Calling addToolBacktestResult');
            addToolBacktestResult(response.data, response.message);
        } else if (response.intent === 'price_query') {
            addPriceQueryResult(response.data, response.message);
        } else if (response.intent === 'tool_execution') {
            addToolExecutionResult(response.data, response.message);
        } else if (response.intent === 'technical_analysis') {
            // Handle TA-Lib technical analysis responses with card format
            if (response.data && response.data.card_format) {
                addStatisticalAnalysisResult(response.data, response.message);
            } else {
                addToolExecutionResult(response.data, response.message);
            }
        } else if (response.intent === 'valuation') {
            addValuationResult(response.data, response.message);
        } else if (response.intent === 'intelligent_analysis') {
            addIntelligentAnalysisResult(response.data, response.message);
        } else if (response.intent === 'planning') {
            addPlanDisplay(response.data, response.message, message);
        } else if (response.intent === 'comparison') {
            addMessage('assistant', response.message);
        } else if (response.intent === 'analysis') {
            // Check if this is a tool execution result (must have success=true or actual data)
            if (response.data && response.data.tool_name && response.data.success) {
                addToolExecutionResult(response.data, response.message);
            } else {
                // Tool not available/selected, or general analysis message
                addMessage('assistant', response.message);
            }
        } else {
            addMessage('assistant', response.message || 'Result received');
        }
    } else {
            addMessage('assistant', response.message || `Sorry, I encountered an error: ${response.error}`);
        }
    } catch (error) {
        console.error('Error calling API:', error);
        console.error('Error details:', error.message);
        addMessage('assistant', `Sorry, I encountered an error while processing your request: ${error.message}. Please try again.`);
    } finally {
        showLoading(false);
        messageInput.focus();
    }
}

function addPriceQueryResult(data, message) {
    const time = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    let priceTable = '';
    if (data.latest_price) {
        // Check if this is fallback data
        const isFallback = data.is_fallback || false;
        const dateInfo = isFallback 
            ? `📅 Requested: ${data.requested_date}, Showing: ${data.actual_date}`
            : `📅 ${data.count} data points from ${data.start || data.actual_date} to ${data.end || data.actual_date}`;
        
        priceTable = `
            <div class="price-result ${isFallback ? 'fallback-data' : ''}">
                <h3>💰 ${data.ticker} Price Data</h3>
                <p>${message}</p>
                <div class="price-details">
                    <div class="price-item">
                        <span class="price-label">Open:</span>
                        <span class="price-value">$${data.latest_price.open.toFixed(2)}</span>
                    </div>
                    <div class="price-item">
                        <span class="price-label">High:</span>
                        <span class="price-value">$${data.latest_price.high.toFixed(2)}</span>
                    </div>
                    <div class="price-item">
                        <span class="price-label">Low:</span>
                        <span class="price-value">$${data.latest_price.low.toFixed(2)}</span>
                    </div>
                    <div class="price-item">
                        <span class="price-label">Close:</span>
                        <span class="price-value price-close">$${data.latest_price.close.toFixed(2)}</span>
                    </div>
                    <div class="price-item">
                        <span class="price-label">Volume:</span>
                        <span class="price-value">${data.latest_price.volume.toLocaleString()}</span>
                    </div>
                </div>
                <div class="price-meta">
                    <small>${dateInfo}</small>
                </div>
            </div>
        `;
    } else {
        priceTable = `<p>${message}</p>`;
    }
    
    messageDiv.innerHTML = `
        ${priceTable}
        <div class="message-time">${time}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    // Store message
    messages.push({ 
        sender: 'assistant', 
        text: message,
        timestamp: time,
        data: data
    });
}

function addValuationResult(data, message) {
    const time = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    if (!data || !data.symbol) {
        // Fallback to simple message
        addMessage('assistant', message);
        return;
    }
    
    const assessment = data.assessment || {};
    const metrics = data.valuation_metrics || {};
    
    // Debug current price
    console.log('Valuation data:', data);
    console.log('Current price value:', data.current_price, 'Type:', typeof data.current_price);
    
    const currentPrice = data.current_price ? `$${data.current_price.toFixed(2)}` : 'N/A';
    
    // Determine rating color
    let ratingClass = 'neutral';
    if (assessment.overall_rating === 'Undervalued') ratingClass = 'positive';
    else if (assessment.overall_rating === 'Overvalued') ratingClass = 'negative';
    
    const valuationHTML = `
        <div class="valuation-result">
            <div class="valuation-header">
                <h3>${data.symbol} Valuation Analysis</h3>
                <div class="valuation-price">Current Price: ${currentPrice}</div>
            </div>
            
            <div class="valuation-assessment ${ratingClass}">
                <div class="assessment-rating">
                    <span class="rating-label">Overall Assessment:</span>
                    <span class="rating-value">${assessment.overall_rating || 'Unknown'}</span>
                </div>
                <div class="assessment-confidence">
                    Confidence: ${assessment.confidence || 'N/A'}
                </div>
            </div>
            
            ${assessment.reasoning ? `
            <div class="valuation-reasoning">
                <strong>Reasoning:</strong> ${escapeHtml(assessment.reasoning)}
            </div>
            ` : ''}
            
            ${Object.keys(metrics).length > 0 ? `
            <div class="valuation-metrics">
                <h4>Key Valuation Metrics</h4>
                <div class="metrics-grid">
                    ${metrics.pe_ratio ? `
                    <div class="metric-card">
                        <div class="metric-label">P/E Ratio</div>
                        <div class="metric-value">${metrics.pe_ratio.toFixed(2)}</div>
                        <div class="metric-assessment">${metrics.pe_assessment?.assessment || ''}</div>
                    </div>
                    ` : ''}
                    ${metrics.pb_ratio ? `
                    <div class="metric-card">
                        <div class="metric-label">P/B Ratio</div>
                        <div class="metric-value">${metrics.pb_ratio.toFixed(2)}</div>
                        <div class="metric-assessment">${metrics.pb_assessment?.assessment || ''}</div>
                    </div>
                    ` : ''}
                    ${metrics.ps_ratio ? `
                    <div class="metric-card">
                        <div class="metric-label">P/S Ratio</div>
                        <div class="metric-value">${metrics.ps_ratio.toFixed(2)}</div>
                        <div class="metric-assessment">${metrics.ps_assessment?.assessment || ''}</div>
                    </div>
                    ` : ''}
                    ${metrics.peg_ratio ? `
                    <div class="metric-card">
                        <div class="metric-label">PEG Ratio</div>
                        <div class="metric-value">${metrics.peg_ratio.toFixed(2)}</div>
                        <div class="metric-assessment">${metrics.peg_assessment?.assessment || ''}</div>
                    </div>
                    ` : ''}
                    ${metrics.dividend_yield ? `
                    <div class="metric-card">
                        <div class="metric-label">Dividend Yield</div>
                        <div class="metric-value">${(metrics.dividend_yield * 100).toFixed(2)}%</div>
                        <div class="metric-assessment">${metrics.dividend_assessment?.assessment || ''}</div>
                    </div>
                    ` : ''}
                    ${metrics.market_cap ? `
                    <div class="metric-card">
                        <div class="metric-label">Market Cap</div>
                        <div class="metric-value">${formatMarketCap(metrics.market_cap)}</div>
                        <div class="metric-assessment">${metrics.market_cap_category || ''}</div>
                    </div>
                    ` : ''}
                </div>
            </div>
            ` : ''}
            
            ${data.data_sources ? `
            <div class="valuation-sources">
                <small>Data sources: ${Object.entries(data.data_sources).filter(([k, v]) => v).map(([k]) => k.replace('_', ' ')).join(', ')}</small>
            </div>
            ` : ''}
        </div>
    `;
    
    messageDiv.innerHTML = `${valuationHTML}<div class="message-time">${time}</div>`;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    messages.push({ 
        sender: 'assistant', 
        text: message,
        timestamp: time,
        data: data
    });
}

function formatMarketCap(value) {
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toFixed(0)}`;
}

function addPlanDisplay(data, message, originalQuery) {
    const time = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    if (!data || !data.plan_markdown) {
        addMessage('assistant', message);
        return;
    }
    
    // Store the pending plan for approval
    pendingPlan = {
        planId: data.plan_id,
        originalQuery: originalQuery,
        data: data
    };
    
    const planHTML = `
        <div class="plan-result">
            <div class="plan-header">
                <h3>📋 Execution Plan</h3>
                <div class="plan-id">Plan ID: ${data.plan_id}</div>
            </div>
            
            <div class="plan-content">
                <div class="plan-markdown">
                    ${formatPlanMarkdown(data.plan_markdown)}
                </div>
            </div>
        </div>
    `;
    
    messageDiv.innerHTML = `${planHTML}<div class="message-time">${time}</div>`;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    messages.push({ 
        sender: 'assistant', 
        text: message,
        timestamp: time,
        data: data
    });
}

function formatPlanMarkdown(markdown) {
    // Convert markdown to HTML for display
    return markdown
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^\*\*([^*]+)\*\*:/gim, '<strong>$1:</strong>')
        .replace(/\*\*([^*]+)\*\*/gim, '<strong>$1</strong>')
        .replace(/^(\d+\) .*$)/gim, '<div class="plan-step">$1</div>')
        .replace(/^   Tool\/Function: (.*$)/gim, '<div class="plan-tool">Tool/Function: <code>$1</code></div>')
        .replace(/^   Inputs: (.*$)/gim, '<div class="plan-inputs">Inputs: <code>$1</code></div>')
        .replace(/^   Output: (.*$)/gim, '<div class="plan-output">Output: <code>$1</code></div>')
        .replace(/^   Why: (.*$)/gim, '<div class="plan-reasoning">Why: <em>$1</em></div>')
        .replace(/\n/g, '<br>');
}

function isApprovalResponse(message) {
    const lower = message.toLowerCase().trim();
    
    // Approval patterns
    const approvalPatterns = [
        /^yes$/i, /^yeah$/i, /^yep$/i, /^sure$/i, /^ok$/i, /^okay$/i,
        /^yes please$/i, /^yes, please$/i, /^go ahead$/i, /^proceed$/i,
        /^do it$/i, /^run it$/i, /^execute$/i, /^approve$/i, /^approved$/i,
        /^looks good$/i, /^sounds good$/i, /^perfect$/i
    ];
    
    // Cancellation patterns
    const cancellationPatterns = [
        /^no$/i, /^nope$/i, /^nah$/i, /^cancel$/i, /^stop$/i, /^abort$/i,
        /^don't$/i, /^dont$/i, /^nevermind$/i, /^never mind$/i
    ];
    
    // Check for approval
    for (const pattern of approvalPatterns) {
        if (pattern.test(lower)) {
            return 'approve';
        }
    }
    
    // Check for cancellation
    for (const pattern of cancellationPatterns) {
        if (pattern.test(lower)) {
            return 'cancel';
        }
    }
    
    return null;
}

async function handlePlanApproval(message) {
    const action = isApprovalResponse(message);
    
    if (action === 'approve') {
        // User approved the plan - execute it
        showLoading(true, 'Executing approved plan...');
        
        try {
            const originalQuery = pendingPlan.originalQuery;
            const planData = pendingPlan.data;
            pendingPlan = null; // Clear pending plan
            
            // Check if this is a compound query
            if (planData && planData.steps && planData.steps.length > 0) {
                // For compound queries, we need to execute multiple intents
                // Parse the query to detect all intents
                const query = originalQuery.toLowerCase();
                const hasValuation = query.includes('overvalued') || query.includes('undervalued') || query.includes('valuation');
                const hasTechnical = query.includes('technical') || query.includes('analyze') || query.includes('analysis');
                
                if (hasValuation && hasTechnical) {
                    // Execute both valuation and technical analysis
                    addMessage('assistant', 'Executing comprehensive analysis (valuation + technical)...');
                    
                    // First: Valuation
                    const valuationResponse = await callRequiemAPI(originalQuery.split('?')[0] + '?');
                    if (valuationResponse.success && valuationResponse.intent === 'valuation') {
                        addValuationResult(valuationResponse.data, valuationResponse.message);
                    }
                    
                    // Second: Technical Analysis - execute each indicator
                    const ticker = originalQuery.match(/\$([A-Z]+)/)?.[1] || 'SPY';
                    
                    // RSI
                    const rsiResponse = await callRequiemAPI(`calculate rsi for ${ticker} over last 6 months`);
                    if (rsiResponse.success && rsiResponse.data && rsiResponse.data.tool_name) {
                        addToolExecutionResult(rsiResponse.data, rsiResponse.message);
                    } else {
                        addMessage('assistant', `RSI analysis failed: ${rsiResponse.message || 'Unknown error'}`);
                    }
                    
                    // MACD
                    const macdResponse = await callRequiemAPI(`calculate macd for ${ticker} over last 6 months`);
                    if (macdResponse.success && macdResponse.data && macdResponse.data.tool_name) {
                        addToolExecutionResult(macdResponse.data, macdResponse.message);
                    } else {
                        addMessage('assistant', `MACD analysis failed: ${macdResponse.message || 'Unknown error'}`);
                    }
                    
                    // SMA
                    const smaResponse = await callRequiemAPI(`calculate sma 50 for ${ticker} over last 6 months`);
                    if (smaResponse.success && smaResponse.data && smaResponse.data.tool_name) {
                        addToolExecutionResult(smaResponse.data, smaResponse.message);
                    } else {
                        addMessage('assistant', `SMA analysis failed: ${smaResponse.message || 'Unknown error'}`);
                    }
                    
                } else {
                    // Single intent - execute normally
                    const response = await callRequiemAPI(originalQuery);
                    handleSingleIntentResponse(response);
                }
            } else {
                // No plan data - execute as single query
                const response = await callRequiemAPI(originalQuery);
                handleSingleIntentResponse(response);
            }
            
        } catch (error) {
            console.error('Error executing plan:', error);
            addMessage('assistant', `Error executing plan: ${error.message}`);
        } finally {
            showLoading(false);
            messageInput.focus();
        }
        
    } else if (action === 'cancel') {
        // User cancelled the plan
        pendingPlan = null;
        addMessage('assistant', 'Plan cancelled. No execution will occur.');
        messageInput.focus();
    }
}

function handleSingleIntentResponse(response) {
    if (response.success) {
        console.log('handleSingleIntentResponse - Processing response with intent:', response.intent);
        if (response.intent === 'backtest') {
            addStrategyResult(response.data);
        } else if (response.intent === 'tool_backtest') {
            console.log('handleSingleIntentResponse - Calling addToolBacktestResult');
            addToolBacktestResult(response.data, response.message);
        } else if (response.intent === 'price_query') {
            addPriceQueryResult(response.data, response.message);
        } else if (response.intent === 'valuation') {
            addValuationResult(response.data, response.message);
        } else if (response.intent === 'analysis') {
            if (response.data && response.data.tool_name && response.data.success) {
                addToolExecutionResult(response.data, response.message);
            } else {
                addMessage('assistant', response.message);
            }
        } else {
            addMessage('assistant', response.message || 'Plan executed successfully!');
        }
    } else {
        addMessage('assistant', response.message || `Error: ${response.error}`);
    }
}

function togglePlanningMode() {
    planningMode = !planningMode;
    planningModeBtn.classList.toggle('active', planningMode);
    
    if (planningMode) {
        messageInput.placeholder = "Ask Requiem (Planning Mode) - I'll show you my plan first...";
        addMessage('assistant', '📋 Planning Mode enabled! I will now show you detailed execution plans before running any code or tools.');
    } else {
        messageInput.placeholder = "Ask Requiem...";
        addMessage('assistant', '⚡ Direct Mode enabled! I will execute queries immediately.');
    }
}

function addToolExecutionResult(data, message) {
    const time = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    // Check if data has the required fields
    if (!data || !data.tool_name) {
        // Fallback to simple message display
        messageDiv.innerHTML = `
            <div class="message-bubble">
                <p>${escapeHtml(message)}</p>
            </div>
            <div class="message-time">${time}</div>
        `;
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
        messages.push({ sender: 'assistant', text: message, timestamp: time });
        return;
    }
    
    // Format values safely
    const latestValue = data.latest_value !== null && data.latest_value !== undefined 
        ? (typeof data.latest_value === 'number' ? data.latest_value.toFixed(4) : data.latest_value)
        : 'N/A';
    
    const meanValue = data.mean_value !== null && data.mean_value !== undefined
        ? (typeof data.mean_value === 'number' ? data.mean_value.toFixed(4) : data.mean_value)
        : 'N/A';
    
    // Determine if we should show a chart (>= 10 data points)
    const shouldShowChart = data.series_data && data.data_points >= 10;
    const chartId = `tool-chart-${Date.now()}`;
    
    const toolResult = `
        <div class="tool-execution-result">
            <h3>🔧 ${data.tool_name.toUpperCase()} Analysis</h3>
            <div class="tool-details">
                <div class="tool-item">
                    <span class="tool-label">Ticker:</span>
                    <span class="tool-value">${data.ticker || 'N/A'}</span>
                </div>
                <div class="tool-item">
                    <span class="tool-label">Period:</span>
                    <span class="tool-value">${data.period || 'N/A'}</span>
                </div>
                <div class="tool-item">
                    <span class="tool-label">Latest Value:</span>
                    <span class="tool-value">${latestValue}</span>
                </div>
                <div class="tool-item">
                    <span class="tool-label">Mean Value:</span>
                    <span class="tool-value">${meanValue}</span>
                </div>
                ${data.min_value !== undefined && data.min_value !== null ? `
                <div class="tool-item">
                    <span class="tool-label">Min Value:</span>
                    <span class="tool-value">${typeof data.min_value === 'number' ? data.min_value.toFixed(4) : data.min_value}</span>
                </div>
                ` : ''}
                ${data.max_value !== undefined && data.max_value !== null ? `
                <div class="tool-item">
                    <span class="tool-label">Max Value:</span>
                    <span class="tool-value">${typeof data.max_value === 'number' ? data.max_value.toFixed(4) : data.max_value}</span>
                </div>
                ` : ''}
                <div class="tool-item">
                    <span class="tool-label">Data Points:</span>
                    <span class="tool-value">${data.data_points || 0}</span>
                </div>
                ${data.parameters_used ? `
                <div class="tool-item">
                    <span class="tool-label">Parameters:</span>
                    <span class="tool-value">${JSON.stringify(data.parameters_used)}</span>
                </div>
                ` : ''}
            </div>
            ${shouldShowChart ? `
            <div class="chart-container" style="margin-top: 16px; min-width: 400px;">
                <canvas id="${chartId}" style="max-width: 100%;"></canvas>
            </div>
            ` : ''}
            ${data.description ? `
            <div class="tool-description">
                <p><em>${escapeHtml(data.description)}</em></p>
            </div>
            ` : ''}
        </div>
    `;
    
    messageDiv.innerHTML = `${toolResult}<div class="message-time">${time}</div>`;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    // Create chart if we should show one
    if (shouldShowChart && data.series_data) {
        setTimeout(() => {
            createToolChart(chartId, data);
        }, 100);
    }
    
    messages.push({ sender: 'assistant', text: message, timestamp: time, data: data });
}

function createToolChart(chartId, data) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Choose color based on tool type
    const colorMap = {
        'rsi': { line: '#9333ea', bg: 'rgba(147, 51, 234, 0.1)' },  // Purple
        'sma': { line: '#3b82f6', bg: 'rgba(59, 130, 246, 0.1)' },  // Blue
        'macd': { line: '#10b981', bg: 'rgba(16, 185, 129, 0.1)' }, // Green
        'bollinger': { line: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)' }, // Orange
        'zscore': { line: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)' }, // Red
        'default': { line: '#6366f1', bg: 'rgba(99, 102, 241, 0.1)' }  // Indigo
    };
    
    const colors = colorMap[data.tool_name] || colorMap.default;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.series_data.dates,
            datasets: [{
                label: `${data.tool_name.toUpperCase()} - ${data.ticker}`,
                data: data.series_data.values,
                borderColor: colors.line,
                backgroundColor: colors.bg,
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            aspectRatio: 2.5,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15
                    }
                },
                title: {
                    display: true,
                    text: `${data.tool_name.toUpperCase()} Time Series`,
                    font: { size: 14, weight: 'bold' },
                    padding: 10
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: { 
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        padding: 8
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        maxTicksLimit: 12,
                        autoSkip: true,
                        padding: 8
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            elements: {
                point: {
                    radius: 0,
                    hoverRadius: 4
                },
                line: {
                    borderWidth: 2
                }
            }
        }
    });
}

function addMessage(sender, text, timestamp = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const time = timestamp || new Date();
    const timeStr = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            ${sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>'}
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
            <div class="message-time">${timeStr}</div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Store message
    messages.push({ sender, text, timestamp: time });
}

function addToolBacktestResult(data, message) {
    console.log('addToolBacktestResult called with:', { data, message });
    
    const messagesContainer = document.getElementById('messagesContainer');
    console.log('messagesContainer:', messagesContainer);
    
    if (!messagesContainer) {
        console.error('Messages container not found');
        // Fallback to simple message
        addMessage('assistant', `🔧 Tool-Based Backtest: ${message}`);
        return;
    }
    
    try {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        let content = `
            <div class="message-bubble">
                <div class="tool-backtest-result">
                    <div class="tool-backtest-header">
                        <h3>🔧 Tool-Based Backtest Results</h3>
                        <p class="tool-backtest-message">${message}</p>
                    </div>
        `;
        
        if (data.signal_rules && data.signal_rules.length > 0) {
            content += `
                <div class="signal-rules-section">
                    <h4>Signal Rules Detected:</h4>
                    <ul class="signal-rules-list">
            `;
            
            data.signal_rules.forEach(rule => {
                content += `
                    <li class="signal-rule">
                        <span class="rule-action">${rule.action.toUpperCase()}</span> when 
                        <span class="rule-tool">${rule.tool_name}</span> 
                        <span class="rule-comparison">${rule.comparison === 'less_than' ? '<' : '>'}</span> 
                        <span class="rule-threshold">${rule.threshold}</span>
                    </li>
                `;
            });
            
            content += `
                    </ul>
                </div>
            `;
        }
        
        if (data.ready_to_execute) {
            content += `
                <div class="execution-section">
                    <p class="execution-status">✅ Ready to execute backtest</p>
                    <p class="execution-note">The tool-based signals have been generated and are ready for backtest execution.</p>
                </div>
            `;
        }
        
        content += `
                </div>
            </div>
        `;
        
        messageDiv.innerHTML = content;
        console.log('About to append messageDiv to messagesContainer');
        messagesContainer.appendChild(messageDiv);
        console.log('Successfully appended messageDiv');
        
        // Use the global scrollToBottom function
        if (typeof scrollToBottom === 'function') {
            scrollToBottom();
        } else {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
    } catch (error) {
        console.error('Error in addToolBacktestResult:', error);
        // Fallback to simple message
        addMessage('assistant', `🔧 Tool-Based Backtest: ${message}`);
    }
}

function addStrategyResult(tearsheet = null) {
    // Use real tearsheet data if provided, otherwise sample data
    let data;
    if (tearsheet && tearsheet.summary && tearsheet.metrics) {
        data = {
            title: tearsheet.summary.title || "Strategy Results",
            period: tearsheet.summary.period ? 
                `${tearsheet.summary.period.start} to ${tearsheet.summary.period.end}` : 
                "Period not specified",
            actual_period: tearsheet.summary.actual_period ? 
                `${tearsheet.summary.actual_period.start} to ${tearsheet.summary.actual_period.end}` : 
                null,
            config: tearsheet.summary.config || {},
            metrics: tearsheet.metrics.performance || {}
        };
    } else {
        // Fallback sample data
        data = {
            title: "SPY Momentum Strategy (12-month, Monthly Rebalancing)",
            period: "2023-01-01 to 2023-12-31",
            actual_period: null,
            config: {},
            metrics: {
                cagr: 0.2236,
                sharpe: 1.5502,
                sortino: 2.7636,
                hit_rate: 0.5440,
                vol_annual: 0.1307,
                max_drawdown_pct: -0.1008,
                avg_turnover: 8.0964,
                exposure_share: 0.9840
            }
        };
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const time = new Date();
    const timeStr = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="strategy-result">
                    <div class="result-header">
                        <div class="result-title">${escapeHtml(data.title)}</div>
                        <div class="result-actions">
                            <button class="action-btn primary" onclick="downloadResults()">
                                <i class="fas fa-download"></i> Download
                            </button>
                            <button class="action-btn" onclick="shareResults()">
                                <i class="fas fa-share"></i> Share
                            </button>
                        </div>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-value">${data.metrics.cagr ? (data.metrics.cagr * 100).toFixed(1) + '%' : 'N/A'}</div>
                            <div class="metric-label">CAGR</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">${data.metrics.sharpe ? data.metrics.sharpe.toFixed(2) : 'N/A'}</div>
                            <div class="metric-label">Sharpe</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">${data.metrics.sortino ? data.metrics.sortino.toFixed(2) : 'N/A'}</div>
                            <div class="metric-label">Sortino</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">${data.metrics.hit_rate ? (data.metrics.hit_rate * 100).toFixed(1) + '%' : 'N/A'}</div>
                            <div class="metric-label">Hit Rate</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">${data.metrics.vol_annual ? (data.metrics.vol_annual * 100).toFixed(1) + '%' : 'N/A'}</div>
                            <div class="metric-label">Volatility</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">${data.metrics.max_drawdown_pct ? (data.metrics.max_drawdown_pct * 100).toFixed(1) + '%' : 'N/A'}</div>
                            <div class="metric-label">Max DD</div>
                        </div>
                    </div>
                    
                    <div class="charts-grid">
                        <div class="chart-container">
                            <canvas id="equityChart"></canvas>
                        </div>
                        <div class="chart-container">
                            <canvas id="drawdownChart"></canvas>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="rollingSharpeChart"></canvas>
                    </div>
                    
                    <div class="result-footer">
                        <small style="color: #666;">
                            Period: ${data.period}${data.actual_period && data.actual_period !== data.period ? ` (Actual: ${data.actual_period})` : ''} | 
                            Generated: ${time.toLocaleString()}
                        </small>
                    </div>
                </div>
            </div>
            <div class="message-time">${timeStr}</div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Create charts after the DOM element is added
    setTimeout(() => {
        createCharts(tearsheet);
    }, 100);
    
    // Store message
    messages.push({ 
        sender: 'assistant', 
        text: `Strategy Results: ${data.title}`, 
        timestamp: time,
        data: data
    });
}

// Helper function to detect intelligent analysis queries
function _isIntelligentQuery(prompt) {
    const promptLower = prompt.toLowerCase();
    
    // Exclude tool-based backtest queries
    const toolBacktestPatterns = [
        'backtest', 'tool', 'buy when', 'sell when', 'long when', 'short when'
    ];
    
    // If it contains tool backtest patterns, it's not an intelligent query
    if (toolBacktestPatterns.some(pattern => promptLower.includes(pattern))) {
        return false;
    }
    
    const intelligentKeywords = [
        'entry', 'enter', 'buy', 'when to buy', 'good price', 'should i buy',
        'buying opportunity', 'entry point', 'timing', 'what price',
        'technical indicators', 'relevant', 'decide', 'recommendation',
        'what technical indicators', 'which indicators', 'help me decide',
        'earnings', 'earnings call', 'quarterly', 'q1', 'q2', 'q3', 'q4',
        'transcript', 'conference call', 'investor call', 'results',
        'quarterly results', 'annual results', 'guidance', 'outlook',
        'revenue guidance', 'eps guidance', 'earnings report'
    ];
    
    // Check for technical indicator queries
    const technicalIndicators = [
        'rsi', 'sma', 'ema', 'macd', 'bollinger', 'stochastic', 'williams',
        'cci', 'adx', 'aroon', 'obv', 'atr', 'mfi', 'roc', 'momentum',
        'moving average', 'relative strength', 'commodity channel',
        'directional movement', 'on balance volume', 'money flow',
        'average true range', 'true range', 'simple moving average',
        'exponential moving average', 'relative strength index',
        'bollinger bands', 'williams %r', 'williams r', 'commodity channel index',
        'average directional movement index', 'on balance volume',
        'money flow index', 'rate of change', 'chande momentum oscillator',
        'ultimate oscillator', 'percentage price oscillator', 'trix',
        'balance of power', 'aroon oscillator', 'accumulation distribution',
        'hilbert transform', 'minus directional', 'plus directional'
    ];
    
    const hasIntelligentKeywords = intelligentKeywords.some(keyword => promptLower.includes(keyword));
    const hasTechnicalIndicators = technicalIndicators.some(indicator => promptLower.includes(indicator));
    
    // Debug logging for RSI queries
    if (promptLower.includes('rsi')) {
        console.log('🔍 _isIntelligentQuery Debug:', {
            prompt: prompt,
            promptLower: promptLower,
            hasIntelligentKeywords: hasIntelligentKeywords,
            hasTechnicalIndicators: hasTechnicalIndicators,
            technicalIndicators: technicalIndicators.filter(indicator => promptLower.includes(indicator)),
            result: hasIntelligentKeywords || hasTechnicalIndicators
        });
    }
    
    return hasIntelligentKeywords || hasTechnicalIndicators;
}

// API integration
async function callRequiemAPI(prompt) {
    try {
        // Stage 1: Understanding query
        updateLoadingStage('thinking', 'Understanding your query...');
        
        // Get selected tools from user's settings
        const selectedTools = getSelectedTools();
        
        // Check if this is an intelligent analysis query
        const isIntelligentQuery = _isIntelligentQuery(prompt);
        const endpoint = isIntelligentQuery ? '/query/intelligent' : '/query';
        
        console.log('🔍 Query analysis:', {
            prompt: prompt,
            isIntelligentQuery: isIntelligentQuery,
            endpoint: endpoint
        });
        
        // Additional debug for RSI queries
        if (prompt.toLowerCase().includes('rsi')) {
            console.log('🔍 RSI Query Debug:', {
                prompt: prompt,
                promptLower: prompt.toLowerCase(),
                hasRsi: prompt.toLowerCase().includes('rsi'),
                isIntelligentQuery: isIntelligentQuery,
                endpoint: endpoint
            });
        }
        
        // Use the appropriate endpoint
        const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
        console.log('🔗 API Base URL:', apiBaseUrl);
        console.log('🔗 Full URL:', `${apiBaseUrl}${endpoint}`);
        const queryResponse = await fetch(`${apiBaseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                query: prompt,
                selected_tools: selectedTools
            })
        });
        
        if (!queryResponse.ok) {
            throw new Error(`Query failed: ${queryResponse.status}`);
        }
        
        const result = await queryResponse.json();
        
        // Stage 2: Planning based on intent
        updateLoadingStage('planning', getIntentMessage(result.intent));
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Route based on intent
        if (result.intent === 'backtest' && result.data.ready_to_execute) {
            // Stage 3: Executing backtest
            updateLoadingStage('executing', 'Running backtest simulation...');
            
            // Execute the backtest
            const executeResponse = await fetch(`${apiBaseUrl}/runs/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ spec: result.data.spec_skeleton })
            });
            
            if (!executeResponse.ok) {
                throw new Error(`Execution failed: ${executeResponse.status}`);
            }
            
            const executeResult = await executeResponse.json();
            
            // Fetch the tearsheet using the run_id
            const tearsheetResponse = await fetch(`${apiBaseUrl}/runs/${executeResult.run_id}/tearsheet`);
            
            if (!tearsheetResponse.ok) {
                throw new Error(`Failed to fetch tearsheet: ${tearsheetResponse.status}`);
            }
            
            const tearsheet = await tearsheetResponse.json();
            return { success: true, intent: 'backtest', data: tearsheet };
        } 
        else if (result.intent === 'price_query') {
            updateLoadingStage('executing', 'Fetching price data...');
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Price query - format the response
            return { 
                success: true, 
                intent: 'price_query', 
                data: result.data,
                message: result.message
            };
        }
        else if (result.intent === 'valuation') {
            updateLoadingStage('executing', 'Analyzing valuation...');
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Valuation query
            return { 
                success: true, 
                intent: 'valuation', 
                data: result.data,
                message: result.message
            };
        }
        else if (result.intent === 'comparison') {
            updateLoadingStage('executing', 'Comparing assets...');
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Comparison query
            return { 
                success: true, 
                intent: 'comparison', 
                data: result.data,
                message: result.message
            };
        }
        else if (result.intent === 'analysis') {
            updateLoadingStage('executing', 'Running technical analysis...');
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Analysis query
            return { 
                success: true, 
                intent: 'analysis', 
                data: result.data,
                message: result.message
            };
        }
        else if (result.intent === 'intelligent_analysis') {
            updateLoadingStage('executing', 'Generating intelligent insights...');
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Intelligent analysis query
            return { 
                success: true, 
                intent: 'intelligent_analysis', 
                data: result.data,
                message: result.message
            };
        }
        else if (result.intent === 'technical_analysis') {
            updateLoadingStage('executing', 'Processing technical analysis...');
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Technical analysis query (TA-Lib comparison)
            return { 
                success: true, 
                intent: 'technical_analysis', 
                data: result.data,
                message: result.message
            };
        }
        else {
            // Check if this might be a tool execution request
            const selectedTools = getSelectedTools();
            if (selectedTools.length > 0) {
                updateLoadingStage('executing', 'Executing tool calculations...');
                
                const toolResponse = await fetch(`${apiBaseUrl}/tools/execute`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        query: prompt,
                        selected_tools: selectedTools
                    })
                });

                if (toolResponse.ok) {
                    const toolResult = await toolResponse.json();
                    if (toolResult.success) {
                        return { 
                            success: true, 
                            intent: 'tool_execution', 
                            data: toolResult.results,
                            message: toolResult.message
                        };
                    }
                }
            }
            
            // Unknown intent or not ready to execute
            return { 
                success: false, 
                intent: result.intent,
                message: result.message || 'Could not understand the query',
                data: result.data
            };
        }
    } catch (error) {
        console.error('API Error:', error);
        return { success: false, error: error.message };
    }
}

async function callRequiemPlanningAPI(prompt) {
    try {
        // Stage 1: Understanding query
        updateLoadingStage('thinking', 'Generating execution plan...');
        
        // Get selected tools from user's settings
        const selectedTools = getSelectedTools();
        
        // Use the planning endpoint
        const queryResponse = await fetch(`${apiBaseUrl}/query/plan`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                query: prompt,
                selected_tools: selectedTools
            })
        });
        
        if (!queryResponse.ok) {
            throw new Error(`HTTP error! status: ${queryResponse.status}`);
        }
        
        const result = await queryResponse.json();
        
        // Planning mode - return the plan directly
        return { 
            success: true, 
            intent: 'planning', 
            data: result.data,
            message: result.message
        };
        
    } catch (error) {
        console.error('Planning API Error:', error);
        return { success: false, error: error.message };
    }
}

// Ticker Suggestions
let tickerSuggestions = [];
let selectedTickerIndex = -1;
let tickerSearchTimeout = null;

const TICKER_DATABASE = {
    'AAPL': [
        { type: 'Q', country: 'US', ticker: 'AAPL', name: 'Apple Inc', fullType: 'Equity' },
        { type: 'E', country: 'US', ticker: 'AAPL', name: 'Apple Inc ETF', fullType: 'ETF' }
    ],
    'MSFT': [
        { type: 'Q', country: 'US', ticker: 'MSFT', name: 'Microsoft Corporation', fullType: 'Equity' }
    ],
    'GOOGL': [
        { type: 'Q', country: 'US', ticker: 'GOOGL', name: 'Alphabet Inc Class A', fullType: 'Equity' }
    ],
    'AMZN': [
        { type: 'Q', country: 'US', ticker: 'AMZN', name: 'Amazon.com Inc', fullType: 'Equity' }
    ],
    'NVDA': [
        { type: 'Q', country: 'US', ticker: 'NVDA', name: 'NVIDIA Corporation', fullType: 'Equity' },
        { type: 'E', country: 'US', ticker: 'NVD', name: 'GraniteShares 2x Short NVDA Daily E', fullType: 'ETF' },
        { type: 'E', country: 'US', ticker: 'NVDD', name: 'DIREXION DAILY NVDA BEAR 1X SHARES', fullType: 'ETF' },
        { type: 'Q', country: 'DEK', ticker: 'NVD.FRK', name: 'NVIDIA Corporation', fullType: 'Equity' },
        { type: 'M', country: 'US', ticker: 'NVDAX', name: 'Wells Fargo Diversified Eqo Fd USD', fullType: 'Mutual Fund' }
    ],
    'TSLA': [
        { type: 'Q', country: 'US', ticker: 'TSLA', name: 'Tesla Inc', fullType: 'Equity' }
    ],
    'META': [
        { type: 'Q', country: 'US', ticker: 'META', name: 'Meta Platforms Inc', fullType: 'Equity' }
    ],
    'SPY': [
        { type: 'E', country: 'US', ticker: 'SPY', name: 'SPDR S&P 500 ETF Trust', fullType: 'ETF' }
    ],
    'QQQ': [
        { type: 'E', country: 'US', ticker: 'QQQ', name: 'Invesco QQQ Trust', fullType: 'ETF' }
    ],
    'IWM': [
        { type: 'E', country: 'US', ticker: 'IWM', name: 'iShares Russell 2000 ETF', fullType: 'ETF' }
    ],
    'DIA': [
        { type: 'E', country: 'US', ticker: 'DIA', name: 'SPDR Dow Jones Industrial Average ETF', fullType: 'ETF' }
    ]
};

function handleTickerSuggestions(value, inputElement) {
    const tickerMatch = value.match(/\$([A-Za-z]*)$/);
    
    // Clear previous timeout
    if (tickerSearchTimeout) {
        clearTimeout(tickerSearchTimeout);
    }
    
    if (tickerMatch) {
        const tickerQuery = tickerMatch[1].toUpperCase();
        
        // Smart debouncing: faster for longer queries, instant for cache hits
        let delay = 150;
        if (tickerQuery.length >= 3) {
            delay = 100;  // Even faster for longer queries
        }
        
        tickerSearchTimeout = setTimeout(() => {
            if (tickerQuery.length >= 2) {
                showTickerSuggestions(tickerQuery, inputElement);
            } else if (tickerQuery.length === 0) {
                hideTickerSuggestions();
            }
        }, delay);
    } else {
        hideTickerSuggestions();
    }
}

async function showTickerSuggestions(tickerQuery, inputElement) {
    // Show loading state
    showTickerLoading(inputElement);
    
    // Get API base URL
    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    
    // Fetch suggestions from API
    try {
        const response = await fetch(`${apiBaseUrl}/ticker-suggestions?q=${encodeURIComponent(tickerQuery)}`);
        const data = await response.json();
        
        const matches = data.suggestions || [];
        
        if (matches.length === 0) {
            hideTickerSuggestions();
            return;
        }
        
        tickerSuggestions = matches;
        selectedTickerIndex = 0;
    
    // Create or update suggestions dropdown
    let dropdown = document.querySelector('.ticker-suggestions');
    if (!dropdown) {
        dropdown = document.createElement('div');
        dropdown.className = 'ticker-suggestions';
        document.body.appendChild(dropdown);
    }
    
    dropdown.innerHTML = matches.map((ticker, index) => `
        <div class="ticker-suggestion ${index === 0 ? 'selected' : ''}" data-index="${index}">
            <div class="ticker-format">${ticker.type}:${ticker.country}:${ticker.ticker}</div>
            <div class="ticker-details">
                <div class="ticker-name">${ticker.name}</div>
                <div class="ticker-info">${ticker.country} - ${ticker.fullType}</div>
            </div>
        </div>
    `).join('');
    
    // Position dropdown above input
    const inputRect = inputElement.getBoundingClientRect();
    dropdown.style.position = 'fixed';
    dropdown.style.left = `${inputRect.left}px`;
    dropdown.style.bottom = `${window.innerHeight - inputRect.top + 10}px`;
    dropdown.style.width = `${Math.max(inputRect.width, 400)}px`;
    dropdown.style.display = 'block';
    
    // Add click handlers
    dropdown.querySelectorAll('.ticker-suggestion').forEach((item, index) => {
        item.addEventListener('click', () => {
            selectedTickerIndex = index;
            selectTickerSuggestion();
        });
    });
    
    } catch (error) {
        console.error('Failed to fetch ticker suggestions:', error);
        hideTickerSuggestions();
    }
}

function showTickerLoading(inputElement) {
    // Create or update loading dropdown
    let dropdown = document.querySelector('.ticker-suggestions');
    if (!dropdown) {
        dropdown = document.createElement('div');
        dropdown.className = 'ticker-suggestions';
        document.body.appendChild(dropdown);
    }
    
    dropdown.innerHTML = `
        <div class="ticker-loading">
            <div class="loading-spinner"></div>
            <span>Searching tickers...</span>
        </div>
    `;
    
    // Position dropdown above input
    const inputRect = inputElement.getBoundingClientRect();
    dropdown.style.position = 'fixed';
    dropdown.style.left = `${inputRect.left}px`;
    dropdown.style.bottom = `${window.innerHeight - inputRect.top + 10}px`;
    dropdown.style.width = `${Math.max(inputRect.width, 400)}px`;
    dropdown.style.display = 'block';
}

function hideTickerSuggestions() {
    const dropdown = document.querySelector('.ticker-suggestions');
    if (dropdown) {
        dropdown.style.display = 'none';
    }
    tickerSuggestions = [];
    selectedTickerIndex = -1;
}

function navigateTickerSuggestions(direction) {
    if (tickerSuggestions.length === 0) return;
    
    selectedTickerIndex += direction;
    
    if (selectedTickerIndex < 0) {
        selectedTickerIndex = tickerSuggestions.length - 1;
    } else if (selectedTickerIndex >= tickerSuggestions.length) {
        selectedTickerIndex = 0;
    }
    
    // Update visual selection
    const dropdown = document.querySelector('.ticker-suggestions');
    if (dropdown) {
        dropdown.querySelectorAll('.ticker-suggestion').forEach((item, index) => {
            item.classList.toggle('selected', index === selectedTickerIndex);
        });
    }
}

function selectTickerSuggestion() {
    if (selectedTickerIndex < 0 || selectedTickerIndex >= tickerSuggestions.length) return;
    
    const selectedTicker = tickerSuggestions[selectedTickerIndex];
    const currentValue = messageInput.value;
    
    // Replace $tickerQuery with the full formatted ticker
    const newValue = currentValue.replace(/\$[A-Za-z]*$/, `$${selectedTicker.ticker}`);
    
    messageInput.value = newValue;
    hideTickerSuggestions();
    
    // Auto-resize input
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// Tools Management
let toolsData = {
    orchid: [
        { name: 'zscore', description: 'Z-Score normalization for mean reversion', selected: true },
        { name: 'realized_vol', description: 'Realized volatility calculation', selected: true },
        { name: 'momentum', description: '12-month momentum with 1-month skip', selected: true },
        { name: 'sma', description: 'Simple Moving Average (20 & 50 day)', selected: true },
        { name: 'rsi', description: 'Relative Strength Index (14-day)', selected: true },
        { name: 'aroon', description: 'Aroon indicator for trend detection', selected: true },
        { name: 'macd', description: 'MACD oscillator', selected: true },
        { name: 'bollinger', description: 'Bollinger Bands with squeeze detection', selected: true },
        { name: 'williams_r', description: 'Williams %R momentum oscillator', selected: true },
        { name: 'stochastic', description: 'Stochastic oscillator', selected: true }
    ],
    user: []
};

function getSelectedTools() {
    const selected = [];
    toolsData.orchid.forEach(tool => {
        if (tool.selected) selected.push(tool.name);
    });
    toolsData.user.forEach(tool => {
        if (tool.selected) selected.push(tool.name);
    });
    
    console.log('Selected tools:', selected);
    return selected;
}

function showToolsSection() {
    document.getElementById('toolsOverlay').style.display = 'flex';
    
    console.log('Showing tools overlay, current toolsData:', toolsData);
    
    // Force load tools immediately
    loadTools();
    
    // Also try to reload from API
    loadToolsFromAPI();
    setupUploadHandlers();
}

function closeToolsOverlay() {
    document.getElementById('toolsOverlay').style.display = 'none';
}

async function loadToolsFromAPI() {
    // Get API base URL
    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    
    try {
        const response = await fetch(`${apiBaseUrl}/tools`);
        if (response.ok) {
            const data = await response.json();
            toolsData.orchid = data.requiem_tools.map(tool => ({
                name: tool.name,
                description: tool.description,
                selected: tool.selected
            }));
            toolsData.user = data.user_tools.map(tool => ({
                name: tool.name,
                description: tool.description,
                selected: tool.selected
            }));
            console.log('Loaded tools from API:', toolsData);
        }
    } catch (error) {
        console.error('Failed to load tools from API:', error);
        // Fall back to local data
    }
    
    loadTools();
}

function loadTools() {
    console.log('loadTools() called');
    loadOrchidTools();
    loadUserTools();
}

function loadOrchidTools() {
    const container = document.getElementById('requiemTools');
    console.log('Loading Orchid tools, container:', container);
    console.log('toolsData.orchid:', toolsData.orchid);
    
    if (!container) {
        console.error('requiemTools container not found!');
        return;
    }
    
    container.innerHTML = '';
    
    toolsData.orchid.forEach(tool => {
        console.log('Creating tool card for:', tool.name);
        const toolCard = createToolCard(tool, 'orchid');
        container.appendChild(toolCard);
    });
    
    console.log('Loaded', toolsData.orchid.length, 'Orchid tools');
}

function loadUserTools() {
    const container = document.getElementById('userTools');
    container.innerHTML = '';
    
    if (toolsData.user.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-upload"></i>
                <h3>No custom tools</h3>
                <p>Upload your own calculation tools to get started</p>
            </div>
        `;
        return;
    }
    
    toolsData.user.forEach(tool => {
        const toolCard = createToolCard(tool, 'user');
        container.appendChild(toolCard);
    });
}

function createToolCard(tool, type) {
    const card = document.createElement('div');
    card.className = `tool-card ${tool.selected ? 'selected' : ''}`;
    card.innerHTML = `
        <div class="tool-info">
            <div class="tool-name">${tool.name}</div>
            <div class="tool-description">${tool.description}</div>
        </div>
        <div class="tool-actions">
            ${tool.selected ? 
                `<button class="tool-btn disable" onclick="toggleTool('${tool.name}', '${type}')">Disable</button>` :
                `<button class="tool-btn enable" onclick="toggleTool('${tool.name}', '${type}')">Enable</button>`
            }
            <button class="tool-btn edit" onclick="editTool('${tool.name}', '${type}')">Edit</button>
            ${type === 'user' ? `<button class="tool-btn delete" onclick="deleteTool('${tool.name}', '${type}')">Delete</button>` : ''}
        </div>
    `;
    return card;
}

function toggleTool(toolName, type) {
    const tool = toolsData[type].find(t => t.name === toolName);
    if (tool) {
        tool.selected = !tool.selected;
        loadTools();
        
        // Automatically save changes
        saveToolsToStorage();
        
        // Show brief feedback
        const status = tool.selected ? 'enabled' : 'disabled';
        console.log(`Tool '${toolName}' ${status}`);
    }
}

function editTool(toolName, type) {
    const tool = toolsData[type].find(t => t.name === toolName);
    if (tool) {
        const newDescription = prompt('Edit tool description:', tool.description);
        if (newDescription !== null) {
            tool.description = newDescription;
            loadTools();
            saveToolsToStorage();
        }
    }
}

function deleteTool(toolName, type) {
    if (confirm(`Are you sure you want to delete the tool "${toolName}"?`)) {
        toolsData[type] = toolsData[type].filter(t => t.name !== toolName);
        loadTools();
        saveToolsToStorage();
    }
}

function setupUploadHandlers() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('toolUpload');
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload({ target: { files: files } });
        }
    });
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.endsWith('.zip')) {
        alert('Please upload a ZIP file containing your tool.');
        return;
    }
    
    try {
        // Show loading state
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = '<i class="fas fa-spinner fa-spin"></i><br>Uploading...';
        
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', file);
        
        // Get API base URL
        const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
        
        // Upload to API
        const response = await fetch(`${apiBaseUrl}/tools/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Add to local tools data
            const newTool = {
                name: result.tool_name,
                description: `Custom tool uploaded from ${file.name}`,
                selected: true,
                filename: file.name,
                uploadedAt: new Date().toISOString()
            };
            
            toolsData.user.push(newTool);
            loadTools();
            saveToolsToStorage();
            
            alert(`Tool "${result.tool_name}" uploaded successfully!`);
        } else {
            const error = await response.json();
            alert(`Upload failed: ${error.detail}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed. Please try again.');
    } finally {
        // Reset upload area
        document.getElementById('uploadArea').innerHTML = `
            <i class="fas fa-arrow-up upload-icon"></i>
            <span class="upload-text">Upload new tool</span>
        `;
    }
}

function saveToolsToStorage() {
    localStorage.setItem('toolsData', JSON.stringify(toolsData));
}

function loadToolsFromStorage() {
    const saved = localStorage.getItem('toolsData');
    if (saved) {
        toolsData = JSON.parse(saved);
    }
}

// Utility functions
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showWelcomeScreen() {
    document.getElementById('toolsSection').style.display = 'none';
    document.getElementById('chatContainer').style.display = 'none';
    document.getElementById('welcomeSection').style.display = 'block';
    messagesContainer.innerHTML = '';
}

function showLoading(show, statusText = 'Processing...') {
    isLoading = show;
    loadingOverlay.style.display = show ? 'flex' : 'none';
    sendBtn.disabled = show || messageInput.value.trim() === '';
    
    if (show) {
        document.getElementById('loadingStatus').textContent = statusText;
        // Reset all stages
        document.querySelectorAll('.stage').forEach(stage => {
            stage.classList.remove('active', 'completed');
        });
    }
}

function updateLoadingStage(stageName, statusText) {
    const stages = document.querySelectorAll('.stage');
    const statusEl = document.getElementById('loadingStatus');
    
    if (statusText) {
        statusEl.textContent = statusText;
    }
    
    // Mark stages as completed or active
    let foundCurrent = false;
    stages.forEach(stage => {
        const stageData = stage.getAttribute('data-stage');
        
        if (stageData === stageName) {
            stage.classList.add('active');
            stage.classList.remove('completed');
            foundCurrent = true;
        } else if (!foundCurrent) {
            stage.classList.add('completed');
            stage.classList.remove('active');
        } else {
            stage.classList.remove('active', 'completed');
        }
    });
}

function getIntentMessage(intent) {
    const messages = {
        'backtest': 'Planning backtest strategy...',
        'price_query': 'Preparing to fetch price data...',
        'valuation': 'Setting up valuation analysis...',
        'comparison': 'Preparing asset comparison...',
        'analysis': 'Setting up technical analysis...',
        'tool_execution': 'Planning tool execution...',
        'unknown': 'Analyzing query...'
    };
    return messages[intent] || 'Planning response...';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function loadChatHistory() {
    // In a real app, this would load from localStorage or a database
    // For now, we'll just ensure the first chat is active
    const firstChat = document.querySelector('.chat-item');
    if (firstChat) {
        firstChat.classList.add('active');
    }
}

// Action handlers
function downloadResults() {
    // Placeholder for download functionality
    console.log('Downloading results...');
    // In a real app, this would generate and download a PDF or CSV
}

function shareResults() {
    // Placeholder for share functionality
    console.log('Sharing results...');
    // In a real app, this would copy to clipboard or open share dialog
}

// Chart creation functions
function createCharts(tearsheet) {
    try {
        // Get chart data from tearsheet
        const figures = tearsheet.figures || [];
        
        // Find equity curve data
        const equityData = figures.find(fig => fig.id === 'equity_curve');
        const drawdownData = figures.find(fig => fig.id === 'drawdown');
        const rollingSharpeData = figures.find(fig => fig.id === 'rolling_sharpe');
        
        console.log('Creating charts with data:', {
            equityData: equityData?.data ? 'Found' : 'Missing',
            drawdownData: drawdownData?.data ? 'Found' : 'Missing', 
            rollingSharpeData: rollingSharpeData?.data ? 'Found' : 'Missing'
        });
        
        if (equityData && equityData.data && equityData.data.labels && equityData.data.values) {
            createEquityChart(equityData.data);
        } else {
            console.log('Creating sample equity chart');
            createEquityChart({ labels: [], values: [] });
        }
        
        if (drawdownData && drawdownData.data && drawdownData.data.labels && drawdownData.data.values) {
            createDrawdownChart(drawdownData.data);
        } else {
            console.log('Creating sample drawdown chart');
            createDrawdownChart({ labels: [], values: [] });
        }
        
        if (rollingSharpeData && rollingSharpeData.data && rollingSharpeData.data.labels && rollingSharpeData.data.values) {
            createRollingSharpeChart(rollingSharpeData.data);
        } else {
            console.log('Creating sample rolling Sharpe chart');
            createRollingSharpeChart({ labels: [], values: [] });
        }
        
    } catch (error) {
        console.error('Error creating charts:', error);
        // Fallback: create sample charts
        console.log('Falling back to sample charts');
        createSampleCharts();
    }
}

function createEquityChart(data) {
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;
    
    // If no real data, use sample data
    const labels = data.labels && data.labels.length > 0 ? data.labels : 
        ['2023-01-03', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', 
         '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', 
         '2023-11-01', '2023-12-01'];
    const values = data.values && data.values.length > 0 ? data.values : 
        [1.0, 1.05, 1.12, 1.08, 1.15, 1.18, 1.22, 1.19, 1.25, 1.28, 1.24, 1.22];
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Equity Curve',
                data: values,
                borderColor: '#10a37f',
                backgroundColor: 'rgba(16, 163, 127, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Equity Curve',
                    font: { size: 14, weight: 'bold' }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Cumulative Return'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createDrawdownChart(data) {
    const ctx = document.getElementById('drawdownChart');
    if (!ctx) return;
    
    // If no real data, use sample data
    const labels = data.labels && data.labels.length > 0 ? data.labels : 
        ['2023-01-03', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', 
         '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', 
         '2023-11-01', '2023-12-01'];
    const values = data.values && data.values.length > 0 ? data.values : 
        [0, -2.1, -3.2, -5.1, -1.8, -0.5, 0.8, -2.3, 1.2, 2.8, -1.5, -0.8];
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Drawdown',
                data: values,
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Drawdown',
                    font: { size: 14, weight: 'bold' }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Drawdown %'
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createRollingSharpeChart(data) {
    const ctx = document.getElementById('rollingSharpeChart');
    if (!ctx) return;
    
    // If no real data, use sample data
    const labels = data.labels && data.labels.length > 0 ? data.labels : 
        ['2023-01-03', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', 
         '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', 
         '2023-11-01', '2023-12-01'];
    const values = data.values && data.values.length > 0 ? data.values : 
        [1.2, 1.4, 1.6, 1.5, 1.3, 1.8, 1.9, 1.7, 1.6, 1.4, 1.8, 1.9];
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Rolling Sharpe (3M)',
                data: values,
                borderColor: '#6f42c1',
                backgroundColor: 'rgba(111, 66, 193, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Rolling Sharpe Ratio (~3 months)',
                    font: { size: 14, weight: 'bold' }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Sharpe Ratio'
                    },
                    grid: {
                        color: function(context) {
                            if (context.tick.value === 0) return '#000';
                            if (context.tick.value === 1) return '#ffa500';
                            if (context.tick.value === 2) return '#800080';
                            return 'rgba(0,0,0,0.1)';
                        }
                    }
                }
            }
        }
    });
}

function createSampleCharts() {
    // Fallback sample data if real data isn't available
    const sampleDates = ['2023-01-03', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', 
                        '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', 
                        '2023-11-01', '2023-12-01'];
    
    const sampleEquity = [1.0, 1.05, 1.12, 1.08, 1.15, 1.18, 1.22, 1.19, 1.25, 1.28, 1.24, 1.22];
    const sampleDrawdown = [0, -2.1, -3.2, -5.1, -1.8, -0.5, 0.8, -2.3, 1.2, 2.8, -1.5, -0.8];
    const sampleSharpe = [null, null, null, 1.2, 1.4, 1.6, 1.5, 1.3, 1.8, 1.9, 1.7, 1.6];
    
    createEquityChart({ labels: sampleDates, values: sampleEquity });
    createDrawdownChart({ labels: sampleDates, values: sampleDrawdown });
    createRollingSharpeChart({ labels: sampleDates, values: sampleSharpe });
}

// Export for global access
window.downloadResults = downloadResults;

function addIntelligentAnalysisResult(data, message) {
    const time = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    if (!data) {
        addMessage('assistant', message || 'Intelligent analysis completed');
        return;
    }
    
    const ticker = data.ticker || 'Unknown';
    const analysisType = data.analysis_type || 'analysis';
    
    // Create a formatted message based on the analysis type
    let formattedMessage = message;
    
    // If it's earnings analysis, use the card format
    if (analysisType === 'earnings_analysis') {
        addStatisticalAnalysisResult(data, message);
        return;
    }
    
    // If it's technical analysis, create a professional display
    if (analysisType === 'technical_analysis') {
        const technicalSignals = data.technical_signals || {};
        const marketContext = data.market_context || {};
        const quantitativeInsights = data.quantitative_insights || {};
        
        formattedMessage = `
            <div class="intelligent-analysis-result">
                <div class="analysis-header">
                    <h3>📊 ${ticker} Technical Analysis</h3>
                    <div class="current-price">$${data.current_price?.toFixed(2) || 'N/A'}</div>
                </div>
                
                <div class="market-context-section">
                    <h4>Market Context</h4>
                    <div class="context-item">
                        <span class="label">52W Range Position:</span>
                        <span class="value">${(marketContext.price_position_52w * 100)?.toFixed(1) || 0}%</span>
                    </div>
                    <div class="context-item">
                        <span class="label">1M Change:</span>
                        <span class="value">${(marketContext.recent_volatility?.['1m_change'] * 100)?.toFixed(1) || 0}%</span>
                    </div>
                    <div class="context-item">
                        <span class="label">3M Change:</span>
                        <span class="value">${(marketContext.recent_volatility?.['3m_change'] * 100)?.toFixed(1) || 0}%</span>
                    </div>
                </div>
                
                <div class="technical-indicators-section">
                    <h4>Technical Indicators</h4>
                    ${this._formatTechnicalIndicators(technicalSignals)}
                </div>
                
                <div class="risk-metrics-section">
                    <h4>Risk Metrics</h4>
                    <div class="risk-item">
                        <span class="label">Downside Risk:</span>
                        <span class="value">${(quantitativeInsights.risk_metrics?.downside_risk * 100)?.toFixed(1) || 0}%</span>
                    </div>
                    <div class="risk-item">
                        <span class="label">Upside Potential:</span>
                        <span class="value">${(quantitativeInsights.risk_metrics?.upside_potential * 100)?.toFixed(1) || 0}%</span>
                    </div>
                    <div class="risk-item">
                        <span class="label">Risk/Reward Ratio:</span>
                        <span class="value">${quantitativeInsights.risk_metrics?.risk_reward_ratio || 'N/A'}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-bubble">
            ${formattedMessage}
        </div>
        <div class="message-time">${time}</div>
    `;
    
    messages.push({
        role: 'assistant',
        content: message,
        timestamp: time,
        data: data
    });
    
    const messagesContainer = document.getElementById('messagesContainer');
    if (messagesContainer) {
        messagesContainer.appendChild(messageDiv);
    } else {
        console.error('Messages container not found in addIntelligentAnalysisResult');
        addMessage('assistant', message || 'Intelligent analysis completed');
        return;
    }
    scrollToBottom();
}

function _formatTechnicalIndicators(technicalSignals) {
    let indicatorsHTML = '';
    
    // RSI
    if (technicalSignals.overbought_oversold) {
        technicalSignals.overbought_oversold.forEach(signal => {
            if (signal.indicator === 'RSI') {
                indicatorsHTML += `
                    <div class="indicator-item">
                        <div class="indicator-header">
                            <span class="indicator-name">RSI</span>
                            <span class="indicator-value">${signal.value?.toFixed(1) || 'N/A'}</span>
                        </div>
                        <div class="indicator-interpretation">${signal.interpretation || ''}</div>
                        <div class="indicator-note">${signal.quantitative_note || ''}</div>
                    </div>
                `;
            }
        });
    }
    
    // MACD
    if (technicalSignals.momentum_signals) {
        technicalSignals.momentum_signals.forEach(signal => {
            if (signal.indicator === 'MACD') {
                indicatorsHTML += `
                    <div class="indicator-item">
                        <div class="indicator-header">
                            <span class="indicator-name">MACD</span>
                            <span class="indicator-value">${signal.histogram?.toFixed(3) || 'N/A'}</span>
                        </div>
                        <div class="indicator-interpretation">${signal.interpretation || ''}</div>
                        <div class="indicator-note">${signal.quantitative_note || ''}</div>
                    </div>
                `;
            }
        });
    }
    
    // SMA
    if (technicalSignals.trend_signals) {
        technicalSignals.trend_signals.forEach(signal => {
            if (signal.indicator === 'SMA') {
                indicatorsHTML += `
                    <div class="indicator-item">
                        <div class="indicator-header">
                            <span class="indicator-name">SMA</span>
                            <span class="indicator-value">$${signal.sma_value?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div class="indicator-interpretation">${signal.interpretation || ''}</div>
                        <div class="indicator-note">${signal.quantitative_note || ''}</div>
                    </div>
                `;
            }
        });
    }
    
    // Bollinger Bands
    if (technicalSignals.volatility_signals) {
        technicalSignals.volatility_signals.forEach(signal => {
            if (signal.indicator === 'Bollinger Bands') {
                indicatorsHTML += `
                    <div class="indicator-item">
                        <div class="indicator-header">
                            <span class="indicator-name">Bollinger Bands</span>
                            <span class="indicator-value">${signal.band_position || 'N/A'}</span>
                        </div>
                        <div class="indicator-interpretation">${signal.interpretation || ''}</div>
                        <div class="indicator-note">${signal.quantitative_note || ''}</div>
                    </div>
                `;
            }
        });
    }
    
    return indicatorsHTML;
}

// Add chart interactivity
function addChartInteractivity() {
    try {
        // Find all data points in the current message
        const dataPoints = document.querySelectorAll('.data-point');
        const tooltips = document.querySelectorAll('[id^="tooltip-"]');
        const tooltipTexts = document.querySelectorAll('[id^="tooltip-text-"]');
        
        if (dataPoints.length === 0) {
            console.log('No data points found for interactivity');
            return;
        }
        
        dataPoints.forEach((point, index) => {
            const tooltip = tooltips[index];
            const tooltipText = tooltipTexts[index];
            
            if (tooltip && tooltipText) {
                point.addEventListener('mouseenter', (e) => {
                    try {
                        const value = e.target.getAttribute('data-value');
                        const date = e.target.getAttribute('data-date');
                        const pointIndex = e.target.getAttribute('data-index');
                        
                        // Position tooltip near the point
                        const rect = e.target.getBoundingClientRect();
                        const svgElement = e.target.closest('svg');
                        
                        if (svgElement) {
                            const svgRect = svgElement.getBoundingClientRect();
                            
                            tooltip.style.display = 'block';
                            tooltip.style.left = (rect.left - svgRect.left + 10) + 'px';
                            tooltip.style.top = (rect.top - svgRect.top - 50) + 'px';
                            tooltipText.style.display = 'block';
                            
                            // Show date and value in tooltip
                            if (date && date !== `Point ${parseInt(pointIndex) + 1}`) {
                                tooltipText.textContent = `${date}: ${value}`;
                            } else {
                                tooltipText.textContent = `Point ${parseInt(pointIndex) + 1}: ${value}`;
                            }
                        }
                    } catch (error) {
                        console.log('Error in mouseenter:', error);
                    }
                });
                
                point.addEventListener('mouseleave', () => {
                    try {
                        tooltip.style.display = 'none';
                        tooltipText.style.display = 'none';
                    } catch (error) {
                        console.log('Error in mouseleave:', error);
                    }
                });
            }
        });
    } catch (error) {
        console.log('Error in addChartInteractivity:', error);
    }
}

// Extract AI insights from the message
function extractAIInsights(message) {
    if (!message) return 'AI analysis will be displayed here...';
    
    // Look for the AI Insight section in the message
    const aiInsightMatch = message.match(/🔍 \*\*AI Insight\*\*\n(.*?)(?:\n\n|$)/s);
    if (aiInsightMatch) {
        return aiInsightMatch[1].trim();
    }
    
    // Fallback: look for any content after "AI Insight"
    const fallbackMatch = message.match(/AI Insight[:\s]*\n?(.*?)(?:\n\n|$)/s);
    if (fallbackMatch) {
        return fallbackMatch[1].trim();
    }
    
    return 'AI analysis will be displayed here...';
}

// Format parameters for display
function formatParameters(parameters) {
    if (!parameters || Object.keys(parameters).length === 0) {
        return 'Default';
    }
    
    // Shorten common parameter names
    const shortNames = {
        'timeperiod': 'timeperiod',
        'fastperiod': 'fast',
        'slowperiod': 'slow',
        'signalperiod': 'signal',
        'stddev': 'std',
        'matype': 'ma'
    };
    
    const formatted = Object.entries(parameters)
        .map(([key, value]) => {
            const shortKey = shortNames[key] || key;
            return `${shortKey}: ${value}`;
        })
        .join(', ');
    
    return formatted;
}

// Convert markdown to HTML for better formatting
function markdownToHtml(text) {
    if (!text) return '';
    
    return text
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')            // ### headers
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')             // ## headers
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')              // # headers
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // **text** -> <strong>text</strong>
        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // *text* -> <em>text</em>
        .replace(/\n\n/g, '</p><p>')                       // Double newlines -> paragraph breaks
        .replace(/\n/g, '<br>')                            // Single newlines -> line breaks
        .replace(/^/, '<p>')                               // Start with paragraph
        .replace(/$/, '</p>');                             // End with paragraph
}

// Generate simple SVG chart using real data
function generateSimpleChart(indicatorName, talib) {
    const width = 500; // Fixed width for viewBox
    const height = 300; // Fixed height for viewBox
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    
    const minVal = talib.min_value;
    const maxVal = talib.max_value;
    const latestVal = talib.latest_value;
    const meanVal = talib.mean_value;
    
    // Use real data if available, otherwise fall back to sample data
    let dataPoints = [];
    let dates = [];
    let macdData = null;
    let signalData = null;
    let histogramData = null;
    
    if (talib.chart_data && talib.chart_data.dates) {
        // Use real data from backend
        dates = talib.chart_data.dates;
        
        // Check if this is a multi-series indicator
        const isMultiSeries = (talib.chart_data.macd && talib.chart_data.signal && talib.chart_data.histogram) ||
                             (talib.chart_data.upper_band && talib.chart_data.middle_band && talib.chart_data.lower_band) ||
                             (talib.chart_data.stoch_k && talib.chart_data.stoch_d) ||
                             (talib.chart_data.aroon_up && talib.chart_data.aroon_down);
        
        if (isMultiSeries) {
            // Determine the indicator type and prepare data accordingly
            let allValues = [];
            let seriesData = {};
            
            if (talib.chart_data.macd && talib.chart_data.signal && talib.chart_data.histogram) {
                // MACD
                seriesData = {
                    macd: talib.chart_data.macd,
                    signal: talib.chart_data.signal,
                    histogram: talib.chart_data.histogram
                };
                allValues = [...seriesData.macd, ...seriesData.signal, ...seriesData.histogram];
            } else if (talib.chart_data.upper_band && talib.chart_data.middle_band && talib.chart_data.lower_band) {
                // Bollinger Bands
                seriesData = {
                    upper_band: talib.chart_data.upper_band,
                    middle_band: talib.chart_data.middle_band,
                    lower_band: talib.chart_data.lower_band,
                    price: talib.chart_data.price
                };
                allValues = [...seriesData.upper_band, ...seriesData.middle_band, ...seriesData.lower_band, ...(seriesData.price || [])];
            } else if (talib.chart_data.stoch_k && talib.chart_data.stoch_d) {
                // Stochastic
                seriesData = {
                    stoch_k: talib.chart_data.stoch_k,
                    stoch_d: talib.chart_data.stoch_d,
                    price: talib.chart_data.price
                };
                allValues = [...seriesData.stoch_k, ...seriesData.stoch_d, ...(seriesData.price || [])];
            } else if (talib.chart_data.aroon_up && talib.chart_data.aroon_down) {
                // Aroon
                seriesData = {
                    aroon_up: talib.chart_data.aroon_up,
                    aroon_down: talib.chart_data.aroon_down,
                    price: talib.chart_data.price
                };
                allValues = [...seriesData.aroon_up, ...seriesData.aroon_down, ...(seriesData.price || [])];
            }
            
            // Filter out null/undefined values and find min/max
            const validValues = allValues.filter(v => v !== null && v !== undefined);
            const seriesMin = Math.min(...validValues);
            const seriesMax = Math.max(...validValues);
            
            // Scale values to chart coordinates
            const scaleY = (value) => {
                if (value === null || value === undefined) return padding + chartHeight / 2;
                return padding + chartHeight - ((value - seriesMin) / (seriesMax - seriesMin)) * chartHeight;
            };
            const scaleX = (index) => padding + (index / (dates.length - 1)) * chartWidth;
            
            // Create data points for each series
            dataPoints = {};
            for (const [seriesName, seriesValues] of Object.entries(seriesData)) {
                if (seriesValues) {
                    dataPoints[seriesName] = seriesValues.map((value, index) => {
                        if (value === null || value === undefined) return null;
                        return {
                            x: scaleX(index),
                            y: scaleY(value),
                            value: value,
                            date: dates[index] || `Point ${index + 1}`
                        };
                    }).filter(point => point !== null);
                }
            }
            
            // For multi-series indicators, filter out null values and start from first valid data point
            if (isMultiSeries && Object.keys(dataPoints).length > 0) {
                // Find the first valid data point across all series
                let firstValidIndex = Infinity;
                for (const [seriesName, seriesPoints] of Object.entries(dataPoints)) {
                    if (seriesPoints && seriesPoints.length > 0) {
                        for (let i = 0; i < seriesPoints.length; i++) {
                            if (seriesPoints[i] && seriesPoints[i].value !== null && seriesPoints[i].value !== undefined) {
                                firstValidIndex = Math.min(firstValidIndex, i);
                                break;
                            }
                        }
                    }
                }
                
                if (firstValidIndex !== Infinity && firstValidIndex > 0) {
                    // Filter all series to start from the first valid data point
                    for (const [seriesName, seriesPoints] of Object.entries(dataPoints)) {
                        if (seriesPoints && seriesPoints.length > 0) {
                            dataPoints[seriesName] = seriesPoints.slice(firstValidIndex);
                        }
                    }
                    
                    // Also filter the dates array to match
                    dates = dates.slice(firstValidIndex);
                }
            }
            
            // Additional filtering for raw data arrays (before they're converted to dataPoints)
            if (isMultiSeries && talib.chart_data) {
                // Find the first valid data point in the raw data
                let firstValidIndex = Infinity;
                
                // Check MACD line first (main indicator)
                if (talib.chart_data.macd) {
                    for (let i = 0; i < talib.chart_data.macd.length; i++) {
                        if (talib.chart_data.macd[i] !== null && talib.chart_data.macd[i] !== undefined) {
                            firstValidIndex = Math.min(firstValidIndex, i);
                            break;
                        }
                    }
                }
                
                console.log(`🔍 MACD filtering: firstValidIndex = ${firstValidIndex}, total data points = ${talib.chart_data.dates ? talib.chart_data.dates.length : 'N/A'}`);
                
                if (firstValidIndex !== Infinity && firstValidIndex > 0) {
                    console.log(`✂️ Filtering out first ${firstValidIndex} null values`);
                    
                    // Filter all raw data arrays
                    if (talib.chart_data.dates) {
                        talib.chart_data.dates = talib.chart_data.dates.slice(firstValidIndex);
                    }
                    if (talib.chart_data.macd) {
                        talib.chart_data.macd = talib.chart_data.macd.slice(firstValidIndex);
                    }
                    if (talib.chart_data.signal) {
                        talib.chart_data.signal = talib.chart_data.signal.slice(firstValidIndex);
                    }
                    if (talib.chart_data.histogram) {
                        talib.chart_data.histogram = talib.chart_data.histogram.slice(firstValidIndex);
                    }
                    if (talib.chart_data.price) {
                        talib.chart_data.price = talib.chart_data.price.slice(firstValidIndex);
                    }
                    
                    // Update the dates array used for chart generation
                    dates = talib.chart_data.dates;
                    
                    console.log(`✅ After filtering: ${dates.length} data points, starting from ${dates[0]}`);
                }
            }
        } else if (talib.chart_data.values) {
            // Single series chart (RSI, SMA, etc.)
            const values = talib.chart_data.values;
            
            // Scale values to chart coordinates
            const scaleY = (value) => {
                if (value === null || value === undefined) return padding + chartHeight / 2; // Center for null values
                return padding + chartHeight - ((value - minVal) / (maxVal - minVal)) * chartHeight;
            };
            const scaleX = (index) => padding + (index / (values.length - 1)) * chartWidth;
            
            dataPoints = values.map((value, index) => {
                if (value === null || value === undefined) {
                    return null; // Mark null values for filtering
                }
                return {
                    x: scaleX(index),
                    y: scaleY(value),
                    value: value,
                    date: dates[index] || `Point ${index + 1}`
                };
            }).filter(point => point !== null); // Filter out null points
        }
    } else {
        // Fallback to sample data if real data not available
        const scaleY = (value) => padding + chartHeight - ((value - minVal) / (maxVal - minVal)) * chartHeight;
        const numPoints = 30;
        for (let i = 0; i < numPoints; i++) {
            const x = padding + (i / (numPoints - 1)) * chartWidth;
            const progress = i / (numPoints - 1);
            const baseValue = meanVal + (maxVal - minVal) * 0.3 * Math.sin(progress * Math.PI * 2) * Math.exp(-progress * 0.5);
            const noise = (Math.random() - 0.5) * (maxVal - minVal) * 0.1;
            const value = Math.max(minVal, Math.min(maxVal, baseValue + noise));
            const y = scaleY(value);
            dataPoints.push({ x, y, value, date: `Point ${i + 1}` });
        }
    }
    
    // Check if this is a multi-series indicator
    const isMultiSeries = dataPoints && typeof dataPoints === 'object' && !Array.isArray(dataPoints);
    
    let pathData = '';
    let dataPointsElements = '';
    let currentY = padding + chartHeight / 2;
    let meanY = padding + chartHeight / 2;
    
    if (isMultiSeries) {
        // Multi-series indicator (MACD, Bollinger Bands, Stochastic, Aroon)
        const seriesColors = {
            macd: '#3B82F6',
            signal: '#f59e0b',
            histogram: '#10b981',
            upper_band: '#ef4444',
            middle_band: '#6b7280',
            lower_band: '#10b981',
            stoch_k: '#3B82F6',
            stoch_d: '#f59e0b',
            aroon_up: '#10b981',
            aroon_down: '#ef4444',
            price: '#000000'
        };
        
        const seriesLabels = {
            macd: 'MACD Line',
            signal: 'Signal Line',
            histogram: 'Histogram',
            upper_band: 'Upper Band',
            middle_band: 'Middle Band',
            lower_band: 'Lower Band',
            stoch_k: 'Stochastic %K',
            stoch_d: 'Stochastic %D',
            aroon_up: 'Aroon Up',
            aroon_down: 'Aroon Down',
            price: 'Price'
        };
        
        // Generate paths for each series
        const paths = [];
        const points = [];
        
        for (const [seriesName, seriesPoints] of Object.entries(dataPoints)) {
            if (seriesPoints && seriesPoints.length > 0) {
                const color = seriesColors[seriesName] || '#3B82F6';
                const label = seriesLabels[seriesName] || seriesName;
                
                if (seriesName === 'histogram') {
                    // Special handling for histogram bars
                    const histogramBars = seriesPoints.map((point, index) => {
                        const barHeight = Math.abs(point.y - (padding + chartHeight / 2));
                        const barY = point.value >= 0 ? (padding + chartHeight / 2) : (padding + chartHeight / 2) - barHeight;
                        const barColor = point.value >= 0 ? '#10b981' : '#ef4444';
                        return `<rect x="${point.x - 1}" y="${barY}" width="2" height="${barHeight}" fill="${barColor}" opacity="0.7" />`;
                    }).join('');
                    paths.push(`<!-- Histogram -->\n${histogramBars}`);
                } else {
                    // Regular line series
                    const path = seriesPoints.map((point, index) => 
                        `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
                    ).join(' ');
                    paths.push(`<!-- ${label} -->\n<path d="${path}" fill="none" stroke="${color}" stroke-width="2" class="data-line" />`);
                    
                    // Add data points (show fewer for cleaner look)
                    const filteredPoints = seriesPoints.filter((_, index) => index % Math.max(1, Math.floor(seriesPoints.length / 15)) === 0);
                    points.push(...filteredPoints.map((point, index) => 
                        `<circle cx="${point.x}" cy="${point.y}" r="2" fill="${color}" 
                                 class="data-point" data-value="${(point.value || 0).toFixed(4)}" 
                                 data-date="${point.date}" data-series="${label}" style="cursor: pointer;" />`
                    ));
                }
            }
        }
        
        pathData = paths.join('\n');
        dataPointsElements = points.join('');
    } else {
        // Single series chart (RSI, SMA, etc.)
        const pathString = dataPoints.map((point, index) => 
            `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
        ).join(' ');
        pathData = `<path d="${pathString}" fill="none" stroke="#3B82F6" stroke-width="2" class="data-line" />`;
        
        // Current value line
        currentY = padding + chartHeight - ((latestVal - minVal) / (maxVal - minVal)) * chartHeight;
        meanY = padding + chartHeight - ((meanVal - minVal) / (maxVal - minVal)) * chartHeight;
        
        // Data points with hover effects (show fewer points to make line more prominent)
        dataPointsElements = dataPoints.filter((_, index) => index % Math.max(1, Math.floor(dataPoints.length / 10)) === 0).map((point, index) => 
            `<circle cx="${point.x}" cy="${point.y}" r="1.5" fill="#3B82F6" 
                     class="data-point" data-value="${(point.value || 0).toFixed(2)}" 
                     data-date="${point.date}" data-index="${index}" style="cursor: pointer;" />`
        ).join('');
    }
    
    // Generate date labels for x-axis (show every 5th date to avoid crowding)
    const dateLabels = [];
    if (dates.length > 0) {
        const step = Math.max(1, Math.floor(dates.length / 6)); // Show ~6 date labels
        for (let i = 0; i < dates.length; i += step) {
            const date = new Date(dates[i]);
            const label = `${date.getMonth() + 1}/${date.getDate()}`;
            const x = padding + (i / (dates.length - 1)) * chartWidth;
            dateLabels.push(`<text x="${x}" y="${height - padding + 15}" text-anchor="middle" class="chart-label">${label}</text>`);
        }
    }
    
    // Create legend for multi-series indicators
    let legend = '';
    if (isMultiSeries) {
        const seriesColors = {
            macd: '#3B82F6',
            signal: '#f59e0b',
            histogram: '#10b981',
            upper_band: '#ef4444',
            middle_band: '#6b7280',
            lower_band: '#10b981',
            stoch_k: '#3B82F6',
            stoch_d: '#f59e0b',
            aroon_up: '#10b981',
            aroon_down: '#ef4444',
            price: '#000000'
        };
        
        const seriesLabels = {
            macd: 'MACD Line',
            signal: 'Signal Line',
            histogram: 'Histogram',
            upper_band: 'Upper Band',
            middle_band: 'Middle Band',
            lower_band: 'Lower Band',
            stoch_k: 'Stochastic %K',
            stoch_d: 'Stochastic %D',
            aroon_up: 'Aroon Up',
            aroon_down: 'Aroon Down',
            price: 'Price'
        };
        
        // Generate legend items for all available series
        const legendItems = [];
        for (const [seriesName, seriesPoints] of Object.entries(dataPoints)) {
            if (seriesPoints && seriesPoints.length > 0) {
                const color = seriesColors[seriesName] || '#3B82F6';
                const label = seriesLabels[seriesName] || seriesName;
                
                if (seriesName === 'histogram') {
                    legendItems.push(`
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 12px; height: 8px; background: linear-gradient(to top, #ef4444 50%, #10b981 50%);"></div>
                            <span>${label}</span>
                        </div>
                    `);
                } else {
                    legendItems.push(`
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 12px; height: 2px; background: ${color};"></div>
                            <span>${label}</span>
                        </div>
                    `);
                }
            }
        }
        
        if (legendItems.length > 0) {
            legend = `
                <div class="chart-legend" style="display: flex; justify-content: center; gap: 20px; margin-bottom: 10px; font-size: 12px; flex-wrap: wrap;">
                    ${legendItems.join('')}
                </div>
            `;
        }
    }
    
    return `
        <div class="chart-info">${indicatorName.toLowerCase()} Chart - ${talib.data_points} data points</div>
        ${legend}
        <div class="chart-wrapper">
            <svg viewBox="0 0 ${width} ${height}" class="simple-chart" preserveAspectRatio="xMidYMid meet">
                <!-- Grid lines -->
                <defs>
                    <pattern id="grid-${Date.now()}" width="50" height="50" patternUnits="userSpaceOnUse">
                        <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#f0f0f0" stroke-width="1"/>
                    </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid-${Date.now()})" />
                
                <!-- Chart area -->
                <rect x="${padding}" y="${padding}" width="${chartWidth}" height="${chartHeight}" 
                      fill="white" stroke="#e0e0e0" stroke-width="1" />
                
                <!-- Data lines and elements -->
                ${pathData}
                
                <!-- Data points with hover effects -->
                ${dataPointsElements}
                
                <!-- Mean line -->
                <line x1="${padding}" y1="${meanY}" x2="${width - padding}" y2="${meanY}" 
                      stroke="#10B981" stroke-width="2" stroke-dasharray="5,5" />
                
                <!-- Current value line -->
                <line x1="${padding}" y1="${currentY}" x2="${width - padding}" y2="${currentY}" 
                      stroke="#EF4444" stroke-width="2" stroke-dasharray="3,3" />
                
                <!-- Y-axis labels -->
                <text x="${padding - 10}" y="${padding + 5}" text-anchor="end" class="chart-label">${maxVal.toFixed(1)}</text>
                <text x="${padding - 10}" y="${meanY + 5}" text-anchor="end" class="chart-label">${meanVal.toFixed(1)}</text>
                <text x="${padding - 10}" y="${height - padding + 5}" text-anchor="end" class="chart-label">${minVal.toFixed(1)}</text>
                
                <!-- X-axis date labels -->
                ${dateLabels.join('')}
                
                <!-- Current value label -->
                <text x="${width - padding + 10}" y="${currentY + 5}" class="chart-current-label">
                    Current: ${latestVal.toFixed(2)}
                </text>
                
                <!-- Tooltip -->
                <rect id="tooltip-${Date.now()}" x="0" y="0" width="120" height="40" 
                      fill="rgba(0,0,0,0.8)" stroke="none" rx="4" 
                      style="display: none; pointer-events: none;" />
                <text id="tooltip-text-${Date.now()}" x="10" y="20" fill="white" 
                      font-size="12" style="display: none; pointer-events: none;" />
            </svg>
        </div>
    `;
}

// Generate technical insight for TA-Lib indicators
function generateTechnicalInsight(indicatorName, latestValue, meanValue, parameters) {
    const indicator = indicatorName.toLowerCase();
    
    if (indicator === 'rsi') {
        if (latestValue > 70) {
            return `
                <div class="insight-section">
                    <div class="insight-signal overbought">OVERBOUGHT</div>
                    <div class="insight-recommendation">Consider selling or waiting for pullback</div>
                </div>
                <div class="insight-details">
                    <div class="insight-item"><strong>Current Value:</strong> ${latestValue.toFixed(2)}</div>
                    <div class="insight-item"><strong>Average:</strong> ${meanValue.toFixed(2)}</div>
                    <div class="insight-item"><strong>Overbought Level:</strong> 70</div>
                    <div class="insight-item"><strong>Oversold Level:</strong> 30</div>
                </div>
            `;
        } else if (latestValue < 30) {
            return `
                <div class="insight-section">
                    <div class="insight-signal oversold">OVERSOLD</div>
                    <div class="insight-recommendation">Potential buying opportunity</div>
                </div>
                <div class="insight-details">
                    <div class="insight-item"><strong>Current Value:</strong> ${latestValue.toFixed(2)}</div>
                    <div class="insight-item"><strong>Average:</strong> ${meanValue.toFixed(2)}</div>
                    <div class="insight-item"><strong>Overbought Level:</strong> 70</div>
                    <div class="insight-item"><strong>Oversold Level:</strong> 30</div>
                </div>
            `;
        } else {
            return `
                <div class="insight-section">
                    <div class="insight-signal neutral">NEUTRAL</div>
                    <div class="insight-recommendation">Wait for clearer signals</div>
                </div>
                <div class="insight-details">
                    <div class="insight-item"><strong>Current Value:</strong> ${latestValue.toFixed(2)}</div>
                    <div class="insight-item"><strong>Average:</strong> ${meanValue.toFixed(2)}</div>
                    <div class="insight-item"><strong>Overbought Level:</strong> 70</div>
                    <div class="insight-item"><strong>Oversold Level:</strong> 30</div>
                </div>
            `;
        }
    } else if (indicator === 'sma') {
        const period = parameters.timeperiod || 20;
        const trend = latestValue > meanValue ? 'Bullish' : 'Bearish';
        return `
            <div class="insight-section">
                <div class="insight-signal ${trend.toLowerCase()}">${trend} TREND</div>
                <div class="insight-recommendation">Current price relative to ${period}-day average</div>
            </div>
            <div class="insight-details">
                <div class="insight-item"><strong>Current Value:</strong> ${latestValue.toFixed(2)}</div>
                <div class="insight-item"><strong>Average:</strong> ${meanValue.toFixed(2)}</div>
                <div class="insight-item"><strong>Period:</strong> ${period} days</div>
            </div>
        `;
    } else if (indicator === 'macd') {
        const signal = latestValue > 0 ? 'Positive' : 'Negative';
        return `
            <div class="insight-section">
                <div class="insight-signal ${signal.toLowerCase()}">${signal} MOMENTUM</div>
                <div class="insight-recommendation">MACD momentum indicator showing trend changes</div>
            </div>
            <div class="insight-details">
                <div class="insight-item"><strong>Current Value:</strong> ${latestValue.toFixed(4)}</div>
                <div class="insight-item"><strong>Average:</strong> ${meanValue.toFixed(4)}</div>
                <div class="insight-item"><strong>Signal:</strong> ${signal} momentum</div>
            </div>
        `;
    } else {
        return `
            <div class="insight-section">
                <div class="insight-signal neutral">${indicatorName.toUpperCase()} ANALYSIS</div>
                <div class="insight-recommendation">Technical indicator providing market insights</div>
            </div>
            <div class="insight-details">
                <div class="insight-item"><strong>Current Value:</strong> ${latestValue.toFixed(4)}</div>
                <div class="insight-item"><strong>Average:</strong> ${meanValue.toFixed(4)}</div>
                <div class="insight-item"><strong>Parameters:</strong> ${JSON.stringify(parameters)}</div>
            </div>
        `;
    }
}

// Statistical Analysis Result Display
function addStatisticalAnalysisResult(data, message) {
    console.log('🎯 addStatisticalAnalysisResult called with:', data);
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) {
        console.error('❌ messagesContainer not found');
        return;
    }
    
    let analysisHTML = '';
    
    // Check if this is an earnings analysis response
    if (data.analysis_type === 'earnings_analysis' && data.earnings_result) {
        const earnings = data.earnings_result;
        const ticker = data.ticker || 'Unknown';
        const quarter = earnings.quarter || 'Unknown';
        
        
        // Extract AI insights from the message
        const aiInsights = extractAIInsights(message);
        
        analysisHTML = `
            <div class="message assistant-message">
                <div class="analysis-card earnings-card">
                    <div class="card-header">
                        <div class="card-title">
                            <span class="card-icon">📊</span>
                            <span class="card-name">Earnings Analysis</span>
                        </div>
                        <div class="card-meta">
                            <span class="ticker">${ticker}</span>
                            <span class="quarter">${quarter}</span>
                        </div>
                    </div>
                    
                    <div class="card-content">
                        ${earnings.tldr && earnings.tldr.length > 0 ? `
                        <div class="highlights-section">
                            <h4>📋 Key Highlights</h4>
                            <ul class="highlights-list">
                                ${earnings.tldr.map(highlight => `<li>${highlight}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        ${earnings.kpis && earnings.kpis.length > 0 ? `
                        <div class="metrics-section">
                            <h4>📈 Key Metrics</h4>
                            <div class="metrics-grid">
                                ${earnings.kpis.map(kpi => `
                                    <div class="metric-item">
                                        <span class="metric-name">${kpi.metric || 'Unknown'}</span>
                                        <span class="metric-value">${kpi.value || 'N/A'}</span>
                                        ${kpi.change ? `<span class="metric-change">${kpi.change}</span>` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}
                        
                        ${earnings.risks ? `
                        <div class="risks-section">
                            <h4>⚠️ Key Risks</h4>
                            <ul class="risks-list">
                                ${earnings.risks.map(risk => `<li>${risk}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        ${earnings.quotes ? `
                        <div class="quotes-section">
                            <h4>💬 Key Quotes</h4>
                            <div class="quotes-list">
                                ${earnings.quotes.map(quote => `
                                    <div class="quote-item">
                                        <div class="quote-text">"${quote.quote || quote}"</div>
                                        <div class="quote-attribution">- ${quote.speaker || 'Unknown'} ${quote.page ? `(p.${quote.page})` : ''}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}
                        
                        ${earnings.sources ? `
                        <div class="sources-section">
                            <h4>🔗 Sources</h4>
                            <ul class="sources-list">
                                ${earnings.sources.map(source => `<li><a href="${source}" target="_blank">${source}</a></li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        ${aiInsights && aiInsights !== 'AI analysis will be displayed here...' ? `
                        <div class="insights-section">
                            <h4>🔍 AI Insight</h4>
                            <div class="insight-text">${aiInsights}</div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    // Check if this is a TA-Lib technical analysis response
    else if (data.card_format && data.talib_result) {
        const talib = data.talib_result;
        const indicatorName = data.indicator_name || talib.tool_name || 'Technical Indicator';
        const ticker = data.ticker || talib.ticker || 'Unknown';
        
        // Format the card for TA-Lib technical analysis
        // Extract AI insights from the message
        const aiInsights = extractAIInsights(message);
        
        analysisHTML = `
            <div class="message assistant-message">
                <div class="message-content">
                    <div class="statistical-analysis-card">
                        <div class="card-header">
                            <h2 class="card-title">📊 ${indicatorName.toUpperCase()} Analysis — ${ticker}</h2>
                            <div class="card-meta">Period: ${talib.period} · Data Points: ${talib.data_points} · Tool: TA-Lib · ${formatParameters(talib.parameters_used)}</div>
                        </div>
                        
                        <div class="card-metrics">
                            <div class="metric-item">
                                <div class="metric-label">Latest</div>
                                <div class="metric-value">${talib.latest_value.toFixed(4)}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Mean</div>
                                <div class="metric-value">${talib.mean_value.toFixed(4)}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Min</div>
                                <div class="metric-value">${talib.min_value.toFixed(4)}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Max</div>
                                <div class="metric-value">${talib.max_value.toFixed(4)}</div>
                            </div>
                        </div>
                        
                        <div class="card-chart">
                            <div class="chart-title">📊 Chart</div>
                            <div class="chart-container" id="chart-${Date.now()}">
                                <!-- Chart will be rendered here by renderTechnicalChart -->
                            </div>
                        </div>
                        
                        <div class="card-insight">
                            <div class="insight-title">🔍 AI Insight</div>
                            <div class="insight-content">
                                <div class="insight-text">${markdownToHtml(aiInsights)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    // Check if this is a statistical analysis with card data
    else if (data.card_data) {
        const card = data.card_data;
        
        analysisHTML = `
            <div class="message assistant-message">
                <div class="message-content">
                    <div class="statistical-analysis-card">
                        <div class="card-header">
                            <h2 class="card-title">${card.title}</h2>
                            <div class="card-meta">${card.meta}</div>
                        </div>
                        
                        <div class="card-metrics">
                            <h3>📊 Key Metrics</h3>
                            <div class="metrics-grid">
                                ${card.metrics.map(metric => `
                                    <div class="metric-item">
                                        <div class="metric-key">${metric.key}</div>
                                        <div class="metric-value">${metric.value}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        
                        ${data.analysis_type === 'regression_analysis' ? `
                            <div class="card-coefficients">
                                <h3>β Coefficients</h3>
                                <div class="coefficients-table">
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Variable</th>
                                                <th>Beta</th>
                                                <th>p-value</th>
                                                <th>Sig</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${card.coefficients.map(coeff => `
                                                <tr>
                                                    <td>${coeff.var}</td>
                                                    <td class="beta-value ${coeff.beta.startsWith('+') ? 'positive' : 'negative'}">${coeff.beta}</td>
                                                    <td>${coeff.p}</td>
                                                    <td class="significance">${coeff.sig ? '✅' : '❌'}</td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ` : ''}
                        
                        ${data.analysis_type === 'correlation_analysis' ? `
                            <div class="card-correlations">
                                <h3>🔗 Correlations</h3>
                                <div class="correlations-table">
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Pair</th>
                                                <th>Value</th>
                                                <th>Strength</th>
                                                <th>Sig</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${card.correlations.map(corr => `
                                                <tr>
                                                    <td>${corr.pair}</td>
                                                    <td class="correlation-value ${parseFloat(corr.value) > 0 ? 'positive' : 'negative'}">${corr.value}</td>
                                                    <td class="strength ${corr.strength}">${corr.strength}</td>
                                                    <td class="significance">${corr.sig ? '✅' : '❌'}</td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ` : ''}
                        
                        ${data.analysis_type === 'cointegration_test' ? `
                            <div class="card-results">
                                <h3>📊 Test Results</h3>
                                <div class="results-table">
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>${card.results[0].pair ? 'Pair' : 'Test'}</th>
                                                <th>Score</th>
                                                <th>p-value</th>
                                                <th>Sig</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${card.results.map(result => `
                                                <tr>
                                                    <td>${result.pair || result.test}</td>
                                                    <td class="score-value">${result.score || result.rank}</td>
                                                    <td>${result.pvalue}</td>
                                                    <td class="significance">${result.sig ? '✅' : '❌'}</td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ` : ''}
                        
                        ${data.analysis_type === 'risk_metrics' ? `
                            <div class="card-risk-metrics">
                                <h3>⚠️ Risk Metrics</h3>
                                <div class="risk-metrics-grid">
                                    ${card.metrics.map(metric => `
                                        <div class="risk-metric-item">
                                            <div class="risk-metric-key">${metric.key}</div>
                                            <div class="risk-metric-value ${metric.key.toLowerCase().includes('var') || metric.key.toLowerCase().includes('drawdown') ? 'negative' : 'positive'}">${metric.value}</div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                        
                        ${data.analysis_type === 'mathematical_calculation' ? `
                            <div class="card-math-calc">
                                <h3>🧮 Mathematical Results</h3>
                                <div class="math-calc-grid">
                                    ${card.metrics.map(metric => `
                                        <div class="math-calc-item">
                                            <div class="math-calc-key">${metric.key}</div>
                                            <div class="math-calc-value">${metric.value}</div>
                                        </div>
                                    `).join('')}
                                </div>
                                ${data.calculation_type === 'monte_carlo_simulation' ? `
                                    <div class="monte-carlo-info">
                                        <h4>📊 Simulation Details</h4>
                                        <p>Monte Carlo simulation using ${data.parameters?.model_type?.replace('_', ' ').toUpperCase() || 'Geometric Brownian Motion'} model with ${data.parameters?.n_simulations?.toLocaleString() || '10,000'} simulations.</p>
                                        ${data.parameters?.ticker ? `<p><strong>Data Source:</strong> Real historical data for ${data.parameters.ticker}</p>` : ''}
                                    </div>
                                ` : ''}
                            </div>
                        ` : ''}
                        
                        ${(data.analysis_type === 'technical_analysis' || data.analysis_type === 'monte_carlo_simulation' || (data.tool_name && card.chart_data)) && card.chart_data ? `
                            <div class="card-chart">
                                <h3>📈 Chart</h3>
                                <div class="chart-container">
                                    <canvas id="chart-${data.analysis_type === 'monte_carlo_simulation' ? 'monte-carlo-simulation' : (data.ticker || 'unknown') + '-' + (data.tool_name || 'indicator')}" width="400" height="200"></canvas>
                                </div>
                            </div>
                        ` : ''}
                        
                        <div class="card-insight">
                            <h3>🔍 AI Insight</h3>
                            <p>${card.insight}</p>
                        </div>
                        
                        ${data.analysis_type === 'regression_analysis' && card.model_comparison && card.model_comparison.length > 0 ? `
                            <div class="card-comparison">
                                <h3>Model Comparison</h3>
                                <div class="comparison-table">
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Model</th>
                                                <th>Variables</th>
                                                <th>R²</th>
                                                <th>Beta</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${card.model_comparison.map(comp => `
                                                <tr>
                                                    <td>${comp.model}</td>
                                                    <td>${comp.vars}</td>
                                                    <td>${comp.r2}</td>
                                                    <td>${comp.beta || '—'}</td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    } else {
        // Fallback to regular message for other analysis types
        analysisHTML = `
            <div class="message assistant-message">
                <div class="message-content">
                    <div class="analysis-result">
                        <h3>📊 ${data.analysis_type?.replace('_', ' ').toUpperCase() || 'Statistical Analysis'}</h3>
                        <div class="analysis-content">
                            ${message.replace(/\n/g, '<br>')}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    messagesContainer.insertAdjacentHTML('beforeend', analysisHTML);
    
    // Add chart interactivity with error handling
    try {
        // Find the card element that was just added
        const cards = document.querySelectorAll('.analysis-card');
        const lastCard = cards[cards.length - 1];
        if (lastCard) {
            const chartElement = lastCard.querySelector('.simple-chart');
            if (chartElement) {
                addChartInteractivity();
            }
        }
    } catch (error) {
        console.log('Chart interactivity error (non-critical):', error);
    }
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Render chart if it's technical analysis or Monte Carlo simulation with chart data
    const chartData = data.talib_result?.chart_data || data.card_data?.chart_data;
    const isTechnicalAnalysis = data.analysis_type === 'technical_analysis' || data.card_format === true || data.talib_result;
    const hasChartData = chartData && (chartData.dates || chartData.time_steps) && (chartData.values || chartData.price || chartData.simulation_paths || chartData.macd);
    
    console.log('Chart rendering debug:', {
        isTechnicalAnalysis,
        hasChartData,
        chartData: chartData ? Object.keys(chartData) : 'none',
        dataKeys: Object.keys(data)
    });
    
    if (isTechnicalAnalysis && hasChartData) {
        console.log('Chart rendering condition met, looking for canvas...');
        setTimeout(() => {
            // Generate appropriate canvas ID based on analysis type
            let canvasId;
            if (data.analysis_type === 'monte_carlo_simulation') {
                // Use a fixed ID for Monte Carlo simulations
                canvasId = 'chart-monte-carlo-simulation';
            } else {
                canvasId = `chart-${data.ticker || 'unknown'}-${data.tool_name || 'indicator'}`;
            }
            console.log('Looking for chart container...');
            // Look for the most recent chart container
            const chartContainers = document.querySelectorAll('.chart-container');
            const lastChartContainer = chartContainers[chartContainers.length - 1];
            console.log('Chart containers found:', chartContainers.length);
            console.log('Last chart container:', lastChartContainer);
            
            if (lastChartContainer) {
                console.log('Chart container found, calling renderTechnicalChart...');
                console.log('Chart data:', chartData);
                renderTechnicalChart(lastChartContainer, chartData, data.tool_name || data.analysis_type || 'Analysis');
            } else {
                console.error('No chart container found');
                console.log('Available chart containers:', document.querySelectorAll('.chart-container'));
            }
        }, 100);
    } else {
        console.log('Chart rendering condition not met');
        console.log('data.analysis_type:', data.analysis_type);
        console.log('data.tool_name:', data.tool_name);
        console.log('data.card_data:', data.card_data);
        console.log('data.card_data.chart_data:', data.card_data?.chart_data);
    }
}

function renderTechnicalChart(canvas, chartData, toolName) {
    console.log('renderTechnicalChart called with:', { chartData, toolName, canvas });
    
    // Clear any existing content
    canvas.innerHTML = '';
    
    // Create a mock talib object that matches the structure expected by generateSimpleChart
    let latestValue = 0;
    let meanValue = 0;
    let minValue = 0;
    let maxValue = 0;
    
    // Get the correct latest value based on chart type
    if (chartData.macd && chartData.signal && chartData.histogram) {
        // MACD - use MACD line as main value
        const macdValues = chartData.macd.filter(v => v !== null && v !== undefined);
        latestValue = macdValues[macdValues.length - 1] || 0;
        meanValue = macdValues.reduce((a, b) => a + b, 0) / macdValues.length || 0;
        minValue = Math.min(...macdValues, ...chartData.signal, ...chartData.histogram);
        maxValue = Math.max(...macdValues, ...chartData.signal, ...chartData.histogram);
    } else if (chartData.values) {
        // Single series
        const values = chartData.values.filter(v => v !== null && v !== undefined);
        latestValue = values[values.length - 1] || 0;
        meanValue = values.reduce((a, b) => a + b, 0) / values.length || 0;
        minValue = Math.min(...values);
        maxValue = Math.max(...values);
    }
    
    const mockTalib = {
        min_value: minValue,
        max_value: maxValue,
        latest_value: latestValue,
        mean_value: meanValue,
        data_points: chartData.dates ? chartData.dates.length : 0,
        chart_data: chartData
    };
    
    // Use the beautiful generateSimpleChart function
    const chartHTML = generateSimpleChart(toolName, mockTalib);
    
    // Insert the chart HTML
    canvas.innerHTML = chartHTML;
    
    // Add chart interactivity
    addChartInteractivity(canvas, chartData.values || [], chartData.dates || [], minValue, maxValue, maxValue - minValue);
    
    console.log('Chart rendered successfully with generateSimpleChart');
}

function addChartInteractivity(chartDiv, values, dates, minValue, maxValue, valueRange, bollingerData = null) {
    const chartArea = chartDiv.querySelector('div[style*="position: absolute; top: 40px"]');
    if (!chartArea) return;
    
    // Create tooltip
    const tooltip = document.createElement('div');
    tooltip.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        z-index: 1000;
        display: none;
    `;
    
    // Add tooltip to chart
    chartDiv.appendChild(tooltip);
    
    // Add hover effects to data points
    const dataPoints = chartArea.querySelectorAll('.data-point');
    dataPoints.forEach(point => {
        point.addEventListener('mouseenter', (e) => {
            const value = e.target.getAttribute('data-value');
            const date = e.target.getAttribute('data-date');
            const series = e.target.getAttribute('data-series');
            
            tooltip.innerHTML = `
                <div><strong>${series || 'Value'}:</strong> ${value}</div>
                <div><strong>Date:</strong> ${date}</div>
            `;
            
            const rect = chartDiv.getBoundingClientRect();
            tooltip.style.left = (e.clientX - rect.left + 10) + 'px';
            tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
            tooltip.style.display = 'block';
        });
        
        point.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
    });
}

// Test function for development
