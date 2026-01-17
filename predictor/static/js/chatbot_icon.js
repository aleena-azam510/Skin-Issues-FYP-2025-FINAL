document.addEventListener('DOMContentLoaded', function() {
    const chatToggle = document.getElementById('chat-toggle');
    const chatPopup = document.getElementById('chat-popup');
    const chatIcon = document.getElementById('chat-icon');
    const chatClose = document.getElementById('chat-close');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    
    // Sample responses for the chatbot
    const botResponses = {
        "hello": "Hello! I'm your skin care assistant. How can I help you today?",
        "acne": "For acne concerns, I recommend: 1) Gentle cleansing twice daily 2) Non-comedogenic products 3) Seeing a dermatologist for persistent cases.",
        "dry skin": "For dry skin: 1) Use a hydrating cleanser 2) Apply moisturizer on damp skin 3) Consider a humidifier 4) Drink plenty of water.",
        "wrinkles": "For wrinkles: 1) Use sunscreen daily 2) Try retinol products 3) Stay hydrated 4) Consider professional treatments for deeper wrinkles.",
        "default": "I'm sorry, I didn't understand that. Could you ask about specific skin concerns like acne, dry skin, or wrinkles?"
    };
    
    // Toggle chat visibility
    chatToggle.addEventListener('click', function() {
        if (chatPopup.style.display === 'none') {
            chatPopup.style.display = 'flex';
            setTimeout(() => {
                chatPopup.classList.add('show');
            }, 10);
            chatIcon.style.display = 'none';
            chatClose.style.display = 'inline';
            chatInput.focus();
            
            // Add welcome message if chat is empty
            if (chatMessages.children.length === 0) {
                addBotMessage("Hello! I'm your skin care assistant. Ask me about acne, dry skin, wrinkles, or other skin concerns.");
            }
        } else {
            chatPopup.classList.remove('show');
            setTimeout(() => {
                chatPopup.style.display = 'none';
            }, 300);
            chatIcon.style.display = 'inline';
            chatClose.style.display = 'none';
        }
    });
    
    // Send message when clicking send button
    chatSend.addEventListener('click', sendMessage);
    
    // Send message when pressing Enter
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = chatInput.value.trim();
        if (message === '') return;
        
        // Add user message
        addUserMessage(message);
        chatInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Simulate bot thinking
        setTimeout(() => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Generate and add bot response
            const response = generateBotResponse(message);
            addBotMessage(response);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 1000);
    }
    
    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addBotMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    function generateBotResponse(userMessage) {
        const lowerCaseMessage = userMessage.toLowerCase();
        
        if (lowerCaseMessage.includes('hello') || lowerCaseMessage.includes('hi')) {
            return botResponses["hello"];
        } else if (lowerCaseMessage.includes('acne')) {
            return botResponses["acne"];
        } else if (lowerCaseMessage.includes('dry') || lowerCaseMessage.includes('flaky')) {
            return botResponses["dry skin"];
        } else if (lowerCaseMessage.includes('wrinkle') || lowerCaseMessage.includes('aging')) {
            return botResponses["wrinkles"];
        } else {
            return botResponses["default"];
        }
    }
    
    // Initial welcome message when page loads
    setTimeout(() => {
        if (chatMessages.children.length === 0) {
            addBotMessage("Hello! I'm your skin care assistant. Ask me about acne, dry skin, wrinkles, or other skin concerns.");
        }
    }, 1500);
});