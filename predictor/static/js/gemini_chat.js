document.addEventListener('DOMContentLoaded', () => {
    const chatToggle = document.getElementById('chat-toggle');
    const chatPopup = document.getElementById('chat-popup');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send');
    const chatMessages = document.getElementById('chat-messages');
    const chatIcon = document.getElementById('chat-icon');
    const chatClose = document.getElementById('chat-close');

    chatToggle.addEventListener('click', () => {
        const isPopupOpen = chatPopup.classList.toggle('open');
        if (isPopupOpen) {
            chatIcon.style.display = 'none';
            chatClose.style.display = 'inline';
        } else {
            chatIcon.style.display = 'inline';
            chatClose.style.display = 'none';
        }
    });

    chatInput.addEventListener('input', () => {
        chatSendBtn.disabled = chatInput.value.trim() === '';
    });

    chatSendBtn.addEventListener('click', () => {
        sendMessage(chatInput.value.trim());
    });

    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !chatSendBtn.disabled) {
            sendMessage(chatInput.value.trim());
        }
    });

    async function sendMessage(userMessage) {
        if (!userMessage) return;

        displayMessage(userMessage, 'user');
        chatInput.value = '';
        chatSendBtn.disabled = true;

        try {
            const response = await fetch('/api/chatbot_api_gemini/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }),
            });

            const data = await response.json();
            const botResponse = data.response;
            const botButtons = data.buttons || [];

            displayMessage(botResponse, 'bot', botButtons);

        } catch (error) {
            console.error('Error fetching data:', error);
            displayMessage("Sorry, I'm having trouble connecting right now. Please try again later.", 'bot');
        }
    }

    function displayMessage(message, sender, buttons = []) {
    const messageElement = document.createElement('div');
    // Use a consistent class name for message bubbles
    messageElement.classList.add('chat-message');
    
    // Add specific class for user or bot styling
    if (sender === 'user') {
        messageElement.classList.add('user-message');
    } else {
        messageElement.classList.add('bot-message');
    }
    
    // Create a paragraph element for the text content
    const messageText = document.createElement('p');
    messageText.innerHTML = message; // Use innerHTML to render Markdown

    messageElement.appendChild(messageText);

    if (buttons.length > 0) {
        const buttonGroup = document.createElement('div');
        buttonGroup.classList.add('button-group');
        
        buttons.forEach(buttonText => {
            const button = document.createElement('button');
            button.classList.add('chatbot-button');
            button.textContent = buttonText;
            button.addEventListener('click', () => {
                sendMessage(buttonText);
            });
            buttonGroup.appendChild(button);
        });
        messageElement.appendChild(buttonGroup); // Append the button group after the text
    }
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
});