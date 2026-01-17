const chatToggle = document.getElementById("chat-toggle");
const chatPopup = document.getElementById("chat-popup");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const chatIcon = document.getElementById("chat-icon");
const chatClose = document.getElementById("chat-close");

// --- Initial Greeting on Load ---
async function fetchInitialGreeting() {
    try {
        const response = await fetch("/chatbot/greeting/");
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error ${response.status}: ${errorText.substring(0, 200)}`);
        }
        const data = await response.json();
        if (data.response && Array.isArray(data.response)) {
            appendStructuredMessage("bot", data.response);
        } else {
            appendMessage("bot", "Hello! How can I help with your skin concerns?");
        }
    } catch (error) {
        appendMessage("bot", "Hello! How can I help with your skin concerns?");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    fetchInitialGreeting();
});

// Toggle chat popup visibility
chatToggle.onclick = () => {
    if (chatPopup.style.display === "none") {
        chatPopup.style.display = "flex";
        chatIcon.style.display = "none";
        chatClose.style.display = "inline";
        chatInput.focus();
        scrollToBottom();
    } else {
        chatPopup.style.display = "none";
        chatIcon.style.display = "inline";
        chatClose.style.display = "none";
    }
};

// Enable/disable send button based on input
chatInput.addEventListener("input", () => {
    chatSend.disabled = !chatInput.value.trim();
});

chatInput.addEventListener("input", async () => {
    chatSend.disabled = !chatInput.value.trim();

    const query = chatInput.value.trim();
    if (!query) return;

    const response = await fetch(`/chatbot/suggestions/?q=${encodeURIComponent(query)}`);
    const data = await response.json();

    showSuggestions(data.suggestions || []);
});

// Send message on button click
chatSend.onclick = async () => {
    const message = chatInput.value.trim();
    if (!message) return;

    appendMessage("user", message);
    chatInput.value = "";
    chatSend.disabled = true;

    const typingIndicator = createTypingIndicator();
    chatMessages.appendChild(typingIndicator);
    scrollToBottom();

    try {
        const response = await fetch("/chatbot/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({ message }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error ${response.status}: ${errorText.substring(0, 200)}`);
        }

        const data = await response.json();

        typingIndicator.remove();

        if (data.response && Array.isArray(data.response)) {
            appendStructuredMessage("bot", data.response);
        } else if (data.response) {
            appendMessage("bot", data.response);
        } else {
            appendMessage("bot", "Sorry, I didn't get that. The server response was empty or malformed.");
        }
    } catch (error) {
        typingIndicator.remove();
        appendMessage("bot", "Error connecting to the server. Please try again later.");
    }
};

// Send message on Enter key
chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !chatSend.disabled) {
        chatSend.click();
    }
});

// Helper: create typing indicator element
function createTypingIndicator() {
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "typing-indicator";
    typingIndicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    return typingIndicator;
}

// Helper to append simple text messages to chat with fade animation
function appendMessage(sender, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender}-message fade-in-message`;
    
    if (sender === "bot") {
        // Add chatbot icon container
        const iconContainer = document.createElement("div");
        iconContainer.className = "chatbot-icon-container";
        iconContainer.innerHTML = `
            <div class="chatbot-icon">
                <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#48752C"><path d="M440-690v-100q0-42 29-71t71-29h100v100q0 42-29 71t-71 29H440ZM220-450q-58 0-99-41t-41-99v-140h140q58 0 99 41t41 99v140H220ZM640-90q-39 0-74.5-12T501-135l-33 33q-11 11-28 11t-28-11q-11-11-11-28t11-28l33-33q-21-29-33-64.5T400-330q0-100 70-170.5T640-571h241v241q0 100-70.5 170T640-90Zm0-80q67 0 113-47t46-113v-160H640q-66 0-113 46.5T480-330q0 23 5.5 43.5T502-248l110-110q11-11 28-11t28 11q11 11 11 28t-11 28L558-192q18 11 38.5 16.5T640-170Zm1-161Z"/>
                </svg>
            </div>
        `;
        msgDiv.appendChild(iconContainer);
    }
    
    const textContainer = document.createElement("div");
    textContainer.className = "message-text";
    textContainer.textContent = text;
    msgDiv.appendChild(textContainer);
    
    chatMessages.appendChild(msgDiv);

    if (sender === "user") {
        scrollToBottom();
    } else if (sender === "bot") {
        msgDiv.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

// Helper to append structured messages (headings, paragraphs, lists, buttons) with fade animation
function appendStructuredMessage(sender, contentBlocks) {
    const msgContainer = document.createElement("div");
    msgContainer.className = `message ${sender}-message fade-in-message`;
    
    if (sender === "bot") {
        const iconContainer = document.createElement("div");
        iconContainer.className = "chatbot-icon-container";
        iconContainer.innerHTML = `
            <div class="chatbot-icon">
                <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#48752C">
                    <path d="M440-690v-100q0-42 29-71t71-29h100v100q0 42-29 71t-71 29H440ZM220-450q-58 0-99-41t-41-99v-140h140q58 0 99 41t41 99v140H220ZM640-90q-39 0-74.5-12T501-135l-33 33q-11 11-28 11t-28-11q-11-11-11-28t11-28l33-33q-21-29-33-64.5T400-330q0-100 70-170.5T640-571h241v241q0 100-70.5 170T640-90Zm0-80q67 0 113-47t46-113v-160H640q-66 0-113 46.5T480-330q0 23 5.5 43.5T502-248l110-110q11-11 28-11t28 11q11 11 11 28t-11 28L558-192q18 11 38.5 16.5T640-170Zm1-161Z"/>
                </svg>
            </div>
        `;
        msgContainer.appendChild(iconContainer);
    }

    const contentContainer = document.createElement("div");
    contentContainer.className = "message-content";

    let currentList = null;  // To collect listItems in one <ul>

    contentBlocks.forEach(block => {
        let element;
        const safeText = block.text || '';

        switch (block.type) {
            case "heading":
                if (currentList) {
                    contentContainer.appendChild(currentList);
                    currentList = null;
                }
                element = document.createElement(`h${block.level || 1}`);
                element.textContent = safeText;
                contentContainer.appendChild(element);
                break;

            case "paragraph":
                if (currentList) {
                    contentContainer.appendChild(currentList);
                    currentList = null;
                }
                element = document.createElement("p");
                element.textContent = safeText;
                contentContainer.appendChild(element);
                break;

            case "listItem":
                if (!currentList) {
                    currentList = document.createElement("ul");
                    currentList.className = "bot-list";
                }
                element = document.createElement("li");
                if (block.icon) {
                    const icon = document.createElement("i");
                    icon.className = `${block.icon} message-icon`;
                    element.appendChild(icon);
                    element.appendChild(document.createTextNode(' '));
                }
                element.appendChild(document.createTextNode(safeText));
                currentList.appendChild(element);
                break;

            case "tip":
                if (currentList) {
                    contentContainer.appendChild(currentList);
                    currentList = null;
                }
                element = document.createElement("div");
                element.className = "bot-tip";
                const infoIcon = document.createElement("i");
                infoIcon.className = "fas fa-info-circle message-icon";
                element.appendChild(infoIcon);
                element.appendChild(document.createTextNode(' '));
                element.appendChild(document.createTextNode(safeText));
                contentContainer.appendChild(element);
                break;

            case "sectionHeading":
                if (currentList) {
                    contentContainer.appendChild(currentList);
                    currentList = null;
                }
                element = document.createElement("h3");
                element.textContent = safeText;
                contentContainer.appendChild(element);
                break;

            case "button_group":
                if (currentList) {
                    contentContainer.appendChild(currentList);
                    currentList = null;
                }
                element = document.createElement("div");
                element.className = "button-group";

                if (Array.isArray(block.buttons)) {
                    block.buttons.forEach(buttonData => {
                        const button = document.createElement("button");
                        button.className = "chatbot-button";
                        button.textContent = buttonData.text;

                        if (buttonData.icon) {
                            const icon = document.createElement("i");
                            icon.className = `${buttonData.icon} button-icon`;
                            button.prepend(icon);
                            button.prepend(document.createTextNode(' '));
                        }

                        button.addEventListener("click", () => {
                            appendMessage("user", buttonData.text);
                            sendMessagePayload(buttonData.payload || buttonData.text);
                        });

                        element.appendChild(button);
                    });
                }
                contentContainer.appendChild(element);
                break;

            default:
                if (currentList) {
                    contentContainer.appendChild(currentList);
                    currentList = null;
                }
                console.warn("Unknown content block type:", block.type, block);
                element = document.createElement("p");
                element.textContent = safeText;
                contentContainer.appendChild(element);
        }
    });

    if (currentList) {
        contentContainer.appendChild(currentList);
    }

    msgContainer.appendChild(contentContainer);
    chatMessages.appendChild(msgContainer);

    if (sender === "user") {
        scrollToBottom();
    } else if (sender === "bot") {
        msgContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

// Helper to send message payload silently (used for button payloads)
async function sendMessagePayload(payload) {
    const typingIndicator = createTypingIndicator();
    chatMessages.appendChild(typingIndicator);
    scrollToBottom();

    try {
        const response = await fetch("/chatbot/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({ message: payload }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error ${response.status}: ${errorText.substring(0, 200)}`);
        }

        const data = await response.json();

        typingIndicator.remove();

        if (data.response && Array.isArray(data.response)) {
            appendStructuredMessage("bot", data.response);
        } else if (data.response) {
            appendMessage("bot", data.response);
        } else {
            appendMessage("bot", "Sorry, I didn't get that. The server response was empty or malformed.");
        }
    } catch (error) {
        typingIndicator.remove();
        appendMessage("bot", "Error connecting to the server. Please try again later.");
    }
}

// Scroll chat to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Get CSRF token helper
function getCookie(name) {
    const cookies = document.cookie.split(";");
    for (let cookie of cookies) {
        const [key, value] = cookie.trim().split("=");
        if (key === name) return decodeURIComponent(value);
    }
    return "";
}

// Store the attention interval
let attentionInterval;

// Function to start attracting attention
function startAttractingAttention() {
    const toggle = document.getElementById('chat-toggle');
    
    // Add attention class
    toggle.classList.add('attention-mode');
    
    // Optional: Add actual device vibration if supported
    if (navigator.vibrate) {
        attentionInterval = setInterval(() => {
            navigator.vibrate([100, 50, 100]);
        }, 1000);
    }
}

// Function to stop attracting attention
function stopAttractingAttention() {
    const toggle = document.getElementById('chat-toggle');
    toggle.classList.remove('attention-mode');
    
    // Clear vibration interval if it exists
    if (attentionInterval) {
        clearInterval(attentionInterval);
    }
    
    // Stop any ongoing vibration
    if (navigator.vibrate) {
        navigator.vibrate(0);
    }
}

// Example usage:
// Start vibration (call this when you want to attract user)
// startAttractingAttention();

// Stop vibration (call this when user interacts or after timeout)
// stopAttractingAttention();

// Toggle vibration on hover (optional)
document.getElementById('chat-toggle').addEventListener('mouseenter', function() {
    this.style.animation = 'vibrate 0.3s linear infinite, subtle-float 3s ease-in-out infinite';
});

document.getElementById('chat-toggle').addEventListener('mouseleave', function() {
    this.style.animation = 'subtle-float 3s ease-in-out infinite';
});

// Function to add welcome message
function addWelcomeMessage() {
    const chatMessages = document.getElementById('chat-messages');
    
    const welcomeMsg = document.createElement('div');
    welcomeMsg.className = 'bot-message welcome-message';
    welcomeMsg.innerHTML = `
        <div class="welcome-title">Hi there! 👋</div>
        <div class="welcome-text">
            Welcome to your Personal Skin Care Companion!<br>
            I can help you understand various skin issues.<br>
            Which skin condition are you interested in?
        </div>
        <div class="welcome-actions">
            <button class="welcome-button" data-topic="acne">Acne</button>
            <button class="welcome-button" data-topic="dryness">Dryness</button>
            <button class="welcome-button" data-topic="aging">Aging</button>
            <button class="welcome-button" data-topic="sensitivity">Sensitivity</button>
        </div>
    `;
    
    chatMessages.appendChild(welcomeMsg);
    
    // Add click handlers for the buttons
    welcomeMsg.querySelectorAll('.welcome-button').forEach(button => {
        button.addEventListener('click', function() {
            const topic = this.getAttribute('data-topic');
            sendUserMessage(`I'm interested in ${topic}`);
        });
    });
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Call this when the chat opens
// Example: Inside your chat toggle handler
document.getElementById('chat-toggle').addEventListener('click', function() {
    if (chatPopup.classList.contains('hidden')) {
        // Chat is opening
        setTimeout(addWelcomeMessage, 500); // Add slight delay for smoothness
    }
});

// Helper function (you should already have something similar)
function sendUserMessage(message) {
    // Your existing code to handle user messages
    console.log('User selected:', message);
    // Add your logic to process the message and get bot response
}