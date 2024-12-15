// Add smooth scrolling for navigation links
document.querySelectorAll('nav a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});

// Chat functionality
function sendMessage() {
    const userInput = document.getElementById('userInput');
    const chatBox = document.getElementById('chatBox');
    const message = userInput.value.trim();

    if (message !== '') {
        // Add user message
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'message user-message';
        userMessageDiv.textContent = message;
        chatBox.appendChild(userMessageDiv);

        // Clear input
        userInput.value = '';

        // Simulate bot response
        setTimeout(() => {
            const botMessageDiv = document.createElement('div');
            botMessageDiv.className = 'message bot-message';
            botMessageDiv.textContent = getBotResponse(message);
            chatBox.appendChild(botMessageDiv);

            // Scroll to bottom
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 1000);
    }
}

function getBotResponse(message) {
    // Simple response logic - you can expand this
    const responses = [
        "I'd be happy to help you find books on that topic!",
        "Have you tried checking our references section for more information?",
        "That's an interesting question about books. Let me help you explore that.",
        "I can recommend some great books related to your interest.",
        "Would you like me to suggest some reading materials on that subject?"
    ];
    return responses[Math.floor(Math.random() * responses.length)];
}

// Add event listener for Enter key
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
