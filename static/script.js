const container = document.querySelector(".container");
const chatsContainer = document.querySelector(".chats-container");
const promptForm = document.querySelector(".prompt-form");
const promptInput = promptForm.querySelector(".prompt-input");
const themeToggleBtn = document.querySelector("#theme-toggle-btn");

let controller;
const chatHistory = [];

const isLightTheme = localStorage.getItem("themeColor") === "light_mode";
document.body.classList.toggle("light-theme", isLightTheme);
themeToggleBtn.textContent = isLightTheme ? "dark_mode" : "light_mode";

const createMessageElement = (content, ...classes) => {
    const div = document.createElement("div");
    div.classList.add("message", ...classes);
    div.innerHTML = content;
    return div;
};

const scrollToBottom = () => container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });

const addMessageToChat = (message, type) => {
    const msgDiv = createMessageElement(`<p class="message-text">${message}</p>`, `${type}-message`);
    chatsContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
};

const handleFormSubmit = async (e) => {
    e.preventDefault();
    const userMessage = promptInput.value.trim();
    if (!userMessage || document.body.classList.contains("bot-responding")) return;

    document.body.classList.add("chats-active", "bot-responding");

    // Add user message temporarily
    const userMsgDiv = addMessageToChat(userMessage, "user");

    // Prepare bot message
    const botMsgDiv = addMessageToChat("...", "bot", "loading");

    try {
        controller = new AbortController();
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage }),  // ✅ FIXED KEY HERE
            signal: controller.signal
        });

        const data = await response.json();

        // Remove user message after bot responds
        userMsgDiv.remove();

        botMsgDiv.querySelector(".message-text").textContent = data.answer || "No response.";
    } catch (error) {
        userMsgDiv.remove();

        botMsgDiv.querySelector(".message-text").textContent =
            error.name === "AbortError"
                ? "Response stopped."
                : "Error connecting to server.";
    } finally {
        botMsgDiv.classList.remove("loading");
        document.body.classList.remove("bot-responding");
        scrollToBottom();
    }

    promptInput.value = "";
};

document.querySelector("#stop-response-btn").addEventListener("click", () => {
    controller?.abort();
    document.body.classList.remove("bot-responding");
});

themeToggleBtn.addEventListener("click", () => {
    const isLight = document.body.classList.toggle("light-theme");
    localStorage.setItem("themeColor", isLight ? "light_mode" : "dark_mode");
    themeToggleBtn.textContent = isLight ? "dark_mode" : "light_mode";
});

document.querySelector("#delete-chats-btn").addEventListener("click", () => {
    chatHistory.length = 0;
    chatsContainer.innerHTML = "";
    document.body.classList.remove("chats-active", "bot-responding");
});

promptForm.addEventListener("submit", handleFormSubmit);
