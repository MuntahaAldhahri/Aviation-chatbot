const container = document.querySelector(".container");
const chatsContainer = document.querySelector(".chats-container");
const promptForm = document.querySelector(".prompt-form");
const promptInput = promptForm.querySelector(".prompt-input");
const themeToggleBtn = document.querySelector("#theme-toggle-btn");

const AZURE_API_URL = "https://sans.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview";
const AZURE_API_KEY = "EYEl4NjklDJy24f3QfixOeOhuXOr8mdYAFE8XsC1Umn7DEAcQXJoJQQJ99BFACYeBjFXJ3w3AAABACOGBbcl";

let controller, typingInterval;
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

const typingEffect = (text, textElement, botMsgDiv) => {
  textElement.textContent = "";
  const words = text.split(" ");
  let wordIndex = 0;
  typingInterval = setInterval(() => {
    if (wordIndex < words.length) {
      textElement.textContent += (wordIndex === 0 ? "" : " ") + words[wordIndex++];
      scrollToBottom();
    } else {
      clearInterval(typingInterval);
      botMsgDiv.classList.remove("loading");
      document.body.classList.remove("bot-responding");
    }
  }, 40);
};

const generateResponse = async (botMsgDiv, userMessage) => {
  const textElement = botMsgDiv.querySelector(".message-text");
  controller = new AbortController();

  chatHistory.push({ role: "user", content: userMessage });

  try {
    const response = await fetch(AZURE_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
      },
      body: JSON.stringify({
        messages: chatHistory,
        max_tokens: 1000
      }),
      signal: controller.signal
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error.message);

    const responseText = data.choices[0].message.content.trim();
    typingEffect(responseText, textElement, botMsgDiv);

    chatHistory.push({ role: "assistant", content: responseText });
  } catch (error) {
    textElement.textContent = error.name === "AbortError" ? "Response generation stopped." : error.message;
    textElement.style.color = "#d62939";
    botMsgDiv.classList.remove("loading");
    document.body.classList.remove("bot-responding");
    scrollToBottom();
  }
};

const handleFormSubmit = (e) => {
  e.preventDefault();
  const userMessage = promptInput.value.trim();
  if (!userMessage || document.body.classList.contains("bot-responding")) return;

  document.body.classList.add("chats-active", "bot-responding");

  const userMsgDiv = createMessageElement(`<p class="message-text">${userMessage}</p>`, "user-message");
  chatsContainer.appendChild(userMsgDiv);
  scrollToBottom();

  setTimeout(() => {
    const botMsgHTML = `<img class="avatar" src="sans.svg" /> <p class="message-text">Just a sec...</p>`;
    const botMsgDiv = createMessageElement(botMsgHTML, "bot-message", "loading");
    chatsContainer.appendChild(botMsgDiv);
    scrollToBottom();
    generateResponse(botMsgDiv, userMessage);
  }, 600);
};

document.querySelector("#stop-response-btn").addEventListener("click", () => {
  controller?.abort();
  clearInterval(typingInterval);
  const loadingBotMsg = chatsContainer.querySelector(".bot-message.loading");
  if (loadingBotMsg) loadingBotMsg.classList.remove("loading");
  document.body.classList.remove("bot-responding");
});

themeToggleBtn.addEventListener("click", () => {
  const isLightTheme = document.body.classList.toggle("light-theme");
  localStorage.setItem("themeColor", isLightTheme ? "light_mode" : "dark_mode");
  themeToggleBtn.textContent = isLightTheme ? "dark_mode" : "light_mode";
});

document.querySelector("#delete-chats-btn").addEventListener("click", () => {
  chatHistory.length = 0;
  chatsContainer.innerHTML = "";
  document.body.classList.remove("chats-active", "bot-responding");
});

promptForm.addEventListener("submit", handleFormSubmit);
