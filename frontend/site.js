const aiBtn = document.getElementById("ai-btn");
const aiPanel = document.getElementById("ai-panel");
const aiFrame = document.getElementById("ai-frame");
const aiMin = document.getElementById("ai-min");

function setOpen(open) {
  if (open && !aiFrame.getAttribute("src")) aiFrame.setAttribute("src", "assistant.html");
  aiPanel.hidden = !open;
  aiBtn.setAttribute("aria-expanded", String(open));
  aiBtn.setAttribute("aria-label", open ? "Close College AI Assistant" : "Open College AI Assistant");
  if (open) aiFrame.focus();
}

aiBtn.addEventListener("click", () => setOpen(aiPanel.hidden));
aiMin.addEventListener("click", () => { setOpen(false); aiBtn.focus(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !aiPanel.hidden) { setOpen(false); aiBtn.focus(); }
});

window.addEventListener("message", (e) => {
  if (e.origin === window.location.origin && e.source === aiFrame.contentWindow
      && e.data && e.data.type === "close-assistant" && !aiPanel.hidden) {
    setOpen(false);
    aiBtn.focus();
  }
});
