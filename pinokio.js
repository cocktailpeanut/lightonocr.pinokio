module.exports = {
  version: "3.7",
  title: "LightOnOCR",
  description: "Private, local image and PDF OCR with LightOnOCR-3-1B. NVIDIA, Apple Silicon and CPU support.",
  menu: async (kernel, info) => {
    const installed = info.exists("app", ".installed") && info.exists("app", "env")
    const running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    for (const [key, label] of [["update", "Updating / repairing"], ["install", "Installing"], ["reset", "Resetting"]]) {
      if (running[key]) return [{ default: true, icon: "fa-solid fa-terminal", text: label, href: `${key}.js` }]
    }
    if (running.start) {
      const local = info.local("start.js")
      if (local && /^http:\/\/127\.0\.0\.1:\d+$/.test(local.url || "")) {
        return [
          { default: true, icon: "fa-solid fa-file-lines", text: "Open LightOnOCR", href: local.url },
          { icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" }
        ]
      }
      return [{ default: true, icon: "fa-solid fa-spinner", text: "Starting LightOnOCR", href: "start.js" }]
    }
    if (!installed) return [{ default: true, icon: "fa-solid fa-download", text: "Install", href: "install.js" }]
    return [
      { default: true, icon: "fa-solid fa-play", text: "Start", href: "start.js" },
      { icon: "fa-solid fa-arrows-rotate", text: "Update / repair", href: "update.js" },
      {
        icon: "fa-solid fa-trash-can",
        text: "Reset dependencies",
        href: "reset.js",
        confirm: "Remove LightOnOCR's Python environment? Model weights and your files will be kept. Run Install again afterward."
      }
    ]
  }
}
