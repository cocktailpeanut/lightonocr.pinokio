module.exports = {
  version: "3.7",
  title: "LightOnOCR 3",
  icon: "icon.png",
  description: "Private, local image and PDF OCR with your choice of LightOnOCR-3 0.8B, 1B or 4B. NVIDIA, Apple Silicon and CPU support.",
  menu: async (kernel, info) => {
    const installed = info.exists("app", ".installed") && info.exists("app", "env")
    const choices = [["0.8B", "start-08b.js"], ["1B", "start-1b.js"], ["4B", "start-4b.js"]]
    const activeStart = ["start.js", ...choices.map(([, script]) => script)].find(script => info.running(script))
    const running = {
      install: info.running("install.js"),
      start: activeStart,
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    for (const [key, label] of [["update", "Updating / repairing"], ["install", "Installing"], ["reset", "Resetting"]]) {
      if (running[key]) return [{ default: true, icon: "fa-solid fa-terminal", text: label, href: `${key}.js` }]
    }
    if (running.start) {
      const local = info.local(activeStart)
      if (local && /^http:\/\/127\.0\.0\.1:\d+$/.test(local.url || "")) {
        return [
          { default: true, icon: "fa-solid fa-file-lines", text: "Open LightOnOCR", href: local.url },
          { icon: "fa-solid fa-terminal", text: "Terminal", href: activeStart }
        ]
      }
      return [{ default: true, icon: "fa-solid fa-spinner", text: "Starting LightOnOCR", href: activeStart }]
    }
    if (!installed) return [{ default: true, icon: "fa-solid fa-download", text: "Install", href: "install.js" }]
    return [
      // Distinct script paths prevent Pinokio from reusing another model's
      // terminal or query arguments. Each entry fixes its model explicitly.
      ...choices.map(([model, script]) => ({
        mode: "refresh",
        icon: "fa-solid fa-play", text: `Start ${model}`, href: script
      })),
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
