// Pinokio supplies managed Python; bootstrap owns the isolated app/env.
module.exports = async (kernel) => {
  const device = kernel.platform === "darwin"
    ? "mps"
    : (["linux", "win32"].includes(kernel.platform) && kernel.gpu === "nvidia" ? "cuda" : "cpu")
  return {
    requires: { bundle: "ai" },
    run: [{
      method: "shell.run",
      params: {
        path: ".",
        env: { PYTHONUNBUFFERED: "1", PIP_DISABLE_PIP_VERSION_CHECK: "1" },
        message: `python app/bootstrap.py --device ${device}`
      }
    }]
  }
}
