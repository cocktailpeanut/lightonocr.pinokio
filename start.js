// Pinokio supplies kernel.info as the second runner argument.
// Model-specific entry points bind their model through the third argument.
module.exports = async (kernel, info, selectedModel) => {
  const port = await kernel.port()
  const python = kernel.platform === "win32" ? "env\\Scripts\\python.exe" : "env/bin/python"
  const model = selectedModel || "{{['0.8B', '1B', '4B'].includes(args.model) ? args.model : 'invalid'}}"
  return {
    daemon: true,
    run: [
      { method: "local.set", params: { url: null, model } },
      {
        method: "shell.run",
        params: {
          path: "app",
          conda: { skip: true },
          env: {
            PYTHONUNBUFFERED: "1",
            PYTORCH_ENABLE_MPS_FALLBACK: "1",
          },
          message: `"${python}" server.py --port ${port} --model ${model}`,
          on: [{ event: "/LIGHTONOCR_READY (http:\\/\\/127\\.0\\.0\\.1:[0-9]+)/", done: true }]
        }
      },
      { when: "{{input.event && input.event[1]}}", method: "local.set", params: { url: "{{input.event[1]}}" } }
    ]
  }
}
