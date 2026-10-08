module.exports = async (kernel) => {
  const port = await kernel.port()
  const python = kernel.platform === "win32" ? "env\\Scripts\\python.exe" : "env/bin/python"
  return {
    daemon: true,
    run: [
      { method: "local.set", params: { url: null } },
      {
        method: "shell.run",
        params: {
          path: "app",
          conda: { skip: true },
          env: {
            PYTHONUNBUFFERED: "1",
            PYTORCH_ENABLE_MPS_FALLBACK: "1",
          },
          message: `"${python}" server.py --port ${port}`,
          on: [{ event: "/LIGHTONOCR_READY (http:\\/\\/127\\.0\\.0\\.1:[0-9]+)/", done: true }]
        }
      },
      { when: "{{input.event && input.event[1]}}", method: "local.set", params: { url: "{{input.event[1]}}" } }
    ]
  }
}
