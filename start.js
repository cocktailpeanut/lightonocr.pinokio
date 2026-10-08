module.exports = async (kernel) => {
  const port = await kernel.port()
  return {
    daemon: true,
    run: [
      {
        method: "shell.run",
        params: {
          path: "app",
          conda: { skip: true },
          venv: "env",
          env: {
            PYTHONUNBUFFERED: "1",
            LIGHTONOCR_DEVICE: "auto",
            PYTORCH_ENABLE_MPS_FALLBACK: "1",
            GRADIO_ANALYTICS_ENABLED: "False"
          },
          message: `python server.py --port ${port}`,
          on: [{ event: "/LIGHTONOCR_READY (http:\\/\\/127\\.0\\.0\\.1:[0-9]+)/", done: true }]
        }
      },
      { method: "local.set", params: { url: "{{input.event[1]}}" } }
    ]
  }
}
