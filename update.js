// Private/local copies without an origin still support dependency repair.
// A configured origin is only fast-forwarded; local edits are never reset.
module.exports = {
  requires: { bundle: "ai" },
  run: [
    {
      when: "{{exists('.git')}}",
      method: "shell.run",
      params: {
        path: ".",
        message: "python -c \"import shutil, subprocess; g = shutil.which('git'); r = subprocess.run([g, 'remote', 'get-url', 'origin'], capture_output=True, text=True) if g else None; subprocess.run([g, 'pull', '--ff-only'], check=True) if r and r.returncode == 0 and r.stdout.strip() else print('No Git origin configured. Repairing the current local version.')\""
      }
    },
    { method: "script.start", params: { uri: "install.js" } }
  ]
}
