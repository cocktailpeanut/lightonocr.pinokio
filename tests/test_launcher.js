// Static launcher-contract tests; no OS or accelerator inference claim.
const assert = require('node:assert/strict')
const install = require('../install.js')
const start = require('../start.js')
const menu = require('../pinokio.js').menu
async function main () {
  for (const [platform, gpu, device] of [['linux','nvidia','cuda'],['win32','nvidia','cuda'],['darwin','apple','mps'],['linux','amd','cpu'],['win32',null,'cpu']]) {
    const script = await install({platform, gpu})
    assert.equal(script.run[0].params.message, `python app/bootstrap.py --device ${device}`)
  }
  const launch = await start({port: async () => 12345})
  assert.equal(launch.daemon, true)
  assert.equal(launch.run[1].params.message, '"env/bin/python" server.py --port 12345')
  assert.equal(launch.run[1].params.env.LIGHTONOCR_DEVICE, undefined)
  const windows = await start({platform: "win32", port: async () => 12345})
  assert.equal(windows.run[1].params.message, '"env\\Scripts\\python.exe" server.py --port 12345')
  assert.equal(launch.run[0].params.url, null)
  const encoded = launch.run[1].params.on[0].event
  const event = new RegExp(encoded.slice(1, -1)).exec('LIGHTONOCR_READY http://127.0.0.1:12345')
  assert.equal(event[1], 'http://127.0.0.1:12345')
  function info (installed, running, url) {
    return {exists: () => installed, running: name => name === running, local: () => ({url})}
  }
  assert.equal((await menu({}, info(false, null)))[0].href, 'install.js')
  assert.equal((await menu({}, info(true, null)))[0].href, 'start.js')
  for (const task of ['install.js', 'update.js', 'reset.js']) {
    const items = await menu({}, info(true, task)); assert.equal(items.length, 1); assert.equal(items[0].href, task)
  }
  assert.equal((await menu({}, info(true, 'start.js')))[0].href, 'start.js')
  assert.equal((await menu({}, info(true, 'start.js', 'http://127.0.0.1:12345')))[0].href, 'http://127.0.0.1:12345')
  assert.equal((await menu({}, info(true, null, 'http://127.0.0.1:12345')))[0].href, 'start.js')
  assert.deepEqual(require('../reset.js').run.map(step => step.params.path), ['app/.installed', 'app/env'])
  console.log('LAUNCHER_CONTRACTS_PASS')
}
main().catch(error => { console.error(error); process.exit(1) })
