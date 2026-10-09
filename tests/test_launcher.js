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
  assert.ok(launch.run[1].params.message.startsWith('"env/bin/python" server.py --port 12345 --model '))
  assert.ok(launch.run[1].params.message.includes("['0.8B', '1B', '4B'].includes(args.model)"))
  assert.ok(launch.run[1].params.message.includes("args.model : 'invalid'"))
  assert.equal(launch.run[1].params.env.LIGHTONOCR_DEVICE, undefined)
  const windows = await start({platform: "win32", port: async () => 12345})
  assert.ok(windows.run[1].params.message.startsWith('"env\\Scripts\\python.exe" server.py --port 12345 --model '))
  assert.equal(launch.run[0].params.url, null)
  const encoded = launch.run[1].params.on[0].event
  const event = new RegExp(encoded.slice(1, -1)).exec('LIGHTONOCR_READY http://127.0.0.1:12345')
  assert.equal(event[1], 'http://127.0.0.1:12345')
  function info (installed, running, url) {
    return {exists: () => installed, running: name => name === running, local: () => ({url})}
  }
  assert.equal((await menu({}, info(false, null)))[0].href, 'install.js')
  const choices = await menu({}, info(true, null))
  const expected = [['0.8B', 'start-08b.js'], ['1B', 'start-1b.js'], ['4B', 'start-4b.js']]
  assert.deepEqual(choices.slice(0,3).map(item => [item.text,item.href]),
    expected.map(([model, script]) => [`Start ${model}`, script]))
  // Pinokio's native hard tabs reuse frames by pathname, and its callbacks
  // resolve the first matching target. Both identities must distinguish models.
  assert.equal(new Set(choices.slice(0,3).map(item => item.href)).size, 3)
  assert.equal(new Set(choices.slice(0,3).map(item => new URL(item.href, "http://localhost/api/lightonocr.pinokio/").pathname)).size, 3)
  for (const model of ['4B', '0.8B', '1B', '4B']) {
    const [variant, script] = expected.find(([variant]) => variant === model)
    const entry = choices.find(item => item.href === script)
    assert.equal(entry.params, undefined, 'native model selection must not depend on retained query arguments')
    const runner = require('../' + script)
    assert.equal(runner.constructor.name, 'AsyncFunction', 'Pinokio only awaits explicitly async runners')
    const launch = await runner({platform: 'darwin', port: async () => 12345})
    assert.equal(launch.run[0].params.model, variant)
    assert.equal(launch.run[1].params.message, `"env/bin/python" server.py --port 12345 --model ${variant}`)
    assert.equal((await menu({}, info(true, script)))[0].href, script)
    assert.equal((await menu({}, info(true, script, 'http://127.0.0.1:12345')))[1].href, script)
  }
  assert.deepEqual(await start({port: async () => 12345}, {platform: 'darwin'}), await start({port: async () => 12345}))
  assert.equal(choices.some(item => item.default === true), false, 'installation must not autostart any model')
  for (const task of ['install.js', 'update.js', 'reset.js']) {
    const items = await menu({}, info(true, task)); assert.equal(items.length, 1); assert.equal(items[0].href, task)
  }
  assert.equal((await menu({}, info(true, 'start.js')))[0].href, 'start.js')
  assert.equal((await menu({}, info(true, 'start.js', 'http://127.0.0.1:12345')))[0].href, 'http://127.0.0.1:12345')
  assert.equal((await menu({}, info(true, null, 'http://127.0.0.1:12345'))).some(item => item.default === true), false)
  assert.deepEqual(require('../reset.js').run.map(step => step.params.path), ['app/.installed', 'app/env'])
  console.log('LAUNCHER_CONTRACTS_PASS')
}
main().catch(error => { console.error(error); process.exit(1) })
