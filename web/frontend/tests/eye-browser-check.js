// Run with: playwright-cli run-code --filename tests/eye-browser-check.js
async (page) => {
  const errors = []
  const audioFixture = await page.evaluate(() => {
    const samples = 16000, buffer = new ArrayBuffer(44 + samples * 2), view = new DataView(buffer)
    const text = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)))
    text(0,'RIFF'); view.setUint32(4,36 + samples * 2,true); text(8,'WAVE'); text(12,'fmt ')
    view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true)
    view.setUint32(24,8000,true); view.setUint32(28,16000,true); view.setUint16(32,2,true); view.setUint16(34,16,true)
    text(36,'data'); view.setUint32(40,samples * 2,true)
    for (let i=0;i<samples;i++) view.setInt16(44+i*2,Math.sin(i*Math.PI*2*220/8000)*12000,true)
    return btoa(String.fromCharCode(...new Uint8Array(buffer)))
  })
  page.on('pageerror', error => errors.push(String(error)))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    let data = {}
    if (path.includes('wizard-brains')) data = { brains: [{ id: 'fixture', name: 'Development fixture', endpoint: 'http://127.0.0.1:8090', chat_path: '/brain/chat' }], selected: { chat: 'fixture' } }
    if (path.endsWith('/status/')) data = { online: true, health: { status: 'fixture' }, brain: {} }
    if (path.endsWith('/training/live/')) data = { events: [], brain: {} }
    if (path.endsWith('/message/')) {
      const text = route.request().postDataJSON().text
      data = { answer: 'The moon rests above a silver ocean.', concepts: ['txt:word_moon'], node_online: true, confidence_tier: 'high' }
      if (text.includes('vector')) {
        data.eye_signal = { version: 1, source: 'model-vector', coordinate_space: 'fixture-only-v1', timing: 'model', concepts: [], frames: [{ at_ms: 0, vector: [0,1,0,1], word_index: 0 }, { at_ms: 800, vector: [1,0,-1,0], word_index: 3 }] }
        data.audio_b64 = audioFixture
      }
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(data) })
  })
  await page.addInitScript(() => {
    window.fixtureStreams = []
    const capture = () => {
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 360
      const context = canvas.getContext('2d'); context.fillStyle = 'white'; context.fillRect(0,0,640,360)
      context.fillStyle = 'black'; context.font = '24px monospace'; context.fillText('Fixture screen text', 20, 80)
      const stream = canvas.captureStream(2); window.fixtureStreams.push(stream); return Promise.resolve(stream)
    }
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { configurable: true, value: capture })
    Object.defineProperty(navigator.mediaDevices, 'getDisplayMedia', { configurable: true, value: capture })
  })
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.reload()
  await page.locator('.observation summary').waitFor()
  const geometry = await page.locator('.conversation-orb').boundingBox()
  if (!geometry || Math.abs(geometry.width - geometry.height) > 1) throw new Error('Conversation is not circular')
  await page.getByText('Teach the eye a silhouette', { exact: true }).click()
  const file = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 40; canvas.height = 40
    const context = canvas.getContext('2d'); context.fillStyle = 'black'; context.beginPath(); context.arc(20,20,14,0,Math.PI*2); context.fill()
    return canvas.toDataURL('image/png')
  })
  // A data URL is decoded by the browser itself, then uploaded as a real File.
  await page.locator('.silhouette-studio input[type=file]').evaluate(async (input, url) => {
    const blob = await (await fetch(url)).blob()
    const transfer = new DataTransfer(); transfer.items.add(new File([blob], 'moon.png', { type: 'image/png' }))
    input.files = transfer.files; input.dispatchEvent(new Event('change', { bubbles: true }))
  }, file)
  await page.getByRole('img', { name: 'Silhouette to review' }).waitFor()
  await page.getByPlaceholder('e.g. moon').fill('moon')
  await page.getByRole('button', { name: 'Save reviewed association' }).click()
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('wizard-eye-associations-v1') || '[]'))
  if (saved[0]?.concept !== 'moon' || !saved[0].silhouette.path.includes('M')) throw new Error('Confirmed silhouette was not saved')
  await page.getByText('Teach the eye a silhouette', { exact: true }).click()
  const input = page.locator('textarea.chat-input')
  await input.fill('moon'); await input.press('Enter')
  await page.getByText('Recalled silhouette · moon', { exact: true }).waitFor()
  await page.screenshot({ path: '.playwright-cli/wizard-eye-silhouette.png' })
  await input.fill('vector'); await input.press('Enter')
  await page.getByText('Model vector trajectory', { exact: true }).waitFor()
  await page.locator('audio').evaluate(audio => audio.play())
  await page.waitForFunction(() => {
    const canvas = document.querySelector('.audio-viz-overlay')
    return canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data.some((value,index) => index % 4 === 3 && value > 0)
  })
  await page.locator('audio').evaluate(audio => audio.pause())
  await page.locator('.observation summary').click()
  await page.getByRole('button', { name: 'Enable webcam', exact: true }).click()
  await page.getByRole('button', { name: 'Share screen', exact: true }).click()
  await page.waitForFunction(() => [...document.querySelectorAll('.observation canvas')].every(canvas => canvas.width >= 160))
  await page.getByLabel('Observe app interaction activity').check()
  await input.fill('private text is not recorded')
  await input.press('a')
  await page.evaluate(() => { const input = document.createElement('input'); input.id = 'fixture-password'; input.type = 'password'; document.querySelector('.input-bar').append(input) })
  await page.locator('#fixture-password').press('b')
  if (!await page.getByText('1 key events', { exact: false }).isVisible()) throw new Error('Sensitive-field activity was counted')
  await page.locator('#fixture-password').evaluate(input => input.remove())
  await page.getByRole('button', { name: 'Stop observation', exact: true }).click()
  const released = await page.evaluate(() => window.fixtureStreams.length === 2 && window.fixtureStreams.every(stream => stream.getTracks().every(track => track.readyState === 'ended')))
  if (!released) throw new Error('Stop did not release both capture streams')
  await page.locator('.observation summary').click()
  await page.screenshot({ path: '.playwright-cli/wizard-eye-vector.png' })
  await page.setViewportSize({ width: 390, height: 844 })
  const mobile = await page.locator('.conversation-orb').boundingBox()
  if (!mobile || Math.abs(mobile.width - mobile.height) > 1 || mobile.x < -1 || mobile.x + mobile.width > 391) throw new Error('Mobile circle does not fit')
  await page.screenshot({ path: '.playwright-cli/wizard-eye-mobile.png' })
  await page.evaluate(() => window.unmountEyePreview())
  if (errors.length) throw new Error(errors.join('\n'))
  console.log('PASS: circle, reviewed silhouette recall, vector/audio response, sensitive-field exclusion, paired capture cleanup, mobile fit, unmount; fixtures only.')
}
