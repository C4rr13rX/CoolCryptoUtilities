// Explicit API fixture: checks presentation, not a live model assessment.
async (page) => {
  const errors = []
  page.on('pageerror', error => errors.push(String(error)))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    let data = {}
    if (path.includes('wizard-brains')) data = { brains: [{ id: 'fixture', name: 'Development fixture', endpoint: 'http://127.0.0.1:8090', chat_path: '/brain/chat' }], selected: { chat: 'fixture' } }
    if (path.endsWith('/status/')) data = { online: true, health: { status: 'fixture' }, brain: {} }
    if (path.endsWith('/training/live/')) data = { events: [], brain: {} }
    if (path.endsWith('/learning/')) data = {
      gate: { ready: false, blockers: ['Foundation assessment has not passed.'], scope: 'Foundation first' },
      assessment_cases: { recall: 12, grounding: 24, transfer: 24, joint: 4, retention: 24 },
      manifest: { prerogatives: Array.from({ length: 12 }, (_, index) => ({ id: `P${index + 1}`, meaning: `Fixture prerogative ${index + 1}`, source: 'Fixture source' })) },
      ruview: { connected: false, error: 'No sensing server connected (fixture)' },
      queue: [{ id: 'fixture-video', kind: 'video', state: 'queued', eligible: false, blockers: ['Foundation assessment has not passed.'], payload: { title: 'Reviewed real camera clip (fixture)', rights: { license_id: 'USGS-PD', source_page: 'https://www.usgs.gov/media/videos/helicopter-overflight-kilauea-crater' } } }]
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(data) })
  })
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.reload()
  await page.locator('.learning-panel > summary').click()
  await page.getByText('Reviewed real camera clip (fixture)', { exact: true }).waitFor()
  await page.getByText('Core prerogatives', { exact: true }).click()
  if (await page.locator('.learning-panel ol > li').count() !== 12) throw new Error('Missing prerogatives')
  await page.getByRole('button', { name: 'Check RuView connection' }).click()
  await page.getByText('RuView: No sensing server connected (fixture)', { exact: true }).waitFor()
  if (!(await page.locator('.learning-panel').innerText()).includes('Cartoons, generated footage')) throw new Error('Missing real-footage policy')
  await page.screenshot({ path: '.playwright-cli/wizard-learning-panel.png' })
  await page.locator('.learning-panel > summary').click()
  const orb = await page.locator('.conversation-orb').boundingBox()
  if (!orb || Math.abs(orb.width - orb.height) > 1) throw new Error('Conversation lost circular geometry')
  if (errors.length) throw new Error(errors.join('\n'))
  return { prerogatives: 12, assessmentCases: 88, blocked: true, errors, geometry: orb }
}
