async (page) => {
  const errors = []; page.on('pageerror', error => errors.push(String(error)))
  let cpu = 20
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    let data = {}
    if (path.includes('wizard-brains')) data = { brains: [{ id:'fixture', name:'Visual test fixture', endpoint:'http://127.0.0.1:8090',chat_path:'/brain/chat' }], selected:{chat:'fixture'} }
    if (path.endsWith('/status/')) data = { online:true,health:{uptime_secs:1234,cpu_percent:cpu,memory_percent:43,gpu_percent:62,error_rate_percent:.1},brain:{},multi_pool_stats:{totals:{pool_count:4,atoms:12000,concepts:924,exc_synapses:17500,inh_synapses:2200,cross_pool_edges:840},pools:[]} }
    if (path.endsWith('/learning/')) data={gate:{ready:false,scope:'Foundation first',blockers:[]},assessment_cases:{recall:12,grounding:24,transfer:24,joint:4,retention:24},queue:[],manifest:{prerogatives:[]},ruview:{connected:null}}
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(data)})
  })
  await page.clock.install()
  await page.setViewportSize({width:1280,height:1000}); await page.reload()
  await page.locator('.metric').filter({hasText:'CPU utilization'}).filter({hasText:'20%'}).waitFor()
  if(await page.locator('.metric').count()!==24) throw Error('Expected 24 metrics')
  const cpuMetric=page.locator('.metric').filter({hasText:'CPU utilization'})
  if(!(await cpuMetric.getAttribute('class')).includes('blue')) throw Error('Healthy state missing')
  cpu=85
  await page.clock.fastForward(30000)
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('.metric')).some(el=>el.textContent.includes('CPU utilization')&&el.classList.contains('pink')&&el.textContent.includes('85%')),{},{timeout:35000})
  cpu=97
  await page.clock.fastForward(30000)
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('.metric')).some(el=>el.textContent.includes('CPU utilization')&&el.classList.contains('red')&&el.textContent.includes('97%')),{},{timeout:35000})
  await page.screenshot({path:'.playwright-cli/wizard-crystal-desktop.png'})
  await page.setViewportSize({width:390,height:844})
  const orb=await page.locator('.conversation-orb').boundingBox()
  if(Math.abs(orb.width-orb.height)>1||orb.x<0||orb.x+orb.width>390) throw Error('Mobile sphere geometry')
  await page.screenshot({path:'.playwright-cli/wizard-crystal-mobile.png'})
  if(errors.length) throw Error(errors.join('\n'))
  return { symbols:24, liveColorTransitions:['blue','pink','red'], errors, mobileOrb:orb }
}
