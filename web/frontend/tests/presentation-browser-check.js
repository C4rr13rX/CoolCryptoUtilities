// Tiny synthetic glTF exercises the rig protocol, never presented as a real human.
async (page) => {
  const errors=[];page.on('pageerror',error=>errors.push(String(error)))
  await page.clock.resume()
  const binary=new Uint8Array(156)
  binary.set(new Uint8Array(new Float32Array([-.3,0,0,.3,0,0,.3,1.7,0,-.3,1.7,0]).buffer),0)
  binary.set(new Uint8Array(new Uint16Array([0,1,2,0,2,3]).buffer),48)
  binary.set(new Uint8Array(new Float32Array([0,0,0,0,0,0,0,-.2,0,0,-.2,0]).buffer),60)
  binary.set(new Uint8Array(new Float32Array([0,1]).buffer),108)
  binary.set(new Uint8Array(new Float32Array([0,0,0,.1,0,0]).buffer),116)
  const document={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{weights:[0],extras:{targetNames:['jawOpen']},primitives:[{attributes:{POSITION:0},indices:1,material:0,targets:[{POSITION:2}]}]}],materials:[{doubleSided:true,pbrMetallicRoughness:{baseColorFactor:[.5,.25,.3,1],metallicFactor:0,roughnessFactor:.8}}],buffers:[{byteLength:156}],bufferViews:[{buffer:0,byteOffset:0,byteLength:48},{buffer:0,byteOffset:48,byteLength:12},{buffer:0,byteOffset:60,byteLength:48},{buffer:0,byteOffset:108,byteLength:8},{buffer:0,byteOffset:116,byteLength:24}],accessors:[{bufferView:0,componentType:5126,count:4,type:'VEC3',min:[-.3,0,0],max:[.3,1.7,0]},{bufferView:1,componentType:5123,count:6,type:'SCALAR'},{bufferView:2,componentType:5126,count:4,type:'VEC3',min:[0,-.2,0],max:[0,0,0]},{bufferView:3,componentType:5126,count:2,type:'SCALAR',min:[0],max:[1]},{bufferView:4,componentType:5126,count:2,type:'VEC3'}],animations:[{name:'walk',samplers:[{input:3,output:4,interpolation:'LINEAR'}],channels:[{sampler:0,target:{node:0,path:'translation'}}]}]}
  const json=new TextEncoder().encode(JSON.stringify(document)),padded=Math.ceil(json.length/4)*4
  const glb=new Uint8Array(12+8+padded+8+binary.length),view=new DataView(glb.buffer)
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,glb.length,true);view.setUint32(12,padded,true);view.setUint32(16,0x4e4f534a,true);glb.fill(32,20,20+padded);glb.set(json,20);view.setUint32(20+padded,binary.length,true);view.setUint32(24+padded,0x004e4942,true);glb.set(binary,28+padded)
  const digest=await page.evaluate(async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new Uint8Array(bytes)))).map(v=>v.toString(16).padStart(2,'0')).join(''),Array.from(glb))
  const assets={fixture:{url:'/static/wizard-avatars/test.glb',sha256:digest,license:'Synthetic test fixture only',source:'test driver'}}
  const design={version:1,eye:{hue:285,openness:.65,spiral_tightness:.16,nebula_strength:.8,particle_speed:.25},avatar:{visible:false,asset_id:'fixture',opacity:.55,x:.8,scale:1,animation:'walk',morphs:{jawOpen:.2},visemes:[{at_ms:0,morph:'jawOpen',weight:.8}]},personality:{self_description:'Test only',design_intent:'Protocol exercise',prerogative_alignment:{}}}
  let observations=0
  await page.route('**/static/wizard-avatars/test.glb',route=>route.fulfill({status:200,contentType:'model/gltf-binary',body:Buffer.from(glb)}))
  await page.route('**/api/**',async route=>{
    const path=new URL(route.request().url()).pathname;let data={}
    if(path.includes('wizard-brains'))data={brains:[{id:'fixture',name:'Synthetic rig test',endpoint:'http://127.0.0.1:8090',chat_path:'/brain/chat'}],selected:{chat:'fixture'}}
    if(path.endsWith('/status/'))data={online:true,health:{},brain:{}}
    if(path.endsWith('/self-design/')){
      if(route.request().method()==='POST'){observations++;data={recorded:true,learning:false}}
      else data={design:{...design,avatar:{...design.avatar,visible:false}},assets,gate:{ready:true}}
    }
    if(path.endsWith('/learning/'))data={gate:{ready:false,scope:'Fixture',blockers:[]},queue:[],assessment_cases:{},manifest:{prerogatives:[]},ruview:{connected:null}}
    if(path.endsWith('/message/')){const text=route.request().postDataJSON().text;data={answer:text.includes('hide')?'Fixture fade-out command':'Fixture manifestation command',node_online:true,self_design:{applied:true,design:{...design,avatar:{...design.avatar,visible:!text.includes('hide')}},assets}}}
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(data)})
  })
  await page.setViewportSize({width:1280,height:1000});await page.reload()
  await page.locator('.presentation-panel > summary').click()
  await page.getByText('Model visual control admitted',{exact:false}).waitFor()
  const input=page.locator('textarea.chat-input');await input.fill('manifest test rig');await input.press('Enter')
  await page.getByText('Human rig loaded',{exact:false}).waitFor()
  const clickThrough=await page.locator('.human-overlay').evaluate(element=>{const canvas=element.querySelector('canvas');return getComputedStyle(element).pointerEvents==='none'&&getComputedStyle(canvas).pointerEvents==='none'&&window.document.elementFromPoint(640,500)!==canvas})
  if(!clickThrough)throw Error('Overlay intercepts clicks')
  await input.fill('hide test rig');await input.press('Enter');await page.getByText('Fixture fade-out command',{exact:true}).waitFor()
  // Hidden commands retain the renderer so opacity can fade instead of tearing down.
  if(await page.locator('.human-overlay canvas').count()!==1)throw Error('Fade-out renderer was destroyed')
  await page.locator('.presentation-panel textarea').fill('Fixture feedback: improve visual clarity')
  await page.getByRole('button',{name:'Send design feedback'}).click()
  await page.getByText('Feedback recorded for the next response; no training performed.',{exact:true}).waitFor()
  await page.evaluate(()=>window.unmountEyePreview())
  if(errors.length)throw Error(errors.join('\n'))
  return {nativeCommands:true,rigLoaded:true,clickThrough,fadeOut:true,feedbackRecorded:observations>0,errors,syntheticFixtureOnly:true}
}
