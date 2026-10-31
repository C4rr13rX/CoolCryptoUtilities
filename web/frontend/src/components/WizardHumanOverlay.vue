<template><div ref="host" class="human-overlay" aria-hidden="true" /></template>
<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
const props=defineProps<{command?:Record<string,any>;assets:Record<string,any>;playback?:{at_ms:number;present:boolean};amplitude:number}>()
const emit=defineEmits<{status:[value:string];capabilities:[value:Record<string,unknown>]}>()
const host=ref<HTMLDivElement>()
let renderer:THREE.WebGLRenderer|undefined,scene:THREE.Scene,camera:THREE.PerspectiveCamera,model:THREE.Group|undefined,mixer:THREE.AnimationMixer|undefined
let frame=0,last=0,opacity=0,assetId='',generation=0,controller:AbortController|undefined,resize:()=>void
let actions:Record<string,THREE.AnimationAction>={},activeAction='',morphMeshes:THREE.Mesh[]=[],materials:THREE.Material[]=[],started=performance.now()
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)')
function disposeModel(root?:THREE.Group){root?.traverse(object=>{const mesh=object as THREE.Mesh;mesh.geometry?.dispose();const list=Array.isArray(mesh.material)?mesh.material:[mesh.material];list.forEach(material=>{if(!material)return;for(const value of Object.values(material))if(value instanceof THREE.Texture)value.dispose();material.dispose()})});if(root)scene?.remove(root);if(root===model){morphMeshes=[];materials=[];mixer?.stopAllAction();mixer=undefined;actions={};activeAction=''}}
async function load(){
  const own=++generation;controller?.abort();controller=new AbortController()
  const id=props.command?.asset_id||'',asset=props.assets[id]
  if(id===assetId&&model)return
  disposeModel(model);model=undefined;assetId=id;opacity=0
  if(!id||!asset){emit('status',props.command?.visible?'Human manifestation waiting for a reviewed rigged asset':'Human overlay transparent');return}
  try{
    if(!/^\/static\/wizard-avatars\/[\w./-]+\.glb$/.test(asset.url)||asset.url.includes('..'))throw Error('Unregistered asset URL')
    const response=await fetch(asset.url,{signal:controller.signal});if(!response.ok)throw Error('Avatar asset unavailable')
    const data=await response.arrayBuffer();if(data.byteLength>64*1024*1024)throw Error('Avatar exceeds 64 MB')
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data))).map(value=>value.toString(16).padStart(2,'0')).join('')
    if(hash!==asset.sha256)throw Error('Avatar digest mismatch')
    const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(url.startsWith('blob:')||url.startsWith('data:'))return url;throw Error('Avatar must embed its textures and buffers')})
    const gltf=await new GLTFLoader(manager).parseAsync(data,'')
    if(own!==generation){disposeModel(gltf.scene);return}
    let vertices=0;gltf.scene.traverse(object=>{const mesh=object as THREE.Mesh;vertices+=mesh.geometry?.attributes.position?.count||0})
    if(vertices>500000){disposeModel(gltf.scene);throw Error('Avatar exceeds 500,000 vertices')}
    model=gltf.scene;const bounds=new THREE.Box3().setFromObject(model),size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());model.position.sub(center);model.position.y+=size.y/2;const normalized=new THREE.Group();normalized.add(model);normalized.userData.normalizedScale=2.8/Math.max(.01,size.y);normalized.scale.setScalar(normalized.userData.normalizedScale);model=normalized;scene.add(model)
    model.traverse(object=>{const mesh=object as THREE.Mesh;if(mesh.morphTargetDictionary)morphMeshes.push(mesh);const list=Array.isArray(mesh.material)?mesh.material:[mesh.material];list.forEach(material=>{if(material){const copy=material.clone();copy.transparent=true;copy.depthWrite=false;copy.opacity=0;materials.push(copy);if(Array.isArray(mesh.material))mesh.material=mesh.material.map(item=>item===material?copy:item);else mesh.material=copy;material.dispose()}})})
    mixer=new THREE.AnimationMixer(model);gltf.animations.forEach(clip=>actions[clip.name]=mixer!.clipAction(clip));emit('status',`Human rig loaded · ${Object.keys(actions).length} animations · ${morphMeshes.length} facial meshes`)
    emit('capabilities',{asset_id:id,animations:Object.keys(actions),morph_targets:[...new Set(morphMeshes.flatMap(mesh=>Object.keys(mesh.morphTargetDictionary||{})))],lip_sync:props.command?.visemes?.length?'model-timed visemes':'audio envelope jaw fallback only'})
  }catch(error){if(own===generation&&!(error instanceof DOMException&&error.name==='AbortError'))emit('status',String(error))}
}
watch(()=>[props.command?.asset_id,props.assets],()=>void load(),{deep:true})
watch(()=>props.command,()=>{started=performance.now()})
function animate(time:number){
 frame=requestAnimationFrame(animate);if(!renderer||document.hidden)return
 const dt=Math.min(.05,(time-last)/1000||.016);last=time
 const command=props.command,visible=!!command?.visible,target=visible?(command?.opacity??.55):0
 opacity+=(target-opacity)*Math.min(1,dt*2.5)
 if(model){
   model.visible=opacity>.002;materials.forEach(material=>material.opacity=opacity)
   const x=((command?.x??.75)-.5)*camera.aspect*4
   model.position.x=THREE.MathUtils.damp(model.position.x,x,1.2,dt);model.position.y=-1.65;model.scale.setScalar((command?.scale??1)*(model.userData.normalizedScale??1))
   const clip=command?.animation||''
   if(clip!==activeAction){if(actions[activeAction])actions[activeAction].fadeOut(.4);if(actions[clip])actions[clip].reset().fadeIn(.4).play();activeAction=clip}
   if(!reduced.matches)mixer?.update(dt)
   const at=props.playback?.present?props.playback.at_ms:time-started
   const timeline=Array.isArray(command?.visemes)?command!.visemes:[]
   const current=props.playback?.present?timeline.findLast((item:any)=>item.at_ms<=at):undefined
   morphMeshes.forEach(mesh=>{for(const [name,index]of Object.entries(mesh.morphTargetDictionary||{})){
     const indexNumber=index as number;let target=command?.morphs?.[name]??0
     if(current?.morph===name)target=current.weight
     // Audio-amplitude fallback is a jaw opening only, never labeled phoneme sync.
     else if(!timeline.length&&props.playback?.present&&/^(jawOpen|mouthOpen)$/i.test(name))target=Math.min(.8,props.amplitude*.9)
     if(reduced.matches&&name!==current?.morph)target=command?.morphs?.[name]??0
     mesh.morphTargetInfluences![indexNumber]=THREE.MathUtils.damp(mesh.morphTargetInfluences![indexNumber]||0,target,16,dt)
   }})
 }
 renderer.render(scene,camera)
}
onMounted(()=>{try{
 renderer=new THREE.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});renderer.setClearColor(0,0);renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;host.value!.append(renderer.domElement)
 scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(34,innerWidth/innerHeight,.1,50);camera.position.set(0,0,6)
 scene.add(new THREE.HemisphereLight(0xe9e6ff,0x292032,2));const key=new THREE.DirectionalLight(0xffe6dc,3);key.position.set(-2,4,5);scene.add(key);const fill=new THREE.DirectionalLight(0x95baff,2);fill.position.set(3,2,-1);scene.add(fill)
 resize=()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer?.setSize(innerWidth,innerHeight)};window.addEventListener('resize',resize);resize();void load();frame=requestAnimationFrame(animate)
 }catch{emit('status','Human overlay renderer unavailable')}})
onBeforeUnmount(()=>{generation++;controller?.abort();cancelAnimationFrame(frame);window.removeEventListener('resize',resize);disposeModel(model);renderer?.dispose();renderer?.forceContextLoss();renderer?.domElement.remove()})
</script>
<style scoped>.human-overlay { position:fixed;inset:0;z-index:9500;pointer-events:none!important;background:transparent; }.human-overlay :deep(*) { pointer-events:none!important; }</style>

