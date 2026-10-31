<template>
  <div ref="host" class="eye" @pointermove="look" @pointerleave="gaze = [0,0]">
    <span v-if="error" class="render-error">{{ error }}</span>
    <span class="signal-label">{{ signalLabel }}</span>
  </div>
</template>
<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import * as THREE from 'three'
import { vectorAt, type EyeSignal } from '@/utils/eyeSignal'
import type { EyeAssociation } from '@/utils/eyeAssociations'
import type { ObservationCue } from '@/utils/observation'
const props = defineProps<{ concepts: string[]; messageId: string; busy: boolean; amplitude: number; signal?: EyeSignal; playback?: {at_ms:number;present:boolean}; association?: EyeAssociation; observation?: ObservationCue; design?: Record<string,number> }>()
const emit=defineEmits<{telemetry:[value:Record<string,unknown> ]}>()
const host=ref<HTMLDivElement>(), error=ref('')
const signalLabel=computed(()=>props.signal?.source==='model-vector'?'Model vector trajectory':props.association?`Recalled silhouette · ${props.association.concept}`:props.concepts.length?'Concept signature · snapshot':'Ambient visual · awaiting concepts')
let gaze=[0,0], renderer:THREE.WebGLRenderer|undefined, scene:THREE.Scene, camera:THREE.PerspectiveCamera, resize:ResizeObserver, frame=0, start=performance.now(), last=0, frames=0, reportAt=0
let eye:THREE.Mesh, cornea:THREE.Mesh, stars:THREE.Points, nebula:THREE.Mesh
let positions:Float32Array, homes:Float32Array, velocities:Float32Array
let pupilTexture:THREE.CanvasTexture|undefined
const preview=document.createElement('canvas');preview.width=preview.height=64
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)')
const uniforms={time:{value:0},voice:{value:0},hue:{value:.79},openness:{value:.65},tightness:{value:.16},nebula:{value:.8},gaze:{value:new THREE.Vector2()},vector:{value:new THREE.Vector4()},pupil:{value:null as THREE.Texture|null},hasPupil:{value:0}}
const vertex=`varying vec3 localPosition; varying vec3 normalView; varying vec3 viewPosition;
void main(){localPosition=position;normalView=normalize(normalMatrix*normal);vec4 p=modelViewMatrix*vec4(position,1.);viewPosition=-p.xyz;gl_Position=projectionMatrix*p;}`
const noise=`float hash(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float noise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);}
float fbm(vec3 p){float s=0.,a=.5;for(int i=0;i<5;i++){s+=a*noise(p);p=p*2.03+17.1;a*=.5;}return s;}
vec3 hueColor(float h){vec3 p=abs(fract(vec3(h)+vec3(0.,.666667,.333333))*6.-3.);return mix(vec3(1.),clamp(p-1.,0.,1.),.7);}`
const irisFragment=`precision highp float; varying vec3 localPosition;varying vec3 normalView;varying vec3 viewPosition;
uniform float time,voice,hue,openness,tightness,hasPupil;uniform vec2 gaze;uniform vec4 vector;uniform sampler2D pupil;
${noise}
void main(){
 vec2 p=localPosition.xy; float lid=(.12+.66*openness)*pow(max(0.,1.-p.x*p.x),.68);
 float edge=lid-abs(p.y); if(edge<0.)discard;
 vec2 iris=p-vec2(gaze.x*.19,gaze.y*.12);iris.y*=1.02;
 float r=length(iris),a=atan(iris.y,iris.x);float irisSize=.58;
 float fiber=fbm(vec3(cos(a)*32.,sin(a)*32.,r*19.));
 float fine=sin(a*185.+noise(vec3(a*11.,r*23.,2.))*5.+r*21.);
 vec3 pigment=mix(vec3(.09,.055,.12),hueColor(hue)*.3,fiber);
 pigment+=vec3(.4,.31,.38)*pow(max(0.,fine),14.)*.28;
 float folds=sin(a*71.+r*64.+fiber*5.);pigment*=.7+.3*folds;
 float limbus=smoothstep(irisSize,irisSize-.045,r);pigment*=.45+.55*smoothstep(.15,.29,r);
 // A single logarithmic curve: theta = log(r/a)/b, with one branch modulo 2pi.
 float deformation=.05*(vector.x*sin(a*2.)+vector.y*cos(a*3.));
 float phase=a-log(max(.025,r)/.025)/tightness-time*.025+deformation;
 float angularDistance=abs(atan(sin(phase),cos(phase)));
 float spiral=exp(-pow(angularDistance*max(.08,r)*110.,2.))*smoothstep(.10,.15,r)*(1.-smoothstep(.53,.58,r));
 pigment+=mix(vec3(.8,.49,.65),vec3(.57,.71,1.),.5+.5*sin(log(max(.02,r))*3.))*spiral*1.7;
 vec3 sclera=vec3(.11,.09,.14)+vec3(.12,.09,.14)*fbm(vec3(p*9.,1.));
 float vessel=pow(max(0.,sin(a*24.+fbm(vec3(p*18.,3.))*8.+r*19.)),35.)*smoothstep(.58,.9,r);
 sclera+=vec3(.09,.015,.035)*vessel;
 vec3 color=mix(sclera,pigment,limbus);
 float pupilRadius=.125+voice*.018+.008*sin(time*.3);
 float pupilMask=1.-smoothstep(pupilRadius-.008,pupilRadius+.008,r);
 if(hasPupil>.5){vec2 uv=iris/.48+.5;pupilMask=texture2D(pupil,uv).a*step(0.,uv.x)*step(uv.x,1.)*step(0.,uv.y)*step(uv.y,1.);}
 color=mix(color,vec3(.001,.001,.004),pupilMask);
 vec3 n=normalize(normalView),v=normalize(viewPosition),light=normalize(vec3(-.6,.8,2.));
 float diffuse=.28+.72*max(0.,dot(n,light));color*=diffuse;
 float spec=pow(max(0.,dot(n,normalize(light+v))),100.);color+=vec3(.78,.85,1.)*spec*.8;
 color*=smoothstep(0.,.045,edge);color+=vec3(.34,.22,.3)*exp(-abs(edge-.012)*130.)*.35;
 gl_FragColor=vec4(pow(max(color,vec3(0.)),vec3(.65)),smoothstep(0.,.012,edge));
}`
function look(event:PointerEvent){const rect=host.value!.getBoundingClientRect();gaze=[(event.clientX-rect.left)/rect.width-.5,(event.clientY-rect.top)/rect.height-.5]}
function association(){
 pupilTexture?.dispose();pupilTexture=undefined;uniforms.hasPupil.value=0
 const shape=props.association?.silhouette;if(!shape)return
 const canvas=document.createElement('canvas');canvas.width=canvas.height=256;const ctx=canvas.getContext('2d')!
 const scale=230/Math.max(shape.width,shape.height);ctx.translate(128,128);ctx.scale(scale,scale);ctx.translate(-shape.width/2,-shape.height/2);ctx.fillStyle='white';ctx.fill(new Path2D(shape.path),'evenodd')
 pupilTexture=new THREE.CanvasTexture(canvas);uniforms.pupil.value=pupilTexture;uniforms.hasPupil.value=1
}
watch(()=>props.association,association)
watch(()=>props.messageId,()=>{start=performance.now()})
function animate(time:number){
 frame=requestAnimationFrame(animate);if(!renderer||time-last<(reduced.matches?200:33)||document.hidden)return
 const dt=Math.min(.05,(time-last)/1000||.033);last=time;frames++
 const t=reduced.matches?0:time/1000, d=props.design||{}
 uniforms.time.value=t;uniforms.voice.value=props.amplitude;uniforms.hue.value=(d.hue??285)/360;uniforms.tightness.value=d.spiral_tightness??.16
 uniforms.nebula.value=d.nebula_strength??.8
 const expression=props.association?.expression
 const targetOpen=d.openness??(props.busy?.82:expression==='wonder'?.93:expression==='amused'?.58:.67)
 uniforms.openness.value+=(targetOpen-uniforms.openness.value)*.06
 const cue=props.observation&&time-props.observation.at<3500?props.observation:undefined
 uniforms.gaze.value.lerp(new THREE.Vector2(cue?.x??gaze[0],-(cue?.y??gaze[1])),.06)
 const values=props.signal?.source==='model-vector'?vectorAt(props.signal.frames,props.playback?.present?props.playback.at_ms:time-start):[]
 uniforms.vector.value.lerp(new THREE.Vector4(...[0,1,2,3].map(i=>Math.tanh(values[i]||0)) as [number,number,number,number]),.1)
 eye.rotation.y=uniforms.gaze.value.x*.09;eye.rotation.x=-uniforms.gaze.value.y*.06;cornea.rotation.copy(eye.rotation)
 if(!reduced.matches){
  const speed=d.particle_speed??.25
  for(let i=0;i<positions.length;i+=3){
   const x=positions[i],y=positions[i+1],z=positions[i+2],r2=x*x+y*y+z*z+.5
   // Softened central attraction, rotational flow, and damped restoring force.
   const force=.012*speed/Math.pow(r2,1.5)
   for(let j=0;j<3;j++){const swirl=j===0?-y:j===1?x:0;velocities[i+j]+=(swirl*force+(homes[i+j]-positions[i+j])*.025-velocities[i+j]*.2)*dt;positions[i+j]+=velocities[i+j]*dt}
  }
  stars.geometry.attributes.position.needsUpdate=true;stars.rotation.y=t*.003;nebula.rotation.y=t*.001
 }
 renderer.render(scene,camera)
 if(time-reportAt>15000){
  // Exact low-resolution rendered pixels accompany the scene graph, never a
  // fabricated caption that claims the model already understands its picture.
  const ctx=preview.getContext('2d',{willReadFrequently:true})!;ctx.drawImage(renderer.domElement,0,0,64,64)
  const rgba=ctx.getImageData(0,0,64,64).data,packed=new Uint8Array(4096)
  for(let i=0;i<packed.length;i++)packed[i]=((rgba[i*4]>>6)<<4)|((rgba[i*4+1]>>6)<<2)|(rgba[i*4+2]>>6)
  emit('telemetry',{renderer:'threejs-mutable-eye-v1',fps:Math.round(frames/Math.max(.001,(time-reportAt)/1000)),particles:positions.length/3,spiral_count:1,source:props.signal?.source||'ambient',webgl:true,reduced_motion:reduced.matches,preview:{width:64,height:64,encoding:'rgb222-u8-base64',frame:btoa(String.fromCharCode(...packed))}});frames=0;reportAt=time
 }
}
onMounted(()=>{
 try{
  renderer=new THREE.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.35;host.value!.prepend(renderer.domElement);renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','Three dimensional crystal eye with one logarithmic spiral and a particle universe')
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(41,1,.1,30);camera.position.set(0,0,4.5)
  const universeMaterial=new THREE.ShaderMaterial({side:THREE.BackSide,uniforms,vertexShader:vertex,fragmentShader:`precision highp float;varying vec3 localPosition;uniform float time,nebula;${noise}
   void main(){vec3 p=normalize(localPosition);float cloud=fbm(p*5.+vec3(time*.002,0.,0.));float dust=fbm(p*12.+cloud*3.);float band=exp(-pow((p.y+.22*p.x+.08*sin(p.z*9.))*3.,2.));vec3 color=vec3(.002,.003,.012);color+=mix(vec3(.12,.018,.12),vec3(.03,.065,.21),dust)*pow(cloud,2.)*band*nebula*4.;color+=vec3(.32,.2,.25)*pow(cloud*dust,3.)*nebula*3.;gl_FragColor=vec4(pow(max(color,vec3(0.)),vec3(.48)),1.);}`})
  nebula=new THREE.Mesh(new THREE.SphereGeometry(9,48,32),universeMaterial);scene.add(nebula)
  const count=2200;positions=new Float32Array(count*3);velocities=new Float32Array(count*3);const colors=new Float32Array(count*3),sizes=new Float32Array(count)
  let seed=1947;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296}
  for(let i=0;i<count;i++){const a=random()*Math.PI*2,z=random()*2-1,r=3+random()*5,s=Math.sqrt(1-z*z);positions.set([Math.cos(a)*s*r,z*r,Math.sin(a)*s*r],i*3);const tint=random();colors.set(tint>.94?[1,.62,.8]:tint>.85?[.65,.76,1]:[.85,.86,.92],i*3);sizes[i]=random()>.98?3.2:.5+random()*1.3}
  homes=positions.slice();const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));geometry.setAttribute('size',new THREE.BufferAttribute(sizes,1))
  stars=new THREE.Points(geometry,new THREE.ShaderMaterial({transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,vertexColors:true,uniforms:{time:uniforms.time},vertexShader:`attribute float size;varying vec3 tint;varying float sparkle;uniform float time;void main(){tint=color;sparkle=.7+.3*sin(time*.6+position.x*17.);vec4 p=modelViewMatrix*vec4(position,1.);gl_PointSize=clamp(size*8./max(1.,-p.z),.8,8.);gl_Position=projectionMatrix*p;}`,fragmentShader:`varying vec3 tint;varying float sparkle;void main(){vec2 p=gl_PointCoord-.5;float d=length(p);float core=exp(-d*d*65.);float rays=exp(-abs(p.x)*75.-abs(p.y)*8.)+exp(-abs(p.y)*75.-abs(p.x)*8.);gl_FragColor=vec4(tint,(core+rays*.15)*sparkle);}`}));scene.add(stars)
  eye=new THREE.Mesh(new THREE.SphereGeometry(1,128,80),new THREE.ShaderMaterial({uniforms,vertexShader:vertex,fragmentShader:irisFragment,transparent:true,side:THREE.FrontSide}));eye.scale.set(1.18,.82,.48);eye.position.set(0,.31,.15);scene.add(eye)
  cornea=new THREE.Mesh(new THREE.SphereGeometry(1.035,96,64),new THREE.MeshPhysicalMaterial({color:0xdad5ee,roughness:.06,metalness:0,transmission:.95,thickness:.08,ior:1.376,clearcoat:1,clearcoatRoughness:.025,transparent:true,opacity:.13,depthWrite:false}));cornea.scale.copy(eye.scale);cornea.position.copy(eye.position);scene.add(cornea)
  const key=new THREE.DirectionalLight(0xf4e5ff,3);key.position.set(-3,4,5);scene.add(key);const rim=new THREE.PointLight(0x829cff,12);rim.position.set(2,-1,2);scene.add(rim);scene.add(new THREE.AmbientLight(0x9b79c9,.7))
  const glass=new THREE.Mesh(new THREE.SphereGeometry(1.53,96,64),new THREE.MeshPhysicalMaterial({color:0xded8f4,roughness:.045,transmission:1,thickness:.1,ior:1.46,clearcoat:1,clearcoatRoughness:.03,transparent:true,opacity:.09,depthWrite:false}));scene.add(glass)
  resize=new ResizeObserver(()=>{const size=Math.max(1,Math.round(host.value!.clientWidth));renderer?.setSize(size,size)});resize.observe(host.value!);association();reportAt=performance.now();frame=requestAnimationFrame(animate)
 }catch{error.value='3D renderer unavailable on this device';emit('telemetry',{webgl:false,error:'renderer unavailable'})}
})
onBeforeUnmount(()=>{cancelAnimationFrame(frame);resize?.disconnect();pupilTexture?.dispose();scene?.traverse(object=>{const mesh=object as THREE.Mesh;mesh.geometry?.dispose();const materials=Array.isArray(mesh.material)?mesh.material:[mesh.material];materials.forEach(material=>material?.dispose())});renderer?.dispose();renderer?.forceContextLoss();renderer?.domElement.remove()})
</script>
<style scoped>
.eye { position:absolute;inset:0;overflow:hidden;border-radius:50%;background:#01020a; }
.eye :deep(canvas) { width:100%;height:100%;display:block; }
.signal-label { position:absolute;bottom:7%;left:19%;right:19%;text-align:center;color:#cbbad17a;font-size:clamp(.35rem,.85cqw,.55rem);letter-spacing:.08em;text-transform:uppercase;pointer-events:none; }
.render-error { position:absolute;top:30%;left:15%;right:15%;text-align:center;color:#eaa8d0;font-size:.8rem; }
</style>
