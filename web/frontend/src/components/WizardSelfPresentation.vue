<template>
  <WizardHumanOverlay v-if="avatarInitialized" :command="design.avatar" :assets="assets" :playback="playback" :amplitude="amplitude" @status="avatarStatus=$event" @capabilities="rigCapabilities=$event" />
  <details class="presentation-panel"><summary>Self-design and personality learning</summary>
    <p>{{ state }} · {{ avatarStatus }}</p>
    <p>The brain controls declared eye parameters and registered human rigs. Preferences can change through reviewed experiences scoped to all twelve prerogatives.</p>
    <p v-if="!Object.keys(assets).length">No reviewed human rig is installed. The human overlay stays transparent.</p>
    <label>Design feedback<textarea v-model="feedback" maxlength="2000" placeholder="Describe what you observed and what would improve the design" /></label>
    <label><input type="checkbox" v-model="learn" />Learn this reviewed design outcome (requires foundation admission)</label>
    <label v-if="learn"><input type="checkbox" v-model="reviewed" />I verified the observed outcome and checked all twelve prerogatives together.</label>
    <button @click="submit" :disabled="saving||!feedback.trim()||(learn&&!reviewed)">Send design feedback</button>
    <p role="status">{{ result }}</p>
  </details>
</template>
<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import WizardHumanOverlay from './WizardHumanOverlay.vue'
const props=defineProps<{brainId:string;response?:Record<string,any>;telemetry?:Record<string,unknown>;playback?:{at_ms:number;present:boolean};amplitude:number}>()
const emit=defineEmits<{design:[value:Record<string,any>]}>()
const defaults=()=>({version:1,eye:{hue:285,openness:.65,spiral_tightness:.16,nebula_strength:.8,particle_speed:.25},avatar:{visible:false,asset_id:'',opacity:.55,x:.75,scale:1,animation:'',morphs:{},visemes:[]},personality:{self_description:'',design_intent:'',prerogative_alignment:{}}})
const design=ref<Record<string,any>>(defaults()),assets=ref<Record<string,any>>({}),state=ref('Loading self-design interface'),avatarStatus=ref('Human overlay transparent'),feedback=ref(''),learn=ref(false),reviewed=ref(false),saving=ref(false),result=ref('')
const rigCapabilities=ref<Record<string,unknown>>({})
const avatarInitialized=ref(false)
watch(()=>design.value.avatar.visible,visible=>{if(visible)avatarInitialized.value=true})
let controller:AbortController|undefined,lastReport=0,alive=true
const csrf=()=>decodeURIComponent(document.cookie.match(/(?:^|; )csrftoken=([^;]*)/)?.[1]||'')
async function load(){controller?.abort();const own=controller=new AbortController();avatarInitialized.value=false;design.value=defaults();assets.value={};emit('design',design.value)
 try{const response=await fetch('/api/wizard-chat/self-design/',{signal:own.signal});const data=await response.json();if(!response.ok||!data.design)throw Error(data.error||'Self-design interface unavailable');if(own.signal.aborted)return;design.value=data.design;assets.value=data.assets||{};state.value=data.gate?.ready?'Model visual control admitted':'Foundation first · model visual proposals are held';emit('design',design.value)}catch(error){if(!own.signal.aborted)state.value=String(error)}
}
watch(()=>props.brainId,()=>void load())
watch(()=>props.response,response=>{if(response?.applied&&response.design){design.value=response.design;assets.value=response.assets||assets.value;state.value='Native model design applied · awaiting reviewed outcome';emit('design',design.value)}else if(response?.reason)state.value=response.reason})
async function send(body:Record<string,unknown>){const response=await fetch('/api/wizard-chat/self-design/',{method:'POST',headers:{'Content-Type':'application/json','X-CSRFToken':csrf()},body:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw Error(value.error||'Feedback unavailable');return value}
watch(()=>props.telemetry,telemetry=>{if(!telemetry||Date.now()-lastReport<15000)return;lastReport=Date.now();void send({kind:'visual-observation',design:design.value,observed:{...telemetry,rig:rigCapabilities.value}}).catch(()=>{/* Render observation failure never becomes an inferred successful learning outcome. */})})
async function submit(){saving.value=true;result.value='';try{const value=await send({kind:'visual-feedback',design:design.value,feedback:feedback.value,observed:props.telemetry||{},learn:learn.value,reviewed_experience:learn.value?{reviewer:'authenticated operator',outcome_verified:reviewed.value,checked_prerogatives:Array.from({length:12},(_,i)=>`P${String(i+1).padStart(2,'0')}`),conflicts:[],observed_outcome:{design_feedback:feedback.value}}:undefined});if(alive)result.value=value.consolidated?'Design outcome consolidated; retention reassessment required.':'Feedback recorded for the next response; no training performed.'}catch(error){if(alive)result.value=String(error)}finally{if(alive)saving.value=false}}
onMounted(()=>void load());onBeforeUnmount(()=>{alive=false;controller?.abort()})
</script>
<style scoped>.presentation-panel { flex-shrink:0;padding:.5rem 1rem;font-size:.74rem;color:#d9c6e5;max-height:35vh;overflow:auto;border-top:1px solid #bda4cd30; }summary { cursor:pointer; }label { display:block;margin:.5rem 0; }textarea { display:block;width:95%;min-height:3rem;color:#e9d9ef;background:#100c1a;border:1px solid #685371;border-radius:8px;padding:.5rem; }button { color:#f1d8ec;background:#21152a;border:1px solid #9b739f;border-radius:12px;padding:.4rem .7rem; }button:disabled { opacity:.4; }</style>
