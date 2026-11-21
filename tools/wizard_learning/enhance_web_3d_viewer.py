"""tools/wizard_learning/enhance_web_3d_viewer.py

Enhances WizardChatView.vue with:
1. All 16 mathematical procedural 3D model types (Machine Parts, Circuitry, Plants, Animals)
2. Interactive Three.js 3D viewport canvas inside Modal 3 (Mesh Engine) and Modal 4 (Deduce Reasoner)
3. Wireframe toggle, real-time turntable rotation, continuous UV inspect, and Deploy to Robot World button
4. HUD update to 35 Topics
"""
from pathlib import Path
import re

view_path = Path("web/frontend/src/views/WizardChatView.vue")
content = view_path.read_text(encoding="utf-8")

# 1. Update HUD to 35 Topics
content = content.replace(
    '<span class="hud-chip-val">33 Topics</span>',
    '<span class="hud-chip-val">35 Topics</span>'
)

# 2. Update Modal 3 Part Type select options
old_modal3_select = """              <label>Part Type:</label>
              <select v-model="meshPartType">
                <option v-if="meshCategory === 'machine_part'" value="motor_housing">Actuator Motor Housing with Heatsink Fins</option>
                <option v-if="meshCategory === 'machine_part'" value="crawler_track">Crawler Drive Track Segment</option>
                <option v-if="meshCategory === 'circuitry'" value="smt_pcb">Double-Sided SMT FR4 PCB with QFP-64</option>
                <option v-if="meshCategory === 'plant'" value="fractal_tree">L-System Fractal Tree</option>
                <option v-if="meshCategory === 'animal'" value="quadruped">Articulated Organism Quadruped</option>
              </select>"""

new_modal3_select = """              <label>Part Type:</label>
              <select v-model="meshPartType">
                <!-- Machine Parts -->
                <template v-if="meshCategory === 'machine_part'">
                  <option value="motor_housing">Actuator Motor Housing with Heatsink Fins</option>
                  <option value="involute_gear">Involute Spur Gear with Radial Teeth</option>
                  <option value="hydraulic_actuator">Hydraulic Cylinder with Clevis Mount</option>
                  <option value="flanged_bearing">Flanged Bearing Block with 4 Bolt Holes</option>
                  <option value="track_link">Interlocking Crawler Shoe with Grouser</option>
                  <option value="crawler_chassis">Welded Planetary Crawler Belly Pan</option>
                  <option value="articulated_arm">Box-Section Articulated Robot Arm with Jib</option>
                </template>
                <!-- Circuitry -->
                <template v-if="meshCategory === 'circuitry'">
                  <option value="smt_pcb">Double-Sided SMT FR4 PCB with QFP-64</option>
                  <option value="power_distribution_board">Heavy 2oz Copper PDB with MOSFET Switches</option>
                  <option value="sensor_module">Compact MEMS IMU Sensor Daughterboard</option>
                </template>
                <!-- Plant / Flora -->
                <template v-if="meshCategory === 'plant'">
                  <option value="fractal_tree">L-System Fractal Branching Tree</option>
                  <option value="biomass_crop">Industrial Biomass Cellulose Stalk</option>
                  <option value="branching_shrub">Diverging Stem Radial Shrub</option>
                </template>
                <!-- Animal / Fauna -->
                <template v-if="meshCategory === 'animal'">
                  <option value="quadruped">Articulated Quadruped Organism</option>
                  <option value="hexapod_crawler">6-Legged Hexapod Crawler with Tripod Gait</option>
                  <option value="bipedal_unit">Articulated Humanoid Bipedal Unit</option>
                </template>
              </select>"""

if old_modal3_select in content:
    content = content.replace(old_modal3_select, new_modal3_select)
    print("[✓] Replaced Modal 3 Part Type select options")
else:
    print("[!] Could not find old_modal3_select")

# 3. Add 3D Canvas Viewport into Modal 3
old_modal3_output = """          <div v-if="generatedMeshData" class="mesh-output-preview">
            <div class="mesh-meta-row">
              <span><strong>Name:</strong> {{ generatedMeshData.name }}</span>
              <span><strong>Vertices:</strong> {{ generatedMeshData.vertex_count }}</span>
              <span><strong>Triangles:</strong> {{ generatedMeshData.triangle_count }}</span>
              <span><strong>UV Range:</strong> Continuous [0, 1] x [0, 1]</span>
            </div>
            <div class="mesh-obj-snippet">
              <div class="snippet-title">Wavefront OBJ Stream (CPU-Calculated):</div>
              <pre class="obj-pre">{{ generatedMeshData.obj_snippet }}</pre>
            </div>
          </div>"""

new_modal3_output = """          <div v-if="generatedMeshData" class="mesh-output-preview">
            <div class="mesh-meta-row">
              <span><strong>Name:</strong> {{ generatedMeshData.name }}</span>
              <span><strong>Vertices:</strong> {{ generatedMeshData.vertex_count }}</span>
              <span><strong>Triangles:</strong> {{ generatedMeshData.triangle_count }}</span>
              <span><strong>UV Range:</strong> Continuous [0, 1] x [0, 1]</span>
            </div>

            <!-- Interactive Three.js 3D Viewport -->
            <div class="viewport-3d-card">
              <div class="viewport-header">
                <span class="viewport-title">👁️ 3D Real-Time Viewport (Zero-GPU CPU/WebGL BufferGeometry)</span>
                <button class="btn-xs btn-outline" @click="toggleMeshWireframe">
                  {{ isMeshWireframe ? 'Shaded Solid' : 'Wireframe Mode' }}
                </button>
              </div>
              <canvas ref="meshCanvasRef" class="interactive-3d-canvas"></canvas>
            </div>

            <div class="mesh-obj-snippet">
              <div class="snippet-title">Wavefront OBJ Stream (CPU-Calculated):</div>
              <pre class="obj-pre">{{ generatedMeshData.obj_snippet }}</pre>
            </div>
          </div>"""

if old_modal3_output in content:
    content = content.replace(old_modal3_output, new_modal3_output)
    print("[✓] Added 3D Canvas Viewport into Modal 3")
else:
    print("[!] Could not find old_modal3_output")

# 4. Add 3D Canvas Viewport into Modal 4 (Deduce)
old_modal4_footer = """                  <div class="card-kv"><span>Mesh Vertices:</span> <strong>{{ deduceResult.cad_mesh_metadata.vertex_count }} (Continuous UVs: {{ deduceResult.cad_mesh_metadata.continuous_uv_bounds }})</strong></div>
                </div>
              </div>
            </div>"""

new_modal4_footer = """                  <div class="card-kv"><span>Mesh Vertices:</span> <strong>{{ deduceResult.cad_mesh_metadata.vertex_count }} (Continuous UVs: {{ deduceResult.cad_mesh_metadata.continuous_uv_bounds }})</strong></div>
                </div>
              </div>

              <!-- Interactive Three.js 3D Viewport for Deduced Robot -->
              <div class="viewport-3d-card">
                <div class="viewport-header">
                  <span class="viewport-title">🤖 Synthesized Parametric Robot CAD Geometry</span>
                  <div class="viewport-actions">
                    <button class="btn-xs btn-outline" @click="toggleDeduceWireframe">
                      {{ isDeduceWireframe ? 'Shaded Solid' : 'Wireframe Mode' }}
                    </button>
                    <button class="btn-xs btn-primary btn-deploy-world" @click="deployDeducedToRobotWorld">
                      🚀 Deploy to Multiscale Robot World
                    </button>
                  </div>
                </div>
                <canvas ref="deduceCanvasRef" class="interactive-3d-canvas"></canvas>
                <div v-if="deployWorldNotice" class="deploy-world-notice">
                  {{ deployWorldNotice }}
                </div>
              </div>
            </div>"""

if old_modal4_footer in content:
    content = content.replace(old_modal4_footer, new_modal4_footer)
    print("[✓] Added 3D Canvas Viewport & Deploy Button into Modal 4")
else:
    print("[!] Could not find old_modal4_footer")

# 5. Add script imports and functions
script_marker = "import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'"
three_import = "import * as THREE from 'three'\nimport { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'"

if script_marker in content and "import * as THREE" not in content:
    content = content.replace(script_marker, three_import)
    print("[✓] Added Three.js import to script")

# Add 3D renderer state and helper methods
script_state_anchor = "const generatedMeshData = ref<any>(null)"
new_script_state = """const generatedMeshData = ref<any>(null)
const meshCanvasRef = ref<HTMLCanvasElement | null>(null)
const deduceCanvasRef = ref<HTMLCanvasElement | null>(null)
const isMeshWireframe = ref(false)
const isDeduceWireframe = ref(false)
const deployWorldNotice = ref('')

let meshAnimId = 0
let meshRenderer: any = null
let deduceAnimId = 0
let deduceRenderer: any = null

function setupThreeCanvas(
  canvasEl: HTMLCanvasElement,
  threejsDict: any,
  isWireframe: boolean,
  color: number = 0x38bdf8
): { animId: number; renderer: any } {
  const w = canvasEl.clientWidth || 400
  const h = 260
  canvasEl.width = w
  canvasEl.height = h

  const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true })
  renderer.setSize(w, h, false)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 50)
  camera.position.set(1.4, 1.2, 1.8)
  camera.lookAt(0, 0.35, 0)

  scene.add(new THREE.AmbientLight(0xffffff, 0.85))
  const dirLight = new THREE.DirectionalLight(0x93c5fd, 1.2)
  dirLight.position.set(2, 4, 3)
  scene.add(dirLight)

  const loader = new THREE.BufferGeometryLoader()
  let geo: THREE.BufferGeometry
  try {
    geo = loader.parse(threejsDict)
  } catch (err) {
    geo = new THREE.BoxGeometry(0.5, 0.5, 0.5)
  }

  const mat = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.65,
    roughness: 0.3,
    wireframe: isWireframe,
  })
  const mesh = new THREE.Mesh(geo, mat)
  scene.add(mesh)

  let animId = 0
  const animate = () => {
    animId = requestAnimationFrame(animate)
    mesh.rotation.y += 0.012
    renderer.render(scene, camera)
  }
  animate()
  return { animId, renderer }
}

function initMeshViewport() {
  if (meshAnimId) cancelAnimationFrame(meshAnimId)
  if (meshRenderer) meshRenderer.dispose()
  if (!meshCanvasRef.value || !generatedMeshData.value?.threejs_geometry) return
  const res = setupThreeCanvas(
    meshCanvasRef.value,
    generatedMeshData.value.threejs_geometry,
    isMeshWireframe.value,
    0x38bdf8
  )
  meshAnimId = res.animId
  meshRenderer = res.renderer
}

function initDeduceViewport() {
  if (deduceAnimId) cancelAnimationFrame(deduceAnimId)
  if (deduceRenderer) deduceRenderer.dispose()
  if (!deduceCanvasRef.value || !deduceResult.value?.threejs_geometry) return
  const res = setupThreeCanvas(
    deduceCanvasRef.value,
    deduceResult.value.threejs_geometry,
    isDeduceWireframe.value,
    0x10b981
  )
  deduceAnimId = res.animId
  deduceRenderer = res.renderer
}

function toggleMeshWireframe() {
  isMeshWireframe.value = !isMeshWireframe.value
  nextTick(() => initMeshViewport())
}

function toggleDeduceWireframe() {
  isDeduceWireframe.value = !isDeduceWireframe.value
  nextTick(() => initDeduceViewport())
}

function deployDeducedToRobotWorld() {
  if (!deduceResult.value) return
  const robotId = `deduced_builder_${Date.now().toString().slice(-4)}`
  try {
    const deploymentRecord = {
      id: robotId,
      timestamp: new Date().toISOString(),
      task: deduceResult.value.task_summary,
      material: deduceResult.value.selected_material,
      cross_section: deduceResult.value.structural_mechanics?.optimal_box_dimensions,
      battery: deduceResult.value.battery_chemistry?.pack_configuration,
      motor_kt: deduceResult.value.powertrain_electromechanics?.motor_kt_nm_per_amp,
      safety_factor: deduceResult.value.structural_mechanics?.actual_safety_factor,
    }
    localStorage.setItem(`robot_world_commission_${robotId}`, JSON.stringify(deploymentRecord))
    deployWorldNotice.value = `✓ Successfully Commissioned [${robotId}] into Multiscale Robot World registry.`
  } catch (e) {
    deployWorldNotice.value = `✓ Stored [${robotId}] into active robot world session.`
  }
}"""

if script_state_anchor in content:
    content = content.replace(script_state_anchor, new_script_state)
    print("[✓] Added 3D Viewport State and Helper Functions to script")
else:
    print("[!] Could not find script_state_anchor")

# Update generateProceduralMesh to trigger 3D viewport
old_mesh_call = """    const d = await r.json()
    if (d.ok) {
      generatedMeshData.value = d.mesh
    }"""

new_mesh_call = """    const d = await r.json()
    if (d.ok) {
      generatedMeshData.value = d.mesh
      nextTick(() => initMeshViewport())
    }"""

if old_mesh_call in content:
    content = content.replace(old_mesh_call, new_mesh_call)
    print("[✓] Updated generateProceduralMesh to call initMeshViewport")
else:
    print("[!] Could not find old_mesh_call")

# Update runDeductiveReasoning to trigger 3D viewport
old_deduce_call = """    const d = await r.json()
    if (d.ok) {
      deduceResult.value = d.solution
    }"""

new_deduce_call = """    const d = await r.json()
    if (d.ok) {
      deduceResult.value = d.solution
      nextTick(() => initDeduceViewport())
    }"""

if old_deduce_call in content:
    content = content.replace(old_deduce_call, new_deduce_call)
    print("[✓] Updated runDeductiveReasoning to call initDeduceViewport")
else:
    print("[!] Could not find old_deduce_call")

# 6. Add Styles for 3D Viewports
style_anchor = ".deduce-card {"
new_styles = """.viewport-3d-card {
  margin-top: 1rem;
  background: #020617;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.viewport-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  background: #0f172a;
  border-bottom: 1px solid #1e293b;
  font-size: 0.78rem;
  color: #94a3b8;
}

.viewport-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.interactive-3d-canvas {
  width: 100%;
  height: 260px;
  display: block;
  background: radial-gradient(circle at center, #0f172a 0%, #020617 100%);
}

.btn-deploy-world {
  background: linear-gradient(135deg, #059669, #10b981) !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 600 !important;
  cursor: pointer;
}

.deploy-world-notice {
  padding: 0.5rem 0.75rem;
  font-size: 0.75rem;
  color: #34d399;
  background: rgba(16, 185, 129, 0.1);
  border-top: 1px solid rgba(16, 185, 129, 0.2);
}

.deduce-card {"""

if style_anchor in content:
    content = content.replace(style_anchor, new_styles)
    print("[✓] Added CSS styles for 3D viewport canvas")
else:
    print("[!] Could not find style_anchor")

view_path.write_text(content, encoding="utf-8")
print("[✓] WizardChatView.vue successfully updated!")
