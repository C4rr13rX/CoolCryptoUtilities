"""tools/wizard_learning/enhance_blueprint_ui.py

Updates WizardChatView.vue to include:
1. Production ISO 128 / ASME Y14.5 Engineering Vector Blueprint viewer
2. Bill of Materials (BOM) schedule table
3. AWS D1.1 & ISO 2768-mK Manufacturing Process Plan
4. Download SVG Blueprint action button
"""
from pathlib import Path

view_path = Path("web/frontend/src/views/WizardChatView.vue")
content = view_path.read_text(encoding="utf-8")

# 1. Add Blueprint Buttons & Drawer to Modal 4
old_viewport_header = """                <div class="viewport-header">
                  <span class="viewport-title">🤖 Synthesized Parametric Robot CAD Geometry</span>
                  <div class="viewport-actions">
                    <button class="btn-xs btn-outline" @click="toggleDeduceWireframe">
                      {{ isDeduceWireframe ? 'Shaded Solid' : 'Wireframe Mode' }}
                    </button>
                    <button class="btn-xs btn-primary btn-deploy-world" @click="deployDeducedToRobotWorld">
                      🚀 Deploy to Multiscale Robot World
                    </button>
                  </div>
                </div>"""

new_viewport_header = """                <div class="viewport-header">
                  <span class="viewport-title">🤖 Synthesized Parametric Robot CAD Geometry</span>
                  <div class="viewport-actions">
                    <button class="btn-xs btn-outline" @click="toggleDeduceWireframe">
                      {{ isDeduceWireframe ? 'Shaded Solid' : 'Wireframe Mode' }}
                    </button>
                    <button class="btn-xs btn-outline" @click="showBlueprintDrawer = !showBlueprintDrawer">
                      📐 {{ showBlueprintDrawer ? 'Hide Blueprint' : 'View Blueprint' }}
                    </button>
                    <button class="btn-xs btn-outline" @click="downloadBlueprintSVG" :disabled="!deduceResult?.blueprint_svg">
                      📥 Export SVG
                    </button>
                    <button class="btn-xs btn-primary btn-deploy-world" @click="deployDeducedToRobotWorld">
                      🚀 Deploy to World
                    </button>
                  </div>
                </div>"""

if old_viewport_header in content:
    content = content.replace(old_viewport_header, new_viewport_header)
    print("[✓] Updated viewport header with blueprint action buttons")
else:
    print("[!] Could not find old_viewport_header")

# Add Blueprint SVG, BOM, and Manufacturing Plan below the 3D viewport
old_viewport_card_end = """                <div v-if="deployWorldNotice" class="deploy-world-notice">
                  {{ deployWorldNotice }}
                </div>
              </div>
            </div>"""

new_viewport_card_end = """                <div v-if="deployWorldNotice" class="deploy-world-notice">
                  {{ deployWorldNotice }}
                </div>
              </div>

              <!-- ISO 128 / ASME Y14.5 Technical Vector Blueprint -->
              <div v-if="showBlueprintDrawer && deduceResult?.blueprint_svg" class="blueprint-card">
                <div class="blueprint-card-header">
                  <span class="card-title">📐 ISO 128 / ASME Y14.5 Engineering Vector Blueprint</span>
                  <button class="btn-xs btn-outline" @click="downloadBlueprintSVG">📥 Download SVG Blueprint</button>
                </div>
                <div class="blueprint-svg-frame" v-html="deduceResult.blueprint_svg"></div>
              </div>

              <!-- Bill of Materials (BOM) Schedule -->
              <div v-if="deduceResult?.bill_of_materials?.length" class="bom-card">
                <div class="card-title">📋 Production Bill of Materials (BOM Schedule)</div>
                <div class="bom-table-wrapper">
                  <table class="bom-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Part Designation</th>
                        <th>Qty</th>
                        <th>Material Specification</th>
                        <th>Dimensions</th>
                        <th>Standard</th>
                        <th>Manufacturing Process</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="item in deduceResult.bill_of_materials" :key="item.item">
                        <td>{{ item.item }}</td>
                        <td><strong>{{ item.part_name }}</strong></td>
                        <td>{{ item.qty }}</td>
                        <td>{{ item.material }}</td>
                        <td class="mono">{{ item.dimensions }}</td>
                        <td>{{ item.standard }}</td>
                        <td>{{ item.process }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <!-- Manufacturing Process Plan -->
              <div v-if="deduceResult?.manufacturing_plan?.length" class="mfg-card">
                <div class="card-title">🏭 AWS D1.1 & ISO 2768-mK Fabrication Sequence</div>
                <ol class="mfg-step-list">
                  <li v-for="(step, sIdx) in deduceResult.manufacturing_plan" :key="sIdx">{{ step }}</li>
                </ol>
              </div>
            </div>"""

if old_viewport_card_end in content:
    content = content.replace(old_viewport_card_end, new_viewport_card_end)
    print("[✓] Added Blueprint SVG, BOM Schedule, and Manufacturing Plan cards")
else:
    print("[!] Could not find old_viewport_card_end")

# 2. Add Script State and Functions
script_anchor = "const deployWorldNotice = ref('')"
new_script_fields = """const deployWorldNotice = ref('')
const showBlueprintDrawer = ref(true)

function downloadBlueprintSVG() {
  if (!deduceResult.value?.blueprint_svg) return
  const blob = new Blob([deduceResult.value.blueprint_svg], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `W1Z4RD_ENGINEERING_BLUEPRINT_${Date.now()}.svg`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}"""

if script_anchor in content:
    content = content.replace(script_anchor, new_script_fields)
    print("[✓] Added showBlueprintDrawer ref and downloadBlueprintSVG handler")
else:
    print("[!] Could not find script_anchor")

# 3. Add CSS Styles
style_anchor = ".deduce-card {"
new_blueprint_styles = """.blueprint-card, .bom-card, .mfg-card {
  margin-top: 1rem;
  background: #020617;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
}

.blueprint-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0.85rem;
  background: #0f172a;
  border-bottom: 1px solid #1e293b;
}

.blueprint-svg-frame {
  width: 100%;
  max-height: 520px;
  overflow: auto;
  background: #071527;
  padding: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.blueprint-svg-frame svg {
  max-width: 100%;
  height: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  border-radius: 4px;
}

.bom-card {
  padding: 0.85rem;
}

.bom-table-wrapper {
  overflow-x: auto;
  margin-top: 0.5rem;
}

.bom-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
  color: #cbd5e1;
}

.bom-table th {
  background: #0f172a;
  color: #94a3b8;
  text-align: left;
  padding: 0.5rem 0.65rem;
  border: 1px solid #1e293b;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.7rem;
}

.bom-table td {
  padding: 0.45rem 0.65rem;
  border: 1px solid #1e293b;
}

.bom-table tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.5);
}

.mfg-card {
  padding: 0.85rem;
}

.mfg-step-list {
  margin: 0.5rem 0 0 1.2rem;
  padding: 0;
  font-size: 0.78rem;
  color: #cbd5e1;
  line-height: 1.6;
}

.mfg-step-list li {
  margin-bottom: 0.35rem;
}

.deduce-card {"""

if style_anchor in content:
    content = content.replace(style_anchor, new_blueprint_styles)
    print("[✓] Added Blueprint and BOM CSS styles")
else:
    print("[!] Could not find style_anchor")

view_path.write_text(content, encoding="utf-8")
print("[✓] WizardChatView.vue successfully updated with blueprint features!")
