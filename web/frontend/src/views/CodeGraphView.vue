<template>
  <div class="codegraph-view">
    <section class="panel repository-bar">
      <div class="repository-picker">
        <div class="eyebrow">CODEGRAPH / REPOSITORY</div>
        <select v-model="selectedRepositoryId" :disabled="switching" @change="switchRepository">
          <option v-for="repo in repositories" :key="repo.id" :value="repo.id">
            {{ repo.name }} · {{ repo.source_type }}
          </option>
        </select>
        <span class="source-path" :title="activeRepository?.location">{{ activeRepository?.location || 'No repository selected' }}</span>
      </div>
      <div class="repository-actions">
        <button class="btn ghost" type="button" @click="showAddRepository = !showAddRepository">Add codebase</button>
        <button class="btn ghost" type="button" :disabled="repositories.length <= 1 || switching" @click="removeActiveRepository">Remove</button>
        <button class="btn" type="button" :disabled="building || switching" @click="refreshGraph(true)">
          {{ building ? 'Indexing…' : 'Re-index' }}
        </button>
        <button class="btn ghost" type="button" :disabled="!graphReady" @click="fitGraph">Center graph</button>
      </div>
      <form v-if="showAddRepository" class="repository-form" @submit.prevent="saveRepository">
        <label>
          <span>Source</span>
          <select v-model="repositoryDraft.source_type">
            <option value="local">Local folder</option>
            <option value="github">GitHub repository</option>
          </select>
        </label>
        <label>
          <span>Name</span>
          <input v-model="repositoryDraft.name" placeholder="Optional display name" />
        </label>
        <label class="location-field">
          <span>{{ repositoryDraft.source_type === 'github' ? 'GitHub URL' : 'Folder path' }}</span>
          <input
            v-model="repositoryDraft.location"
            :placeholder="repositoryDraft.source_type === 'github' ? 'https://github.com/owner/repository' : 'D:\\Projects\\MyProject'"
            required
          />
        </label>
        <label v-if="repositoryDraft.source_type === 'github'">
          <span>Branch</span>
          <input v-model="repositoryDraft.branch" placeholder="Default branch" />
        </label>
        <button class="btn" type="submit" :disabled="addingRepository">{{ addingRepository ? 'Adding…' : 'Add and inspect' }}</button>
        <button class="btn ghost" type="button" @click="showAddRepository = false">Cancel</button>
        <p v-if="repositoryError" class="form-error">{{ repositoryError }}</p>
      </form>
    </section>

    <section class="panel graph-status">
      <div class="status-copy">
        <span class="status-dot" :class="statusState"></span>
        <strong>{{ statusMessage }}</strong>
        <span v-if="building">{{ Math.round(statusProgress * 100) }}%</span>
      </div>
      <div v-if="building" class="progress-track"><span :style="{ width: `${statusProgress * 100}%` }"></span></div>
      <div class="summary">
        <span><b>{{ summary.modules || 0 }}</b> modules</span>
        <span><b>{{ summary.files || 0 }}</b> files</span>
        <span><b>{{ summary.classes || 0 }}</b> classes</span>
        <span><b>{{ summary.functions || 0 }}</b> functions</span>
        <span><b>{{ summary.relationships || 0 }}</b> wires</span>
      </div>
    </section>

    <section class="panel graph-shell">
      <aside class="graph-tools">
        <label class="search-field">
          <span>Find symbol</span>
          <input v-model="searchText" placeholder="Class, method, file…" @keydown.enter.prevent="focusFirstSearch" />
        </label>
        <div class="relation-filters">
          <label v-for="kind in relationshipKinds" :key="kind">
            <input v-model="visibleRelationships" type="checkbox" :value="kind" @change="applyRelationshipVisibility" />
            <span class="wire-key" :class="kind"></span>{{ kind }}
          </label>
        </div>
        <div class="legend">
          <span v-for="kind in nodeKinds" :key="kind"><i :style="{ background: nodeColorCss(kind) }"></i>{{ kind }}</span>
        </div>
      </aside>

      <div ref="canvasContainer" class="canvas-host">
        <div v-if="hoveredNode" class="node-tooltip" :style="tooltipStyle">
          <div class="tooltip-kind">{{ hoveredNode.kind }}</div>
          <strong>{{ hoveredNode.label }}</strong>
          <span>{{ hoveredNode.file }}<template v-if="hoveredNode.line">:{{ hoveredNode.line }}</template></span>
          <span>{{ ioLabel(hoveredNode) }}</span>
          <small>Click to open source</small>
        </div>
        <div v-if="graphError" class="canvas-message error">{{ graphError }}</div>
        <div v-else-if="!graphReady" class="canvas-message">
          {{ building ? 'Building the dependency map in the background…' : 'No indexed symbols yet.' }}
        </div>
      </div>

      <aside v-if="selectedNode" class="selection-card">
        <button type="button" class="close-mini" @click="selectedNode = null">×</button>
        <div class="eyebrow">{{ selectedNode.kind }}</div>
        <h3>{{ selectedNode.label }}</h3>
        <p>{{ selectedNode.file }}<template v-if="selectedNode.line">:{{ selectedNode.line }}</template></p>
        <code v-if="selectedNode.meta?.signature">{{ selectedNode.meta.signature }}</code>
        <dl>
          <template v-if="selectedNode.meta?.inputs?.length">
            <dt>Inputs</dt><dd>{{ formatPorts(selectedNode.meta.inputs) }}</dd>
          </template>
          <template v-if="selectedNode.meta?.outputs?.length">
            <dt>Outputs</dt><dd>{{ selectedNode.meta.outputs.join(', ') }}</dd>
          </template>
        </dl>
        <button class="btn" type="button" :disabled="!selectedNode.file" @click="openCodeNode(selectedNode)">Open source</button>
      </aside>
    </section>

    <div v-if="codeWorkspaceOpen" class="code-workspace" role="dialog" aria-modal="true">
      <header class="workspace-header">
        <div>
          <div class="eyebrow">SOURCE WORKSPACE</div>
          <h2>{{ workspaceMode === 'grid' ? `${codePanels.length} source panels` : activeCodePanel?.title }}</h2>
        </div>
        <div class="workspace-actions">
          <button class="btn ghost" type="button" :class="{ active: workspaceMode === 'single' }" @click="workspaceMode = 'single'">Focus</button>
          <button class="btn ghost" type="button" :class="{ active: workspaceMode === 'grid' }" @click="workspaceMode = 'grid'">Grid</button>
          <button class="btn ghost" type="button" @click="codeWorkspaceOpen = false">Return to graph</button>
        </div>
      </header>

      <main v-if="codePanels.length" class="code-grid" :class="workspaceMode">
        <article
          v-for="panel in visibleCodePanels"
          :key="panel.id"
          class="code-panel"
          :class="{ active: panel.id === activePanelId }"
          :style="workspaceMode === 'grid' ? { gridColumn: `span ${panel.colSpan}`, gridRow: `span ${panel.rowSpan}` } : {}"
          draggable="true"
          @dragstart="draggedPanelId = panel.id"
          @dragover.prevent
          @drop="dropPanel(panel.id)"
          @click="activePanelId = panel.id"
        >
          <header>
            <div>
              <strong>{{ panel.title }}</strong>
              <span>{{ panel.path }}<template v-if="panel.line">:{{ panel.line }}</template></span>
            </div>
            <div class="panel-controls">
              <template v-if="workspaceMode === 'grid'">
                <button type="button" title="Narrower" @click.stop="resizePanel(panel, -2, 0)">−W</button>
                <button type="button" title="Wider" @click.stop="resizePanel(panel, 2, 0)">+W</button>
                <button type="button" title="Shorter" @click.stop="resizePanel(panel, 0, -1)">−H</button>
                <button type="button" title="Taller" @click.stop="resizePanel(panel, 0, 1)">+H</button>
              </template>
              <button type="button" title="Focus panel" @click.stop="focusPanel(panel.id)">□</button>
              <button type="button" title="Close panel" @click.stop="removePanel(panel.id)">×</button>
            </div>
          </header>
          <div class="source-scroll" :data-panel="panel.id">
            <div
              v-for="(line, index) in panel.lines"
              :key="index"
              class="source-line"
              :class="{ target: index + 1 === panel.line }"
              :data-line="index + 1"
            ><span>{{ index + 1 }}</span><code>{{ line || ' ' }}</code></div>
          </div>
        </article>
      </main>
      <div v-else class="workspace-empty">Select a node in the graph to load its source.</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  activateCodeGraphRepository,
  addCodeGraphRepository,
  fetchCodeGraph,
  fetchCodeGraphChunk,
  fetchCodeGraphRepositories,
  fetchCodeGraphSource,
  removeCodeGraphRepository,
} from '@/api';

type GraphNodePayload = {
  id: string; label: string; kind: string; file: string; status: string;
  line?: number; column?: number; meta?: Record<string, any>;
};
type GraphEdgePayload = { id: string; source: string; target: string; kind: string; meta?: Record<string, any> };
type Repository = {
  id: string; name: string; source_type: string; location: string; branch?: string;
  status?: string; progress?: number; message?: string; error?: string; active?: boolean; building?: boolean;
};
type CodePanel = {
  id: string; title: string; path: string; line: number; language: string;
  lines: string[]; colSpan: number; rowSpan: number;
};

const canvasContainer = ref<HTMLElement | null>(null);
const repositories = ref<Repository[]>([]);
const selectedRepositoryId = ref('');
const switching = ref(false);
const addingRepository = ref(false);
const showAddRepository = ref(false);
const repositoryError = ref('');
const repositoryDraft = reactive({ name: '', source_type: 'local' as 'local' | 'github', location: '', branch: '' });
const nodes = ref<GraphNodePayload[]>([]);
const edges = ref<GraphEdgePayload[]>([]);
const entryPoints = ref<string[]>([]);
const summary = ref<Record<string, any>>({});
const graphReady = ref(false);
const graphError = ref('');
const building = ref(false);
const statusState = ref('idle');
const statusProgress = ref(0);
const statusMessage = ref('Select a repository');
const searchText = ref('');
const selectedNode = ref<GraphNodePayload | null>(null);
const hoveredNode = ref<GraphNodePayload | null>(null);
const tooltipX = ref(0);
const tooltipY = ref(0);
const visibleRelationships = ref<string[]>(['contains', 'imports', 'calls', 'inherits']);
const relationshipKinds = ['contains', 'imports', 'calls', 'inherits'];
const nodeKinds = ['repository', 'module', 'file', 'class', 'function', 'method'];
const codeWorkspaceOpen = ref(false);
const workspaceMode = ref<'single' | 'grid'>('single');
const codePanels = ref<CodePanel[]>([]);
const activePanelId = ref('');
const draggedPanelId = ref('');

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let animationFrame = 0;
let pollTimer = 0;
let requestGeneration = 0;
const instancedMeshes: THREE.InstancedMesh[] = [];
const nodePositions = new Map<string, THREE.Vector3>();
const edgeObjects = new Map<string, THREE.Object3D>();
const labelTextures: THREE.Texture[] = [];
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const graphBounds = new THREE.Box3();
const loadedFileDetails = new Set<string>();
const loadingFileDetails = new Set<string>();
let lastDetailLoad = 0;
let highlightHalo: THREE.Mesh | null = null;

const NODE_COLORS: Record<string, number> = {
  repository: 0xf7c65c,
  module: 0x9b7cff,
  file: 0x35a7ff,
  class: 0x35d6a1,
  function: 0xff9f43,
  method: 0xff6b9d,
};
const EDGE_COLORS: Record<string, number> = {
  contains: 0x426987,
  imports: 0x8a7dff,
  calls: 0x27d7c4,
  inherits: 0xffbf69,
};

const activeRepository = computed(() => repositories.value.find((repo) => repo.id === selectedRepositoryId.value) || null);
const tooltipStyle = computed(() => ({ left: `${tooltipX.value + 16}px`, top: `${tooltipY.value + 16}px` }));
const activeCodePanel = computed(() => codePanels.value.find((panel) => panel.id === activePanelId.value) || codePanels.value[0]);
const visibleCodePanels = computed(() => workspaceMode.value === 'single' ? (activeCodePanel.value ? [activeCodePanel.value] : []) : codePanels.value);

async function loadRepositories() {
  const payload = await fetchCodeGraphRepositories();
  repositories.value = payload.repositories || [];
  if (!selectedRepositoryId.value || !repositories.value.some((repo) => repo.id === selectedRepositoryId.value)) {
    selectedRepositoryId.value = payload.active_id || repositories.value[0]?.id || '';
  }
}

async function switchRepository() {
  if (!selectedRepositoryId.value) return;
  switching.value = true;
  requestGeneration += 1;
  clearPoll();
  clearGraph();
  loadedFileDetails.clear();
  loadingFileDetails.clear();
  codePanels.value = [];
  activePanelId.value = '';
  codeWorkspaceOpen.value = false;
  graphError.value = '';
  statusMessage.value = 'Switching repository…';
  try {
    await activateCodeGraphRepository(selectedRepositoryId.value);
    await loadRepositories();
    await refreshGraph(false);
  } catch (error: any) {
    graphError.value = apiError(error);
  } finally {
    switching.value = false;
  }
}

async function removeActiveRepository() {
  if (!selectedRepositoryId.value || repositories.value.length <= 1) return;
  const repositoryId = selectedRepositoryId.value;
  switching.value = true;
  try {
    await removeCodeGraphRepository(repositoryId);
    selectedRepositoryId.value = '';
    await loadRepositories();
    await switchRepository();
  } catch (error: any) {
    repositoryError.value = apiError(error);
  } finally {
    switching.value = false;
  }
}

async function saveRepository() {
  repositoryError.value = '';
  addingRepository.value = true;
  try {
    const payload = await addCodeGraphRepository({ ...repositoryDraft, activate: true });
    selectedRepositoryId.value = payload.repository.id;
    repositoryDraft.name = '';
    repositoryDraft.location = '';
    repositoryDraft.branch = '';
    showAddRepository.value = false;
    await loadRepositories();
    await refreshGraph(false);
  } catch (error: any) {
    repositoryError.value = apiError(error);
  } finally {
    addingRepository.value = false;
  }
}

async function refreshGraph(force = false) {
  const generation = ++requestGeneration;
  graphError.value = '';
  try {
    const payload = await fetchCodeGraph(force, selectedRepositoryId.value);
    if (generation !== requestGeneration) return;
    building.value = Boolean(payload.building);
    statusState.value = payload.status?.state || (building.value ? 'indexing' : 'ready');
    statusProgress.value = Number(payload.status?.progress || 0);
    statusMessage.value = payload.status?.error || payload.status?.message || (building.value ? 'Indexing codebase…' : 'Index ready');
    summary.value = payload.summary || {};
    const retainedDetailNodes = force ? [] : nodes.value.filter((node) => !['repository', 'module', 'file'].includes(node.kind));
    const retainedDetailIds = new Set(retainedDetailNodes.map((node) => node.id));
    const retainedDetailEdges = force ? [] : edges.value.filter((edge) => retainedDetailIds.has(edge.source) || retainedDetailIds.has(edge.target));
    nodes.value = mergeById(payload.nodes || [], retainedDetailNodes);
    edges.value = mergeById(payload.edges || [], retainedDetailEdges);
    entryPoints.value = payload.entry_points || entryPoints.value;
    if (nodes.value.length) {
      const shouldFit = !graphReady.value;
      buildGraphScene(nodes.value, edges.value, shouldFit);
      graphReady.value = true;
      void maybeLoadNearbyDetails(true);
    } else {
      graphReady.value = false;
      clearGraph();
    }
    await loadRepositories();
    schedulePoll();
  } catch (error: any) {
    if (generation !== requestGeneration) return;
    graphError.value = apiError(error);
    statusState.value = 'error';
    statusMessage.value = graphError.value;
    schedulePoll();
  }
}

function schedulePoll() {
  clearPoll();
  if (!building.value) return;
  pollTimer = window.setTimeout(() => refreshGraph(false), 1000);
}

function clearPoll() {
  if (pollTimer) window.clearTimeout(pollTimer);
  pollTimer = 0;
}

async function maybeLoadNearbyDetails(force = false) {
  if (!graphReady.value || !selectedRepositoryId.value || !controls) return;
  const now = performance.now();
  if (!force && now - lastDetailLoad < 900) return;
  lastDetailLoad = now;
  const files = nodes.value.filter((node) => node.kind === 'file');
  const target = controls.target;
  const available = files
    .filter((node) => !loadedFileDetails.has(node.id) && !loadingFileDetails.has(node.id));
  const candidates = available
    .filter((node) => force
      ? Number(node.meta?.entry_priority || 0) > 0
      : (nodePositions.get(node.id)?.distanceToSquared(target) ?? Infinity) <= 450 ** 2)
    .sort((a, b) => {
      const entryDelta = Number(b.meta?.entry_priority || 0) - Number(a.meta?.entry_priority || 0);
      if (entryDelta) return entryDelta;
      return (nodePositions.get(a.id)?.distanceToSquared(target) || Infinity) - (nodePositions.get(b.id)?.distanceToSquared(target) || Infinity);
    })
    .slice(0, force ? 1 : 6);
  if (!candidates.length) return;
  const ids = candidates.map((node) => node.id);
  ids.forEach((id) => loadingFileDetails.add(id));
  try {
    const payload = await fetchCodeGraphChunk(selectedRepositoryId.value, ids);
    const detailNodes = (payload.nodes || []).filter((node: GraphNodePayload) => node.kind !== 'file');
    if (detailNodes.length) {
      nodes.value = mergeById(nodes.value, payload.nodes || []);
      edges.value = mergeById(edges.value, payload.edges || []);
      const filesWithDetail = new Set(detailNodes.map((node: GraphNodePayload) => `file::${node.file}`));
      filesWithDetail.forEach((id) => loadedFileDetails.add(id));
      buildGraphScene(nodes.value, edges.value, false);
    }
  } catch {
    // A partial index may not have reached these files yet; the next camera tick retries.
  } finally {
    ids.forEach((id) => loadingFileDetails.delete(id));
  }
}

function mergeById<T extends { id: string }>(first: T[], second: T[]): T[] {
  const merged = new Map<string, T>();
  first.forEach((item) => merged.set(item.id, item));
  second.forEach((item) => merged.set(item.id, item));
  return [...merged.values()];
}

function initScene() {
  const host = canvasContainer.value;
  if (!host) return;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(host.clientWidth, host.clientHeight);
  renderer.domElement.setAttribute('aria-label', 'Interactive three-dimensional source dependency graph');
  host.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x030812);
  scene.fog = new THREE.FogExp2(0x030812, 0.00018);
  camera = new THREE.PerspectiveCamera(48, host.clientWidth / host.clientHeight, 1, 30000);
  camera.position.set(0, 1500, 2400);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.minDistance = 80;
  controls.maxDistance = 14000;
  scene.add(new THREE.AmbientLight(0xbad8ff, 1.45));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
  keyLight.position.set(800, 1400, 900);
  scene.add(keyLight);
  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('pointerleave', clearHover);
  renderer.domElement.addEventListener('click', onGraphClick);
  window.addEventListener('resize', resizeRenderer);
  animate();
}

function clearGraph() {
  selectedNode.value = null;
  hoveredNode.value = null;
  graphReady.value = false;
  instancedMeshes.splice(0);
  nodePositions.clear();
  edgeObjects.clear();
  highlightHalo = null;
  labelTextures.splice(0).forEach((texture) => texture.dispose());
  if (!scene) return;
  const keep = scene.children.filter((child) => child.type.includes('Light'));
  scene.children.slice().forEach((child) => {
    if (!keep.includes(child)) {
      disposeObject(child);
      scene!.remove(child);
    }
  });
}

function buildGraphScene(graphNodes: GraphNodePayload[], graphEdges: GraphEdgePayload[], fit = true) {
  const savedCamera = camera?.position.clone();
  const savedTarget = controls?.target.clone();
  clearGraph();
  if (!scene) return;
  const positions = computeCentricLayout(graphNodes, graphEdges);
  graphBounds.makeEmpty();
  positions.forEach((position, nodeId) => {
    nodePositions.set(nodeId, position);
    graphBounds.expandByPoint(position);
  });
  createInstancedNodes(graphNodes, positions);
  createEdges(graphEdges, positions);
  addReferenceAxes();
  highlightHalo = new THREE.Mesh(
    new THREE.SphereGeometry(34, 18, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: .8 }),
  );
  highlightHalo.visible = false;
  scene.add(highlightHalo);
  applyRelationshipVisibility();
  graphReady.value = true;
  if (fit) nextTick(() => fitGraph(false));
  else if (savedCamera && savedTarget && camera && controls) {
    camera.position.copy(savedCamera);
    controls.target.copy(savedTarget);
    controls.update();
  }
}

function computeCentricLayout(graphNodes: GraphNodePayload[], graphEdges: GraphEdgePayload[]) {
  const positions = new Map<string, THREE.Vector3>();
  const children = new Map<string, string[]>();
  graphEdges.filter((edge) => edge.kind === 'contains').forEach((edge) => {
    if (!children.has(edge.source)) children.set(edge.source, []);
    children.get(edge.source)!.push(edge.target);
  });
  const repository = graphNodes.find((node) => node.kind === 'repository');
  if (!repository) return positions;
  positions.set(repository.id, new THREE.Vector3(0, -180, 0));
  const modules = (children.get(repository.id) || []).sort();
  const moduleRadius = Math.max(650, modules.length * 145);
  modules.forEach((moduleId, moduleIndex) => {
    const angle = (moduleIndex / Math.max(1, modules.length)) * Math.PI * 2;
    const modulePosition = new THREE.Vector3(Math.cos(angle) * moduleRadius, 0, Math.sin(angle) * moduleRadius);
    positions.set(moduleId, modulePosition);
    const files = (children.get(moduleId) || []).sort();
    const ringCapacity = 12;
    files.forEach((fileId, fileIndex) => {
      const ring = Math.floor(fileIndex / ringCapacity);
      const slot = fileIndex % ringCapacity;
      const capacity = Math.min(ringCapacity, files.length - ring * ringCapacity);
      const localAngle = angle - 0.85 + (slot / Math.max(1, capacity - 1)) * 1.7;
      const radius = 240 + ring * 180;
      const y = ((fileIndex % 3) - 1) * 90;
      const filePosition = modulePosition.clone().add(new THREE.Vector3(Math.cos(localAngle) * radius, y, Math.sin(localAngle) * radius));
      positions.set(fileId, filePosition);
      const symbols = (children.get(fileId) || []).sort();
      symbols.forEach((symbolId, symbolIndex) => {
        const layer = Math.floor(symbolIndex / 10);
        const symbolAngle = (symbolIndex % 10) / Math.min(10, Math.max(1, symbols.length - layer * 10)) * Math.PI * 2;
        const symbolRadius = 100 + layer * 72;
        const symbolPosition = filePosition.clone().add(new THREE.Vector3(
          Math.cos(symbolAngle) * symbolRadius,
          (layer % 2 === 0 ? 1 : -1) * (80 + layer * 42),
          Math.sin(symbolAngle) * symbolRadius,
        ));
        positions.set(symbolId, symbolPosition);
        const nested = children.get(symbolId) || [];
        nested.forEach((nestedId, nestedIndex) => {
          positions.set(nestedId, symbolPosition.clone().add(new THREE.Vector3((nestedIndex - nested.length / 2) * 52, 90, 0)));
        });
      });
    });
  });
  const primaryEntry = graphNodes
    .filter((node) => node.kind === 'file' && node.meta?.entry_point)
    .sort((a, b) => Number(b.meta?.entry_priority || 0) - Number(a.meta?.entry_priority || 0))[0];
  if (primaryEntry) positions.set(primaryEntry.id, new THREE.Vector3(0, 0, 0));
  return positions;
}

function createInstancedNodes(graphNodes: GraphNodePayload[], positions: Map<string, THREE.Vector3>) {
  if (!scene) return;
  const grouped = new Map<string, GraphNodePayload[]>();
  graphNodes.forEach((node) => {
    if (!grouped.has(node.kind)) grouped.set(node.kind, []);
    grouped.get(node.kind)!.push(node);
  });
  grouped.forEach((kindNodes, kind) => {
    const size = nodeSize(kind);
    const geometry = geometryForKind(kind, size);
    const material = new THREE.MeshStandardMaterial({
      color: NODE_COLORS[kind] || 0x86a6c9,
      emissive: NODE_COLORS[kind] || 0x86a6c9,
      emissiveIntensity: .12,
      metalness: .22,
      roughness: .5,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, kindNodes.length);
    mesh.userData.nodes = kindNodes;
    const matrix = new THREE.Matrix4();
    kindNodes.forEach((node, index) => {
      const position = positions.get(node.id) || new THREE.Vector3();
      matrix.makeTranslation(position.x, position.y, position.z);
      mesh.setMatrixAt(index, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    scene!.add(mesh);
    instancedMeshes.push(mesh);
    const labelEveryNode = graphNodes.length < 1800;
    if (labelEveryNode || ['repository', 'module', 'file'].includes(kind)) {
      kindNodes.forEach((node) => {
        const label = createTextLabel(node.label, node.kind, kind === 'repository' ? 1.45 : kind === 'module' ? 1.15 : .82);
        const position = positions.get(node.id) || new THREE.Vector3();
        label.position.copy(position).add(new THREE.Vector3(0, size + 18, 0));
        scene!.add(label);
      });
    }
  });
}

function nodeSize(kind: string) {
  return kind === 'repository' ? 58 : kind === 'module' ? 42 : kind === 'file' ? 32 : kind === 'class' ? 25 : 18;
}

function geometryForKind(kind: string, size: number): THREE.BufferGeometry {
  if (kind === 'repository') return new THREE.IcosahedronGeometry(size, 2);
  if (kind === 'module') return new THREE.OctahedronGeometry(size, 1);
  if (kind === 'file') return new THREE.BoxGeometry(size * 1.35, size * .34, size);
  if (kind === 'class') return new THREE.BoxGeometry(size * 1.25, size, size * .55);
  if (kind === 'method') return new THREE.CylinderGeometry(size * .72, size * .72, size * .55, 8);
  return new THREE.SphereGeometry(size * .72, 12, 8);
}

function createTextLabel(text: string, subtitle: string, scale: number) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d')!;
  context.fillStyle = 'rgba(3, 9, 20, .88)';
  context.roundRect(1, 1, 254, 62, 12);
  context.fill();
  context.strokeStyle = 'rgba(132, 194, 255, .5)';
  context.stroke();
  context.fillStyle = '#eaf5ff';
  context.font = '600 18px Segoe UI, sans-serif';
  context.textAlign = 'center';
  context.fillText(ellipsize(text, 25), 128, 27);
  context.fillStyle = '#83bde7';
  context.font = '11px Segoe UI, sans-serif';
  context.fillText(subtitle.toUpperCase(), 128, 47);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  labelTextures.push(texture);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
  sprite.scale.set(150 * scale, 37.5 * scale, 1);
  return sprite;
}

function createEdges(graphEdges: GraphEdgePayload[], positions: Map<string, THREE.Vector3>) {
  if (!scene) return;
  relationshipKinds.forEach((kind) => {
    const points: number[] = [];
    graphEdges.filter((edge) => edge.kind === kind).forEach((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return;
      points.push(source.x, source.y, source.z, target.x, target.y, target.z);
    });
    if (!points.length) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const material = new THREE.LineBasicMaterial({
      color: EDGE_COLORS[kind] || 0x6b91b5,
      transparent: true,
      opacity: kind === 'contains' ? 0.24 : 0.68,
    });
    const lines = new THREE.LineSegments(geometry, material);
    lines.userData.relationship = kind;
    scene!.add(lines);
    edgeObjects.set(kind, lines);
  });
}

function addReferenceAxes() {
  if (!scene) return;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(310, 312, 128),
    new THREE.MeshBasicMaterial({ color: 0x173a61, transparent: true, opacity: 0.35, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  scene.add(ring);
}

function computeIo(graphEdges: GraphEdgePayload[]) {
  const map = new Map<string, { inbound: number; outbound: number }>();
  graphEdges.filter((edge) => edge.kind !== 'contains').forEach((edge) => {
    if (!map.has(edge.source)) map.set(edge.source, { inbound: 0, outbound: 0 });
    if (!map.has(edge.target)) map.set(edge.target, { inbound: 0, outbound: 0 });
    map.get(edge.source)!.outbound += 1;
    map.get(edge.target)!.inbound += 1;
  });
  return map;
}

function applyRelationshipVisibility() {
  edgeObjects.forEach((object, kind) => { object.visible = visibleRelationships.value.includes(kind); });
}

function raycast(event: PointerEvent) {
  if (!renderer || !camera) return null;
  const bounds = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(instancedMeshes, false)[0] as THREE.Intersection<THREE.InstancedMesh> | undefined;
  if (!hit || hit.instanceId === undefined) return null;
  return (hit.object.userData.nodes?.[hit.instanceId] || null) as GraphNodePayload | null;
}

function onPointerMove(event: PointerEvent) {
  if (!renderer) return;
  const bounds = renderer.domElement.getBoundingClientRect();
  tooltipX.value = event.clientX - bounds.left;
  tooltipY.value = event.clientY - bounds.top;
  const node = raycast(event);
  if (node?.id !== hoveredNode.value?.id) {
    setHighlighted(hoveredNode.value?.id, false);
    hoveredNode.value = node;
    setHighlighted(node?.id, true);
  }
  renderer.domElement.style.cursor = node ? 'pointer' : 'grab';
}

function clearHover() {
  setHighlighted(hoveredNode.value?.id, false);
  hoveredNode.value = null;
}

async function onGraphClick(event: PointerEvent) {
  const node = raycast(event);
  if (!node) return;
  selectedNode.value = node;
  focusNode(node.id);
  if (node.kind === 'file') void loadFileDetails([node.id]);
  if (node.file) await openCodeNode(node);
}

async function loadFileDetails(ids: string[]) {
  const pending = ids.filter((id) => !loadedFileDetails.has(id) && !loadingFileDetails.has(id));
  if (!pending.length || !selectedRepositoryId.value) return;
  pending.forEach((id) => loadingFileDetails.add(id));
  try {
    const payload = await fetchCodeGraphChunk(selectedRepositoryId.value, pending);
    const detailNodes = (payload.nodes || []).filter((item: GraphNodePayload) => item.kind !== 'file');
    if (detailNodes.length) {
      nodes.value = mergeById(nodes.value, payload.nodes || []);
      edges.value = mergeById(edges.value, payload.edges || []);
      pending.forEach((id) => loadedFileDetails.add(id));
      buildGraphScene(nodes.value, edges.value, false);
    }
  } catch {
    // The index may still be reaching this file; proximity streaming will retry.
  } finally {
    pending.forEach((id) => loadingFileDetails.delete(id));
  }
}

function setHighlighted(nodeId: string | undefined, active: boolean) {
  if (!highlightHalo) return;
  if (!active || !nodeId) {
    highlightHalo.visible = false;
    return;
  }
  const position = nodePositions.get(nodeId);
  if (!position) return;
  highlightHalo.position.copy(position);
  highlightHalo.visible = true;
}

function focusNode(nodeId: string) {
  const position = nodePositions.get(nodeId);
  if (!position || !camera || !controls) return;
  const target = position.clone();
  const direction = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(target);
  camera.position.copy(target.clone().add(direction.multiplyScalar(430)));
  controls.update();
}

function focusFirstSearch() {
  const query = searchText.value.trim().toLowerCase();
  if (!query) return;
  const node = nodes.value.find((candidate) => `${candidate.label} ${candidate.file}`.toLowerCase().includes(query));
  if (node) {
    selectedNode.value = node;
    focusNode(node.id);
    setHighlighted(node.id, true);
  }
}

async function openCodeNode(node: GraphNodePayload) {
  if (!node.file || !selectedRepositoryId.value) return;
  activePanelId.value = node.id;
  codeWorkspaceOpen.value = true;
  let panel = codePanels.value.find((item) => item.id === node.id);
  if (!panel) {
    try {
      const source = await fetchCodeGraphSource(selectedRepositoryId.value, node.file);
      panel = {
        id: node.id, title: node.label, path: source.path, line: Number(node.line || 1),
        language: source.language, lines: source.content.split('\n'), colSpan: 6, rowSpan: 1,
      };
      codePanels.value.push(panel);
    } catch (error: any) {
      graphError.value = apiError(error);
      codeWorkspaceOpen.value = false;
      return;
    }
  }
  await nextTick();
  scrollPanelToLine(panel.id, panel.line);
}

function scrollPanelToLine(panelId: string, line: number) {
  const container = document.querySelector(`[data-panel="${CSS.escape(panelId)}"]`);
  const target = container?.querySelector(`[data-line="${line}"]`);
  target?.scrollIntoView({ block: 'center' });
}

function focusPanel(panelId: string) {
  activePanelId.value = panelId;
  workspaceMode.value = 'single';
  nextTick(() => {
    const panel = codePanels.value.find((item) => item.id === panelId);
    if (panel) scrollPanelToLine(panel.id, panel.line);
  });
}

function removePanel(panelId: string) {
  codePanels.value = codePanels.value.filter((panel) => panel.id !== panelId);
  if (activePanelId.value === panelId) activePanelId.value = codePanels.value[0]?.id || '';
  if (!codePanels.value.length) codeWorkspaceOpen.value = false;
}

function resizePanel(panel: CodePanel, widthDelta: number, heightDelta: number) {
  panel.colSpan = Math.max(2, Math.min(12, panel.colSpan + widthDelta));
  panel.rowSpan = Math.max(1, Math.min(3, panel.rowSpan + heightDelta));
}

function dropPanel(targetId: string) {
  const sourceId = draggedPanelId.value;
  if (!sourceId || sourceId === targetId) return;
  const sourceIndex = codePanels.value.findIndex((panel) => panel.id === sourceId);
  const targetIndex = codePanels.value.findIndex((panel) => panel.id === targetId);
  if (sourceIndex < 0 || targetIndex < 0) return;
  const next = [...codePanels.value];
  const [panel] = next.splice(sourceIndex, 1);
  next.splice(targetIndex, 0, panel);
  codePanels.value = next;
  draggedPanelId.value = '';
}

function fitGraph(animate = true) {
  if (!camera || !controls || graphBounds.isEmpty()) return;
  const center = graphBounds.getCenter(new THREE.Vector3());
  const size = graphBounds.getSize(new THREE.Vector3());
  const distance = Math.max(800, Math.max(size.x, size.y, size.z) * 1.15);
  controls.target.copy(center);
  camera.position.copy(center.clone().add(new THREE.Vector3(distance * 0.22, distance * 0.7, distance)));
  camera.near = Math.max(0.5, distance / 10000);
  camera.far = distance * 8;
  camera.updateProjectionMatrix();
  controls.update();
  void animate;
}

function resizeRenderer() {
  const host = canvasContainer.value;
  if (!host || !renderer || !camera) return;
  renderer.setSize(host.clientWidth, host.clientHeight);
  camera.aspect = host.clientWidth / host.clientHeight;
  camera.updateProjectionMatrix();
}

function animate() {
  animationFrame = requestAnimationFrame(animate);
  controls?.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
  void maybeLoadNearbyDetails(false);
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child: any) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) child.material.forEach((material: any) => material.dispose?.());
    else child.material?.dispose?.();
  });
}

function nodeColorCss(kind: string) {
  return `#${(NODE_COLORS[kind] || 0x86a6c9).toString(16).padStart(6, '0')}`;
}

function ioLabel(node: GraphNodePayload) {
  const inputCount = node.meta?.inputs?.length || 0;
  const outputCount = node.meta?.outputs?.length || 0;
  return `${inputCount} declared inputs · ${outputCount} declared outputs`;
}

function formatPorts(inputs: Array<Record<string, any>>) {
  return inputs.map((input) => input.type ? `${input.name}: ${input.type}` : input.name).join(', ');
}

function ellipsize(value: string, length: number) {
  return value.length <= length ? value : `${value.slice(0, length - 1)}…`;
}

function apiError(error: any) {
  return error?.response?.data?.detail || error?.message || 'Unable to load CodeGraph';
}

onMounted(async () => {
  initScene();
  try {
    await loadRepositories();
    await refreshGraph(false);
  } catch (error: any) {
    graphError.value = apiError(error);
  }
});

onBeforeUnmount(() => {
  requestGeneration += 1;
  clearPoll();
  cancelAnimationFrame(animationFrame);
  window.removeEventListener('resize', resizeRenderer);
  if (renderer) {
    renderer.domElement.removeEventListener('pointermove', onPointerMove);
    renderer.domElement.removeEventListener('pointerleave', clearHover);
    renderer.domElement.removeEventListener('click', onGraphClick);
    renderer.dispose();
  }
  clearGraph();
});
</script>

<style scoped>
.codegraph-view { display: flex; flex-direction: column; gap: 1rem; min-height: 0; }
.panel { background: rgba(4, 11, 22, .94); border: 1px solid rgba(83, 157, 229, .24); border-radius: 16px; }
.repository-bar { padding: 1rem 1.2rem; display: flex; gap: 1rem; align-items: end; flex-wrap: wrap; }
.repository-picker { display: grid; gap: .28rem; min-width: min(480px, 100%); flex: 1; }
.repository-picker select, .repository-form input, .repository-form select, .search-field input { background: #071426; color: #eef8ff; border: 1px solid #254d72; border-radius: 8px; padding: .65rem .75rem; }
.source-path { color: #83a4be; font: .75rem ui-monospace, SFMono-Regular, Consolas, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.repository-actions { display: flex; gap: .55rem; }
.eyebrow { color: #64b5f6; font-size: .68rem; letter-spacing: .16em; font-weight: 700; }
.repository-form { width: 100%; display: grid; grid-template-columns: 150px 1fr minmax(280px, 2fr) 150px auto auto; gap: .65rem; align-items: end; padding-top: .8rem; border-top: 1px solid rgba(100, 181, 246, .16); }
.repository-form label { display: grid; gap: .3rem; color: #a9bfd0; font-size: .75rem; }
.form-error { grid-column: 1 / -1; color: #ff7b86; margin: 0; }
.graph-status { padding: .75rem 1.1rem; display: grid; gap: .65rem; }
.status-copy { display: flex; gap: .55rem; align-items: center; color: #bcd1e0; }
.status-copy strong { color: #eef8ff; }
.status-dot { width: 9px; height: 9px; border-radius: 50%; background: #648099; box-shadow: 0 0 10px currentColor; }
.status-dot.ready { background: #24d39a; }.status-dot.indexing,.status-dot.preparing,.status-dot.cloning { background: #56b8ff; animation: pulse 1s infinite; }.status-dot.error { background: #ff5363; }
.progress-track { height: 5px; background: #0d2136; border-radius: 8px; overflow: hidden; }.progress-track span { display: block; height: 100%; background: linear-gradient(90deg,#498cff,#35d6a1); transition: width .25s; }
.summary { display: flex; flex-wrap: wrap; gap: 1.1rem; color: #7f9bb0; font-size: .8rem; }.summary b { color: #e9f5ff; font-size: 1rem; }
.graph-shell { position: relative; min-height: 680px; overflow: hidden; }
.canvas-host { height: clamp(680px, 74vh, 1120px); width: 100%; position: relative; }
.graph-tools { position: absolute; z-index: 3; top: 1rem; left: 1rem; width: min(290px, calc(100% - 2rem)); padding: .8rem; background: rgba(3, 10, 21, .88); border: 1px solid rgba(93, 168, 235, .3); border-radius: 12px; backdrop-filter: blur(10px); }
.search-field { display: grid; gap: .35rem; color: #9db7ca; font-size: .72rem; }
.relation-filters,.legend { display: flex; flex-wrap: wrap; gap: .55rem .8rem; margin-top: .7rem; font-size: .7rem; color: #b8cad8; }
.relation-filters label,.legend span { display: flex; align-items: center; gap: .28rem; }
.wire-key { width: 18px; height: 2px; background: #6385a3; }.wire-key.imports { background:#8a7dff }.wire-key.calls { background:#27d7c4 }.wire-key.inherits { background:#ffbf69 }
.legend i { width: 8px; height: 8px; border-radius: 50%; }
.node-tooltip { position: absolute; z-index: 5; pointer-events: none; display: grid; gap: .22rem; min-width: 220px; max-width: 340px; background: rgba(2, 8, 17, .96); border: 1px solid #32648f; border-radius: 10px; padding: .7rem .8rem; box-shadow: 0 18px 46px #000b; }
.node-tooltip strong { color:#f3f9ff }.node-tooltip span,.node-tooltip small { color:#8facbf; font-size:.72rem }.node-tooltip small { color:#54d6af }.tooltip-kind { color:#6dbbfa; letter-spacing:.12em; font-size:.62rem; text-transform:uppercase; }
.selection-card { position:absolute; z-index:3; right:1rem; bottom:1rem; width:min(360px,calc(100% - 2rem)); padding:1rem; background:rgba(3,10,21,.94); border:1px solid rgba(93,168,235,.38); border-radius:12px; box-shadow:0 18px 50px #000a; }
.selection-card h3 { margin:.3rem 0; color:#eef8ff }.selection-card p { color:#87a5ba; font:.75rem ui-monospace,monospace; word-break:break-all }.selection-card code { display:block; color:#59dcb3; white-space:pre-wrap }.selection-card dl { font-size:.78rem }.selection-card dt { color:#6dbbfa }.selection-card dd { margin:0 0 .45rem; color:#bdcfdb }.close-mini { float:right; border:0; background:none; color:#9db5c8; font-size:1.25rem; cursor:pointer; }
.canvas-message { position:absolute; inset:0; display:grid; place-items:center; color:#8ca9bd; pointer-events:none }.canvas-message.error { color:#ff6d79; }
.code-workspace { position:fixed; z-index:10000; inset:0; background:#02060d; display:flex; flex-direction:column; color:#edf7ff; }
.workspace-header { min-height:72px; display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.85rem 1.2rem; border-bottom:1px solid #17324d; background:#06101d; }.workspace-header h2 { margin:.2rem 0 0; font-size:1.05rem }.workspace-actions { display:flex; gap:.5rem }.workspace-actions .active { border-color:#48b7ff; color:#7fcfff; }
.code-grid { flex:1; min-height:0; overflow:auto; padding:.8rem; display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); grid-auto-rows:minmax(390px, 46vh); gap:.8rem; }.code-grid.single { display:block; overflow:hidden; }.code-grid.single .code-panel { height:100%; }
.code-panel { min-width:0; min-height:0; display:flex; flex-direction:column; background:#050b13; border:1px solid #183754; border-radius:10px; overflow:hidden; }.code-panel.active { border-color:#3f9dde; box-shadow:0 0 0 1px #3f9dde55; }.code-panel>header { display:flex; justify-content:space-between; gap:.7rem; padding:.55rem .7rem; background:#091727; border-bottom:1px solid #17334d; cursor:grab; }.code-panel>header>div:first-child { min-width:0; display:grid; gap:.15rem }.code-panel header strong { color:#dff2ff }.code-panel header span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#7798af; font:.68rem ui-monospace,monospace; }
.panel-controls { display:flex; align-items:center; gap:.22rem }.panel-controls button { border:1px solid #2d506d; background:#0c2032; color:#a9c5d8; border-radius:5px; min-width:28px; height:26px; cursor:pointer; }
.source-scroll { flex:1; min-height:0; overflow:auto; font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; counter-reset:line; padding:.5rem 0 2rem; }.source-line { display:grid; grid-template-columns:58px minmax(max-content,1fr); min-height:1.55em; white-space:pre; }.source-line>span { position:sticky; left:0; z-index:1; text-align:right; padding-right:12px; color:#425e73; background:#050b13; user-select:none }.source-line code { color:#c8d9e5; padding-right:1rem }.source-line.target { background:#2b401b88; box-shadow:inset 3px 0 #b7e85b }.source-line.target>span { color:#d1ef8d; background:#14210f; }
.workspace-empty { flex:1; display:grid; place-items:center; color:#7795aa; }
.btn { border:1px solid #2f7db5; background:#0b4e7d; color:#eaf7ff; border-radius:8px; padding:.55rem .8rem; cursor:pointer }.btn.ghost { background:#0a1928; border-color:#294d69 }.btn:disabled { opacity:.48; cursor:not-allowed }
@keyframes pulse { 50% { opacity:.35 } }
@media (max-width: 900px) { .repository-form { grid-template-columns:1fr 1fr }.location-field,.form-error { grid-column:1/-1 }.graph-shell { min-height:560px }.canvas-host { height:70vh }.graph-tools { width:240px }.code-grid { grid-template-columns:1fr; }.code-panel { grid-column:1!important; }.workspace-header { align-items:flex-start; flex-direction:column; }.workspace-actions { flex-wrap:wrap; } }
</style>
