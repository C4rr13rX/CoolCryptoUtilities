<template>
  <div class="codegraph-view">
    <section class="panel toolbar">
      <div class="controls">
        <button class="btn" type="button" @click="refreshGraph(true)" :disabled="loading || building">
          <span v-if="loading">Loading…</span>
          <span v-else-if="building">Building…</span>
          <span v-else>Refresh Graph</span>
        </button>
        <button class="btn ghost" type="button" @click="resetView" :disabled="loading">
          Reset View
        </button>
        <button class="btn warning" type="button" @click="captureSnapshots('warnings')" :disabled="capturing || !graphReady">
          Snapshot Warnings
        </button>
        <button class="btn danger" type="button" @click="captureSnapshots('errors')" :disabled="capturing || !graphReady">
          Snapshot Errors
        </button>
      </div>
      <div class="loading-bar" v-if="loading || building || loadingProgress > 0">
        <div class="bar">
          <span class="fill" :style="{ width: `${Math.min(loadingProgress * 100, 100)}%` }"></span>
        </div>
        <div class="loading-meta">
          <span class="file-name">{{ loadingLabel }}</span>
          <span class="build-name">{{ buildingLabel }}</span>
        </div>
      </div>
      <div class="summary">
        <div>
          <span class="label">Files</span>
          <span class="value">{{ summary.files || 0 }}</span>
        </div>
        <div>
          <span class="label">Classes</span>
          <span class="value">{{ summary.classes || 0 }}</span>
        </div>
        <div>
          <span class="label">Functions</span>
          <span class="value">{{ summary.functions || 0 }}</span>
        </div>
        <div>
          <span class="label">Warnings</span>
          <span class="value warn">{{ warnings.length }}</span>
        </div>
        <div>
          <span class="label">Errors</span>
          <span class="value error">{{ errors.length }}</span>
        </div>
      </div>
    </section>
    <section class="panel canvas-panel">
      <div ref="canvasContainer" class="canvas-host"></div>
      <div v-if="!graphReady" class="hint">Graph data will appear here once loaded.</div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, nextTick, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { fetchCodeGraph, fetchCodeGraphFiles, uploadCodeGraphSnapshot } from '@/api';

const canvasContainer = ref<HTMLElement | null>(null);
const loading = ref(false);
const capturing = ref(false);
const graphReady = ref(false);
const summary = ref<Record<string, number>>({});
const warnings = ref<string[]>([]);
const errors = ref<string[]>([]);
const nodes = ref<any[]>([]);
const building = ref(false);
const fileNames = ref<string[]>([]);
const currentFileName = ref('');
const loadingProgress = ref(0);
const loadingPhase = ref<'idle' | 'loading' | 'building' | 'done'>('idle');
const loadingLabel = ref('Idle');
const buildingLabel = ref('Awaiting build plan…');

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let animationId: number | null = null;
let lastInteraction = performance.now();
let defaultCameraPos = new THREE.Vector3(0, 1200, 0);
let defaultTarget = new THREE.Vector3(0, 0, 0);
let inactivityResetMs = 8000;
const nodeMeshes = new Map<string, THREE.Object3D>();
const STATUS_COLORS: Record<string, number> = {
  ok: 0x4c8eda,
  unused: 0xffa726,
  broken: 0xef5350,
};
let loadingTimer: number | null = null;
let progressTimer: number | null = null;
let buildingLabelTimer: number | null = null;
let pollTimer: number | null = null;
let fileIndex = 0;
let buildingIndex = 0;
let phaseStart = performance.now();
const LOADING_EXPECTED_MS = 9000;
const BUILD_EXPECTED_MS = 15000;
const BUILDING_MESSAGES = [
  'Linking call graph…',
  'Resolving class hierarchies…',
  'Indexing functions…',
  'Scoring unused definitions…',
  'Drawing flow channels…',
  'Assembling viewport chunks…',
];

async function refreshGraph(force = false) {
  loading.value = true;
  startLoadingTicker();
  setPhase('loading');
  try {
    const data = await fetchCodeGraph(force);
    building.value = Boolean(data?.building);
    summary.value = data?.summary || {};
    warnings.value = data?.warnings || [];
    errors.value = data?.errors || [];
    nodes.value = data?.nodes || [];
    graphReady.value = Boolean(nodes.value.length);
    if (graphReady.value) {
      buildScene(data);
    }
    if (!graphReady.value) {
await resetView();
    }
    schedulePoll();
  } finally {
    stopLoadingTicker();
    loading.value = false;
  }
}

function initScene() {
  if (!canvasContainer.value) return;
  const container = canvasContainer.value;
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050b16);

  camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 10, 6000);
  camera.position.copy(defaultCameraPos);
  camera.lookAt(defaultTarget);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableRotate = false;
  controls.enablePan = true;
  controls.maxPolarAngle = Math.PI / 2;
  controls.minPolarAngle = Math.PI / 2;
  controls.addEventListener('change', () => markInteraction());
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('wheel', () => markInteraction(), { passive: true });

  const ambient = new THREE.AmbientLight(0xffffff, 0.9);
  scene.add(ambient);
  const dir = new THREE.DirectionalLight(0xffffff, 0.45);
  dir.position.set(500, 1000, -500);
  scene.add(dir);

  window.addEventListener('resize', handleResize);
  animate();
}

function clearScene() {
  if (!scene) return;
  const lights = scene.children.filter((child) => child.type.toLowerCase().includes('light'));
  scene.children.slice().forEach((child) => {
    if (!lights.includes(child)) {
      scene!.remove(child);
    }
  });
  nodeMeshes.clear();
}

function buildScene(data: any) {
  if (!scene) return;
  clearScene();

  const planeGeometry = new THREE.PlaneGeometry(4000, 2000);
  const planeMaterial = new THREE.MeshBasicMaterial({ color: 0x0b1424, side: THREE.DoubleSide });
  const plane = new THREE.Mesh(planeGeometry, planeMaterial);
  plane.rotation.x = -Math.PI / 2;
  scene.add(plane);

  const gridHelper = new THREE.GridHelper(4000, 40, 0x123456, 0x123456);
  scene.add(gridHelper);

  const fileNodes = data.nodes.filter((node: any) => node.kind === 'file');
  const layout = computeLayout(fileNodes);
  const nodeById: Record<string, any> = {};
  data.nodes.forEach((node: any) => {
    nodeById[node.id] = node;
  });

  // draw file nodes
  fileNodes.forEach((file: any, idx: number) => {
    const pos = layout[file.id];
    const mesh = createFileMesh(file, pos);
    scene!.add(mesh);
    nodeMeshes.set(file.id, mesh);
    placeChildren(file, data.nodes, pos);
  });

  drawEdges(data.edges);
}

function computeLayout(files: any[]) {
  const layout: Record<string, THREE.Vector3> = {};
  if (!files.length) {
    layout['default'] = new THREE.Vector3(0, 0, 0);
    return layout;
  }
  const columns = Math.max(1, Math.ceil(Math.sqrt(files.length * 2)));
  const spacingX = 220;
  const spacingZ = spacingX / 2;
  files.forEach((file, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const x = (col - columns / 2) * spacingX * 1.1;
    const z = (row - Math.ceil(files.length / columns) / 2) * spacingZ * 1.4;
    layout[file.id] = new THREE.Vector3(x, 2, z);
  });
  return layout;
}

function createFileMesh(node: any, position: THREE.Vector3) {
  const geom = new THREE.BoxGeometry(140, 8, 90);
  const material = new THREE.MeshStandardMaterial({
    color: 0x0f1f36,
    transparent: true,
    opacity: 0.85,
  });
  const mesh = new THREE.Mesh(geom, material);
  mesh.position.copy(position);
  mesh.userData = { node };
  return mesh;
}

function placeChildren(fileNode: any, allNodes: any[], basePosition: THREE.Vector3) {
  const children = allNodes.filter(
    (node) => node.file === fileNode.file && node.id !== fileNode.id && node.kind !== 'file'
  );
  if (!children.length || !scene) return;
  const maxPerRow = Math.max(1, Math.floor(Math.sqrt(children.length * 2)));
  const offsetX = 30;
  const offsetZ = 30;
  children.forEach((child, index) => {
    const col = index % maxPerRow;
    const row = Math.floor(index / maxPerRow);
    const x = basePosition.x + (col - maxPerRow / 2) * offsetX;
    const z = basePosition.z + (row - Math.ceil(children.length / maxPerRow) / 2) * offsetZ;
    const mesh = createNodeMesh(child, new THREE.Vector3(x, basePosition.y + 12, z));
    scene!.add(mesh);
    nodeMeshes.set(child.id, mesh);
  });
}

function createNodeMesh(node: any, position: THREE.Vector3) {
  const color = STATUS_COLORS[node.status] || STATUS_COLORS.ok;
  const geometry =
    node.kind === 'class'
      ? new THREE.CylinderGeometry(8, 8, 10, 16)
      : new THREE.SphereGeometry(node.kind === 'method' ? 5 : 7, 20, 20);
  const material = new THREE.MeshStandardMaterial({ color, emissive: 0x000000 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(position);
  mesh.userData = { node };
  return mesh;
}

function drawEdges(edges: any[]) {
  if (!scene) return;
  const material = new THREE.LineBasicMaterial({ color: 0x1f3a70, linewidth: 1 });
  const callMaterial = new THREE.LineBasicMaterial({ color: 0x2ec4b6, linewidth: 1 });
  edges.forEach((edge) => {
    const from = nodeMeshes.get(edge.source);
    const to = nodeMeshes.get(edge.target);
    if (!from || !to) return;
    const points = [from.position.clone(), to.position.clone()];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(geometry, edge.kind === 'calls' ? callMaterial : material);
    scene!.add(line);
  });
}

function handleResize() {
  if (!renderer || !camera || !canvasContainer.value) return;
  const container = canvasContainer.value;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
  if (!renderer || !scene || !camera) return;
  animationId = requestAnimationFrame(animate);
  renderer.render(scene, camera);
  if (performance.now() - lastInteraction > inactivityResetMs) {
    lastInteraction = performance.now();
    resetView();
  }
}

function markInteraction() {
  lastInteraction = performance.now();
}

function onPointerDown(event: PointerEvent) {
  markInteraction();
  if (!renderer || !camera || !scene) return;
  const bounds = renderer.domElement.getBoundingClientRect();
  const pointer = new THREE.Vector2(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const intersects = raycaster.intersectObjects(Array.from(nodeMeshes.values()), true);
  if (intersects.length) {
    const hit = intersects[0].object;
    const node = hit.userData?.node;
    if (node) {
      focusOnNode(node.id);
    }
  }
}

async function focusOnNode(nodeId: string, zoomFactor = 0.35, duration = 700) {
  const mesh = nodeMeshes.get(nodeId);
  if (!mesh || !camera || !controls) return;
  const target = mesh.position.clone();
  const offset = new THREE.Vector3(0, 350, 0);
  const direction = target.clone().sub(new THREE.Vector3(0, 0, 0));
  direction.y = 0;
  direction.normalize().multiplyScalar(200 * zoomFactor);
  const destination = target.clone().add(offset).add(direction);
  await animateCamera(destination, target, duration);
}

async function animateCamera(position: THREE.Vector3, target: THREE.Vector3, duration = 600) {
  if (!camera || !controls) return;
  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const start = performance.now();
  return new Promise<void>((resolve) => {
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = t * (2 - t);
      camera!.position.lerpVectors(startPos, position, eased);
      controls!.target.lerpVectors(startTarget, target, eased);
      controls!.update();
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        resolve();
      }
    };
    requestAnimationFrame(step);
  });
}

async function captureSnapshots(kind: 'warnings' | 'errors') {
  if (!renderer || !graphReady.value) return;
  capturing.value = true;
  try {
    const targets =
      kind === 'errors'
        ? nodes.value.filter((node) => node.status === 'broken')
        : nodes.value.filter((node) => node.status === 'unused');
    if (!targets.length) return;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    for (const node of targets) {
      await focusOnNode(node.id, kind === 'errors' ? 0.25 : 0.4, 550);
      await nextTick();
      markInteraction();
      const dataUrl = renderer.domElement.toDataURL('image/png');
      await uploadCodeGraphSnapshot({
        timestamp,
        node_id: node.id,
        image: dataUrl,
      });
    }
    await resetView();
  } finally {
    capturing.value = false;
  }
}

async function resetView() {
  if (!camera || !controls) return;
  markInteraction();
  await animateCamera(defaultCameraPos.clone(), defaultTarget.clone(), 700);
}

async function loadFileList() {
  try {
    const files = await fetchCodeGraphFiles();
    if (Array.isArray(files) && files.length) {
      fileNames.value = files;
    }
  } catch (error) {
    // ignore
  }
}

function startLoadingTicker() {
  if (!fileNames.value.length) {
    loadFileList();
  }
  if (loadingTimer) return;
  loadingTimer = window.setInterval(() => {
    if (!fileNames.value.length) {
      currentFileName.value = 'Scanning project…';
    } else {
      currentFileName.value = fileNames.value[fileIndex % fileNames.value.length];
      fileIndex += 1;
    }
    if (loadingPhase.value === 'loading') {
      loadingLabel.value = currentFileName.value || 'Scanning project…';
    }
  }, 25);
  startProgressLoop();
}

function stopLoadingTicker() {
  if (loadingTimer) {
    window.clearInterval(loadingTimer);
    loadingTimer = null;
  }
  currentFileName.value = '';
}

function setPhase(state: 'idle' | 'loading' | 'building' | 'done') {
  loadingPhase.value = state;
  phaseStart = performance.now();
  if (state === 'loading') {
    loadingProgress.value = 0;
    loadingLabel.value = currentFileName.value || 'Scanning project…';
    buildingLabel.value = 'Preparing build schematic…';
    stopBuildingLabelTicker();
  } else if (state === 'building') {
    buildingLabel.value = BUILDING_MESSAGES[buildingIndex % BUILDING_MESSAGES.length];
    startBuildingLabelTicker();
  } else if (state === 'done') {
    stopBuildingLabelTicker();
    buildingLabel.value = 'Graph up to date.';
    loadingProgress.value = 1;
    window.setTimeout(() => {
      loadingPhase.value = 'idle';
      loadingProgress.value = 0;
    }, 500);
    return;
  }
  startProgressLoop();
}

function startProgressLoop() {
  if (progressTimer) return;
  const loop = () => {
    updateProgress();
    progressTimer = window.requestAnimationFrame(loop);
  };
  progressTimer = window.requestAnimationFrame(loop);
}

function stopProgressLoop() {
  if (progressTimer) {
    window.cancelAnimationFrame(progressTimer);
    progressTimer = null;
  }
  if (loadingPhase.value === 'done' || loadingPhase.value === 'idle') {
    loadingProgress.value = loadingPhase.value === 'done' ? 1 : 0;
  }
}

function updateProgress() {
  if (loadingPhase.value === 'idle') return;
  const now = performance.now();
  const elapsed = now - phaseStart;
  if (loadingPhase.value === 'loading') {
    const pct = Math.min(elapsed / LOADING_EXPECTED_MS, 1);
    loadingProgress.value = Math.min(pct * 0.6, 0.6);
    loadingLabel.value = currentFileName.value || 'Scanning project…';
  } else if (loadingPhase.value === 'building') {
    const pct = Math.min(elapsed / BUILD_EXPECTED_MS, 1);
    loadingProgress.value = 0.6 + pct * 0.38;
  }
}

function startBuildingLabelTicker() {
  if (buildingLabelTimer) return;
  buildingLabel.value = BUILDING_MESSAGES[buildingIndex % BUILDING_MESSAGES.length];
  buildingIndex += 1;
  buildingLabelTimer = window.setInterval(() => {
    buildingLabel.value = BUILDING_MESSAGES[buildingIndex % BUILDING_MESSAGES.length];
    buildingIndex += 1;
  }, 600);
}

function stopBuildingLabelTicker() {
  if (buildingLabelTimer) {
    window.clearInterval(buildingLabelTimer);
    buildingLabelTimer = null;
  }
}

function schedulePoll() {
  if (!building.value) {
    if (pollTimer) {
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }
    return;
  }
  if (pollTimer) {
    return;
  }
  pollTimer = window.setTimeout(() => {
    pollTimer = null;
    refreshGraph();
  }, 2000);
}

onMounted(() => {
  initScene();
  loadFileList();
  refreshGraph();
});

watch(building, (value) => {
  if (value) {
    if (!loadingTimer) {
      startLoadingTicker();
    }
    setPhase('building');
  } else if (!loading.value) {
    setPhase('done');
  }
});

watch(loading, (value) => {
  if (!value && !building.value) {
    setPhase('done');
  }
});

onBeforeUnmount(() => {
  if (animationId) cancelAnimationFrame(animationId);
  stopLoadingTicker();
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
  stopProgressLoop();
  stopBuildingLabelTicker();
  if (renderer) {
    renderer.dispose();
    renderer = null;
  }
  nodeMeshes.clear();
  window.removeEventListener('resize', handleResize);
});
</script>

<style scoped>
.codegraph-view {
  display: flex;
  flex-direction: column;
  gap: 1.4rem;
}

.panel {
  background: rgba(7, 14, 25, 0.92);
  border: 1px solid rgba(66, 147, 255, 0.18);
  border-radius: 18px;
  padding: 1.2rem 1.5rem;
}

.toolbar {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
}

.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.8rem;
}

.loading-bar {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.loading-bar .bar {
  position: relative;
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.loading-bar .fill {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(71, 161, 255, 0.9), rgba(99, 255, 173, 0.9));
  transition: width 0.1s linear;
}

.loading-bar .file-name {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.75);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.loading-meta {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.loading-meta .build-name {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.75rem;
  color: rgba(99, 255, 173, 0.85);
}

.summary .label {
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.1rem;
  color: rgba(255, 255, 255, 0.6);
}

.summary .value {
  font-size: 1.3rem;
  font-weight: 600;
}

.summary .value.warn {
  color: #f59e0b;
}

.summary .value.error {
  color: #ef4444;
}

.canvas-panel {
  position: relative;
  min-height: 540px;
  padding: 0;
  overflow: hidden;
}

.canvas-host {
  position: relative;
  width: 100%;
  height: 100%;
}

.hint {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  color: rgba(255, 255, 255, 0.75);
  font-style: italic;
}

.btn.warning {
  background: rgba(250, 174, 57, 0.16);
  border: 1px solid rgba(250, 174, 57, 0.4);
}

.btn.danger {
  background: rgba(239, 83, 80, 0.16);
  border: 1px solid rgba(239, 83, 80, 0.4);
}

@media (max-width: 768px) {
  .canvas-panel {
    min-height: 360px;
  }
}
</style>
