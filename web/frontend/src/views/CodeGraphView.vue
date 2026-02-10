<template>
  <div class="codegraph-view">
    <section class="panel toolbar">
      <div class="controls">
        <button class="btn" type="button" @click="refreshGraph(true)" :disabled="loading || building">
          <span v-if="loading">{{ t('codegraph.loading') }}</span>
          <span v-else-if="building">{{ t('codegraph.building') }}</span>
          <span v-else>{{ t('codegraph.refresh') }}</span>
        </button>
        <button class="btn ghost" type="button" @click="resetView" :disabled="loading">
          {{ t('codegraph.reset_view') }}
        </button>
        <button class="btn warning" type="button" @click="captureSnapshots('warnings')" :disabled="capturing || !graphReady">
          {{ t('codegraph.snapshot_warnings') }}
        </button>
        <button class="btn danger" type="button" @click="captureSnapshots('errors')" :disabled="capturing || !graphReady">
          {{ t('codegraph.snapshot_errors') }}
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
        <div class="loading-files" v-if="recentFiles.length">
          <span v-for="file in recentFiles" :key="file">{{ file }}</span>
        </div>
      </div>
      <div class="summary">
        <div>
          <span class="label">{{ t('codegraph.files') }}</span>
          <span class="value">{{ summary.files || 0 }}</span>
        </div>
        <div>
          <span class="label">{{ t('codegraph.classes') }}</span>
          <span class="value">{{ summary.classes || 0 }}</span>
        </div>
        <div>
          <span class="label">{{ t('codegraph.functions') }}</span>
          <span class="value">{{ summary.functions || 0 }}</span>
        </div>
        <div>
          <span class="label">{{ t('codegraph.warnings') }}</span>
          <span class="value warn">{{ warnings.length }}</span>
        </div>
        <div>
          <span class="label">{{ t('codegraph.errors') }}</span>
          <span class="value error">{{ errors.length }}</span>
        </div>
      </div>
    </section>
    <section class="panel canvas-panel">
      <div ref="canvasContainer" class="canvas-host"></div>
      <div class="canvas-overlay">
        <div class="pad-wrapper">
          <div class="dpad-grid">
            <div class="pad-cell empty" />
            <button
              type="button"
              class="pad-button up"
              @mousedown="startPad('forward')"
              @mouseup="stopPad('forward')"
              @mouseleave="stopPad('forward')"
              @touchstart.prevent="startPad('forward')"
              @touchend.prevent="stopPad('forward')"
              @touchcancel.prevent="stopPad('forward')"
            >
              ▲
            </button>
            <div class="pad-cell empty" />
            <button
              type="button"
              class="pad-button left"
              @mousedown="startPad('left')"
              @mouseup="stopPad('left')"
              @mouseleave="stopPad('left')"
              @touchstart.prevent="startPad('left')"
              @touchend.prevent="stopPad('left')"
              @touchcancel.prevent="stopPad('left')"
            >
              ◄
            </button>
            <button type="button" class="pad-button center" @click="resetView">✣</button>
            <button
              type="button"
              class="pad-button right"
              @mousedown="startPad('right')"
              @mouseup="stopPad('right')"
              @mouseleave="stopPad('right')"
              @touchstart.prevent="startPad('right')"
              @touchend.prevent="stopPad('right')"
              @touchcancel.prevent="stopPad('right')"
            >
              ►
            </button>
            <div class="pad-cell empty" />
            <button
              type="button"
              class="pad-button down"
              @mousedown="startPad('back')"
              @mouseup="stopPad('back')"
              @mouseleave="stopPad('back')"
              @touchstart.prevent="startPad('back')"
              @touchend.prevent="stopPad('back')"
              @touchcancel.prevent="stopPad('back')"
            >
              ▼
            </button>
            <div class="pad-cell empty" />
          </div>
          <div class="zoom-pad">
            <button type="button" @click="nudgeZoom(-1)" @touchstart.prevent.stop="nudgeZoom(-1)">＋</button>
            <span>{{ zoomLabel }}</span>
            <button type="button" @click="nudgeZoom(1)" @touchstart.prevent.stop="nudgeZoom(1)">－</button>
          </div>
        </div>
      </div>
      <div v-if="graphError" class="hint hint-error">
        {{ graphError }}
      </div>
      <div v-else-if="!graphReady" class="hint">{{ t('codegraph.hint_ready') }}</div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, nextTick, watch, computed } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { fetchCodeGraph, fetchCodeGraphFiles, uploadCodeGraphSnapshot } from '@/api';
import { t } from '@/i18n';

type GraphNodePayload = {
  id: string;
  label: string;
  kind: string;
  file: string;
  status: string;
  line?: number;
  column?: number;
  meta?: Record<string, any>;
};

type GraphEdgePayload = {
  id: string;
  source: string;
  target: string;
  kind: string;
};

type FileLinkPayload = {
  source: string;
  target: string;
  imports: number;
  calls: number;
  weight: number;
};

declare global {
  interface WindowEventMap {
    'codegraph-ready': CustomEvent<{ nodes: number }>;
  }
}

const canvasContainer = ref<HTMLElement | null>(null);
const loading = ref(false);
const capturing = ref(false);
const graphReady = ref(false);
const graphError = ref<string | null>(null);
const summary = ref<Record<string, number>>({});
const warnings = ref<string[]>([]);
const errors = ref<string[]>([]);
const nodes = ref<GraphNodePayload[]>([]);
const graphEdges = ref<GraphEdgePayload[]>([]);
const graphLinks = ref<FileLinkPayload[]>([]);
const building = ref(false);
const buildingCompletionPending = ref(false);
const fileNames = ref<string[]>([]);
const currentFileName = ref('');
const loadingProgress = ref(0);
const loadingPhase = ref<'idle' | 'loading' | 'building' | 'done'>('idle');
const loadingLabel = ref(t('codegraph.idle'));
const buildingLabel = ref(t('codegraph.awaiting_plan'));
const recentFiles = ref<string[]>([]);

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.OrthographicCamera | null = null;
let controls: OrbitControls | null = null;
let animationId: number | null = null;
let lastInteraction = performance.now();
let defaultCameraPos = new THREE.Vector3(0, 2400, 1);
let defaultTarget = new THREE.Vector3(0, 0, 0);
let inactivityResetMs = 8000;
const GRAPH_READY_EVENT = 'codegraph-ready';
const nodeMeshes = new Map<string, THREE.Object3D>();
const labelTextureCache = new Map<string, THREE.Texture>();
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
let graphSettledAt = 0;
const LOADING_EXPECTED_MS = 9000;
const BUILD_EXPECTED_MS = 15000;
const movementState = {
  forward: false,
  back: false,
  left: false,
  right: false,
};
let lastFrameTime = performance.now();
const CAMERA_MIN_HEIGHT = 180;
const CAMERA_MAX_HEIGHT = 2800;
const CAMERA_SPEED = 450;
const CAMERA_LIMITS = { x: 2600, z: 1800 };
const zoomLevel = ref(1);
const zoomLabel = computed(() => `${zoomLevel.value.toFixed(1)}x`);
const ZOOM_STEP = 0.18;
const ZOOM_MIN = 0.12;
const ZOOM_MAX = 3.2;
const ZOOM_DURATION_MS = 260;
let zoomAnimation: number | null = null;
let zoomStartTime = 0;
let zoomFrom = 1;
let zoomTo = 1;
const FILE_NODE_WIDTH = 240;
const FILE_NODE_HEIGHT = 170;
const FILE_NODE_RADIUS = Math.sqrt((FILE_NODE_WIDTH / 2) ** 2 + (FILE_NODE_HEIGHT / 2) ** 2);
const MAX_LINKS_PER_FILE = 6;
const LABEL_CANVAS_WIDTH = 768;
const LABEL_CANVAS_HEIGHT = 256;
const MAX_LABEL_LINES = 3;
const BUILDING_MESSAGES = [
  t('codegraph.build_linking'),
  t('codegraph.build_resolving'),
  t('codegraph.build_indexing'),
  t('codegraph.build_scoring'),
  t('codegraph.build_drawing'),
  t('codegraph.build_assembling'),
];

async function refreshGraph(force = false) {
  loading.value = true;
  graphError.value = null;
  startLoadingTicker();
  setPhase('loading');
  try {
    const data = await fetchCodeGraph(force);
    building.value = Boolean(data?.building);
    summary.value = data?.summary || {};
    warnings.value = data?.warnings || [];
    errors.value = data?.errors || [];
    nodes.value = (data?.nodes || []) as GraphNodePayload[];
    graphEdges.value = (data?.edges || []) as GraphEdgePayload[];
    graphLinks.value = (data?.file_links || []) as FileLinkPayload[];
    graphReady.value = Boolean(nodes.value.length);
    if (graphReady.value) {
      try {
        buildScene(data);
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('[codegraph] failed to render scene', error);
        graphReady.value = false;
        graphError.value = t('codegraph.error_render');
      }
    } else {
      if (building.value) {
        graphError.value = t('codegraph.build_pending');
      } else {
        graphError.value = graphError.value || t('codegraph.no_artifacts');
      }
      await resetView();
    }
    schedulePoll();
  } catch (error: any) {
    const friendly = describeGraphError(error);
    loadingLabel.value = t('codegraph.awaiting_response');
    buildingLabel.value = friendly;
    graphError.value = friendly;
    schedulePoll();
  } finally {
    if (!building.value) {
      stopLoadingTicker();
    }
    loading.value = false;
  }
}

function initScene() {
  if (!canvasContainer.value) return;
  const container = canvasContainer.value;
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  const width = container.clientWidth || container.parentElement?.clientWidth || 800;
  const height = container.clientHeight || 540;
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050b16);

  camera = createTopCamera(width, height);
  camera.position.copy(defaultCameraPos);
  camera.up.set(0, 0, -1);
  camera.lookAt(defaultTarget);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableRotate = false;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.maxPolarAngle = 0;
  controls.minPolarAngle = 0;
  controls.addEventListener('change', () => markInteraction());
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('wheel', onCanvasWheel, { passive: false });

  const ambient = new THREE.AmbientLight(0xffffff, 0.9);
  scene.add(ambient);
  const dir = new THREE.DirectionalLight(0xffffff, 0.45);
  dir.position.set(500, 1000, -500);
  scene.add(dir);

  clampCameraState();
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

function buildScene(payload: { nodes: GraphNodePayload[]; edges: GraphEdgePayload[]; file_links?: FileLinkPayload[] }) {
  if (!scene) return;
  clearScene();
  addBasePlane();

  const fileNodes = payload.nodes.filter((node) => node.kind === 'file');
  if (!fileNodes.length) {
    graphReady.value = false;
    return;
  }

  const layout = computeGridLayout(fileNodes, payload.file_links || []);
  const childrenByFile = groupChildren(payload.nodes);
  const ioStats = computeIoStats(payload.edges || []);

  fileNodes.forEach((file) => {
    const entry = layout[file.id] || { position: new THREE.Vector3(0, 2, 0), grid: { row: 0, col: 0 } };
    const mesh = createFileMesh(file, entry.position, ioStats);
    scene!.add(mesh);
    nodeMeshes.set(file.id, mesh);
    placeChildren(file, childrenByFile.get(file.file) || [], entry.position, ioStats);
  });

  drawFileRelationshipEdges(payload.file_links || [], layout);
  graphReady.value = true;
  announceGraphReady();
}

function addBasePlane() {
  if (!scene) return;
  const planeGeometry = new THREE.PlaneGeometry(5200, 3200);
  const planeMaterial = new THREE.MeshBasicMaterial({ color: 0x040913, side: THREE.DoubleSide });
  const plane = new THREE.Mesh(planeGeometry, planeMaterial);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = -1;
  scene.add(plane);

  const gridHelper = new THREE.GridHelper(5200, 80, 0x123456, 0x0e223f);
  scene.add(gridHelper);
}

function computeGridLayout(files: GraphNodePayload[], links: FileLinkPayload[]) {
  const layout: Record<string, { position: THREE.Vector3; grid: { row: number; col: number } }> = {};
  if (!files.length) {
    return layout;
  }
  const order = orderFilesForGrid(files, links);
  const columns = Math.max(1, Math.ceil(Math.sqrt(order.length) * 1.25));
  const rows = Math.max(1, Math.ceil(order.length / columns));
  const spacingX = 360;
  const spacingZ = 260;
  order.forEach((file, index) => {
    const row = Math.floor(index / columns);
    const col = index % columns;
    const centeredCol = col - (columns - 1) / 2;
    const centeredRow = row - (rows - 1) / 2;
    const position = new THREE.Vector3(centeredCol * spacingX, 3, centeredRow * spacingZ);
    layout[file.id] = { position, grid: { row, col } };
  });
  return layout;
}

function orderFilesForGrid(files: GraphNodePayload[], links: FileLinkPayload[]) {
  const degree = new Map<string, number>();
  links.forEach((link) => {
    degree.set(link.source, (degree.get(link.source) || 0) + (link.weight || 1));
    degree.set(link.target, (degree.get(link.target) || 0) + (link.weight || 1));
  });
  return [...files].sort((a, b) => {
    const moduleA = a.file.split('/')[0] || '';
    const moduleB = b.file.split('/')[0] || '';
    if (moduleA !== moduleB) {
      return moduleA.localeCompare(moduleB);
    }
    const degA = degree.get(a.id) || 0;
    const degB = degree.get(b.id) || 0;
    if (degA !== degB) {
      return degB - degA;
    }
    return a.file.localeCompare(b.file);
  });
}

function groupChildren(nodes: GraphNodePayload[]) {
  const map = new Map<string, GraphNodePayload[]>();
  nodes.forEach((node) => {
    if (node.kind === 'file') return;
    if (!map.has(node.file)) {
      map.set(node.file, []);
    }
    map.get(node.file)!.push(node);
  });
  return map;
}

type IoStats = { inbound: number; outbound: number };

function computeIoStats(edges: GraphEdgePayload[]) {
  const stats = new Map<string, IoStats>();
  edges.forEach((edge) => {
    if (edge.kind !== 'calls') return;
    if (!stats.has(edge.source)) stats.set(edge.source, { inbound: 0, outbound: 0 });
    if (!stats.has(edge.target)) stats.set(edge.target, { inbound: 0, outbound: 0 });
    stats.get(edge.source)!.outbound += 1;
    stats.get(edge.target)!.inbound += 1;
  });
  return stats;
}

function getIoStats(map: Map<string, IoStats>, nodeId: string): IoStats {
  return map.get(nodeId) || { inbound: 0, outbound: 0 };
}

function createFileMesh(node: GraphNodePayload, position: THREE.Vector3, ioStats: Map<string, IoStats>) {
  const width = FILE_NODE_WIDTH;
  const height = FILE_NODE_HEIGHT;
  const shape = roundedRectShape(width, height, 28);
  const geometry = extrudeShape(shape, 10);
  const color = STATUS_COLORS[node.status] || STATUS_COLORS.ok;
  const material = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.35,
    roughness: 0.55,
    transparent: true,
    opacity: 0.92,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.copy(position.clone());
  mesh.position.y = 6;
  mesh.userData = { node };
  const stats = aggregateFileStats(node);
  const label = createLabelPlane(
    node.label,
    t('codegraph.label_classes')
      .replace('{classes}', String(stats.classes))
      .replace('{functions}', String(stats.functions)),
    width - 30
  );
  mesh.add(label);
  const dots = createIoDots(getIoStats(ioStats, node.id), width - 40);
  mesh.add(dots);
  return mesh;
}

function aggregateFileStats(node: GraphNodePayload) {
  const definitions: string[] = (node.meta?.definitions || []) as string[];
  const classCount = definitions.filter((label) => /^[A-Z]/.test(label || '')).length;
  const functionCount = Math.max(definitions.length - classCount, 0);
  return { classes: classCount, functions: functionCount };
}

function placeChildren(
  fileNode: GraphNodePayload,
  children: GraphNodePayload[],
  basePosition: THREE.Vector3,
  ioStats: Map<string, IoStats>
) {
  if (!children.length || !scene) return;
  const sorted = [...children].sort((a, b) => {
    if (a.kind === b.kind) {
      return a.label.localeCompare(b.label);
    }
    const order = ['class', 'function', 'method'];
    return order.indexOf(a.kind) - order.indexOf(b.kind);
  });
  const perRing = 12;
  const radiusStep = 70;
  sorted.forEach((child, index) => {
    const ring = Math.floor(index / perRing);
    const angle = ((index % perRing) / perRing) * Math.PI * 2;
    const radius = 110 + ring * radiusStep;
    const x = basePosition.x + Math.cos(angle) * radius;
    const z = basePosition.z + Math.sin(angle) * radius;
    const mesh = createFlowNodeMesh(child, new THREE.Vector3(x, 4.5, z), ioStats);
    scene!.add(mesh);
    nodeMeshes.set(child.id, mesh);
  });
}

function createFlowNodeMesh(node: GraphNodePayload, position: THREE.Vector3, ioStats: Map<string, IoStats>) {
  const stats = getIoStats(ioStats, node.id);
  const color = STATUS_COLORS[node.status] || STATUS_COLORS.ok;
  const { geometry, labelWidth } = geometryForNode(node);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.55,
    metalness: 0.25,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.copy(position);
  mesh.userData = { node };
  const subtitle = t('codegraph.io_stats')
    .replace('{in}', String(stats.inbound))
    .replace('{out}', String(stats.outbound));
  const label = createLabelPlane(node.label, subtitle, labelWidth);
  mesh.add(label);
  const dots = createIoDots(stats, labelWidth - 12);
  mesh.add(dots);
  return mesh;
}

function geometryForNode(node: GraphNodePayload) {
  switch (node.kind) {
    case 'class':
      return { geometry: extrudeShape(rectShape(120, 80), 6), labelWidth: 90 };
    case 'method':
      return { geometry: extrudeShape(diamondShape(90, 90), 5), labelWidth: 80 };
    default:
      return { geometry: extrudeShape(parallelogramShape(100, 70, 18), 5), labelWidth: 84 };
  }
}

function roundedRectShape(width: number, height: number, radius: number) {
  const shape = new THREE.Shape();
  const hw = width / 2;
  const hh = height / 2;
  const r = Math.min(radius, hw, hh);
  shape.moveTo(-hw + r, -hh);
  shape.lineTo(hw - r, -hh);
  shape.quadraticCurveTo(hw, -hh, hw, -hh + r);
  shape.lineTo(hw, hh - r);
  shape.quadraticCurveTo(hw, hh, hw - r, hh);
  shape.lineTo(-hw + r, hh);
  shape.quadraticCurveTo(-hw, hh, -hw, hh - r);
  shape.lineTo(-hw, -hh + r);
  shape.quadraticCurveTo(-hw, -hh, -hw + r, -hh);
  return shape;
}

function rectShape(width: number, height: number) {
  const shape = new THREE.Shape();
  const hw = width / 2;
  const hh = height / 2;
  shape.moveTo(-hw, -hh);
  shape.lineTo(hw, -hh);
  shape.lineTo(hw, hh);
  shape.lineTo(-hw, hh);
  shape.lineTo(-hw, -hh);
  return shape;
}

function parallelogramShape(width: number, height: number, skew: number) {
  const shape = new THREE.Shape();
  const hw = width / 2;
  const hh = height / 2;
  shape.moveTo(-hw + skew, -hh);
  shape.lineTo(hw + skew, -hh);
  shape.lineTo(hw - skew, hh);
  shape.lineTo(-hw - skew, hh);
  shape.lineTo(-hw + skew, -hh);
  return shape;
}

function diamondShape(width: number, height: number) {
  const shape = new THREE.Shape();
  const hw = width / 2;
  const hh = height / 2;
  shape.moveTo(0, hh);
  shape.lineTo(hw, 0);
  shape.lineTo(0, -hh);
  shape.lineTo(-hw, 0);
  shape.lineTo(0, hh);
  return shape;
}

function extrudeShape(shape: THREE.Shape, depth: number) {
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelSize: 2,
    bevelThickness: 1.2,
    bevelSegments: 1,
    curveSegments: 10,
  });
  geometry.center();
  return geometry;
}

function createLabelPlane(text: string, subtitle: string, width: number) {
  const texture = getLabelTexture(text, subtitle);
  const texImage = texture.image as HTMLCanvasElement | HTMLImageElement;
  const aspect = texImage && texImage.height ? texImage.width / texImage.height : 2;
  const planeWidth = width;
  const planeHeight = planeWidth / aspect;
  const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const plane = new THREE.Mesh(geometry, material);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = 4.2;
  return plane;
}

function getLabelTexture(text: string, subtitle: string) {
  const key = `${text}|${subtitle}`;
  if (labelTextureCache.has(key)) {
    return labelTextureCache.get(key)!;
  }
  const canvas = document.createElement('canvas');
  canvas.width = LABEL_CANVAS_WIDTH;
  canvas.height = LABEL_CANVAS_HEIGHT;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas 2D unavailable');
  }
  drawLabelBackdrop(ctx, canvas.width, canvas.height);
  ctx.textBaseline = 'top';
  const textLines = wrapLabelText(ctx, text, canvas.width - 160, MAX_LABEL_LINES);
  ctx.fillStyle = '#e9f2ff';
  ctx.font = '600 58px "Inter", "Segoe UI", sans-serif';
  textLines.forEach((line, index) => {
    ctx.fillText(line, 60, 36 + index * 64);
  });
  ctx.fillStyle = '#9bd2ff';
  ctx.font = '500 40px "Inter", "Segoe UI", sans-serif';
  ctx.fillText(subtitle, 60, canvas.height - 86);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  labelTextureCache.set(key, texture);
  return texture;
}

function drawLabelBackdrop(ctx: CanvasRenderingContext2D, width: number, height: number) {
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, 'rgba(12, 26, 60, 0.95)');
  gradient.addColorStop(1, 'rgba(7, 16, 32, 0.92)');
  ctx.fillStyle = gradient;
  const radius = 40;
  ctx.beginPath();
  ctx.moveTo(radius, 0);
  ctx.lineTo(width - radius, 0);
  ctx.quadraticCurveTo(width, 0, width, radius);
  ctx.lineTo(width, height - radius);
  ctx.quadraticCurveTo(width, height, width - radius, height);
  ctx.lineTo(radius, height);
  ctx.quadraticCurveTo(0, height, 0, height - radius);
  ctx.lineTo(0, radius);
  ctx.quadraticCurveTo(0, 0, radius, 0);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = 'rgba(122, 181, 255, 0.45)';
  ctx.lineWidth = 4;
  ctx.stroke();
}

function wrapLabelText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, maxLines: number) {
  const sanitized = text.replace(/\\/g, '/');
  const tokens = sanitized.split(/(?=[/._-])/);
  const lines: string[] = [];
  let current = '';
  const pushLine = (line: string) => {
    if (lines.length < maxLines) {
      lines.push(line.trim());
    }
  };
  tokens.forEach((token) => {
    const candidate = current ? `${current}${token}` : token;
    if (ctx.measureText(candidate).width <= maxWidth) {
      current = candidate;
      return;
    }
    if (current) {
      pushLine(current);
      current = token.trimStart();
    } else {
      current = token;
    }
  });
  if (current && lines.length < maxLines) {
    pushLine(current);
  }
  if (lines.length > maxLines) {
    return lines.slice(0, maxLines - 1).concat(`${lines[maxLines - 1]}…`);
  }
  if (lines.length === maxLines && tokens.length > lines.join('').length) {
    const last = lines[maxLines - 1];
    let trimmed = last;
    while (ctx.measureText(`${trimmed}…`).width > maxWidth && trimmed.length > 1) {
      trimmed = trimmed.slice(0, -1);
    }
    lines[maxLines - 1] = `${trimmed}…`;
  }
  return lines;
}

function createIoDots(stats: IoStats, width: number) {
  const group = new THREE.Group();
  const radius = 6;
  if (stats.inbound > 0) {
    const dot = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 24),
      new THREE.MeshBasicMaterial({ color: 0x22c55e })
    );
    dot.rotation.x = -Math.PI / 2;
    dot.position.set(-width / 2, 4.6, -radius * 0.6);
    group.add(dot);
  }
  if (stats.outbound > 0) {
    const dot = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 24),
      new THREE.MeshBasicMaterial({ color: 0xf97316 })
    );
    dot.rotation.x = -Math.PI / 2;
    dot.position.set(width / 2, 4.6, -radius * 0.6);
    group.add(dot);
  }
  return group;
}

function drawFileRelationshipEdges(
  links: FileLinkPayload[],
  layout: Record<string, { position: THREE.Vector3; grid: { row: number; col: number } }>
) {
  if (!scene || !links.length) return;
  const limited: FileLinkPayload[] = [];
  const bySource = new Map<string, FileLinkPayload[]>();
  links.forEach((link) => {
    const sourcePlacement = layout[link.source];
    const targetPlacement = layout[link.target];
    if (!sourcePlacement || !targetPlacement) return;
    if (!bySource.has(link.source)) {
      bySource.set(link.source, []);
    }
    bySource.get(link.source)!.push(link);
  });
  bySource.forEach((entries) => {
    entries.sort((a, b) => (b.weight || 0) - (a.weight || 0));
    limited.push(...entries.slice(0, MAX_LINKS_PER_FILE));
  });
  limited.forEach((link, index) => {
    const sourcePlacement = layout[link.source];
    const targetPlacement = layout[link.target];
    if (!sourcePlacement || !targetPlacement) return;
    const direction = targetPlacement.position.clone().sub(sourcePlacement.position);
    const distance = direction.length();
    if (distance < 1) return;
    const start = computeAnchorPoint(sourcePlacement.position, targetPlacement.position);
    const end = computeAnchorPoint(targetPlacement.position, sourcePlacement.position);
    const perp = new THREE.Vector3(-direction.z, 0, direction.x);
    if (perp.lengthSq() === 0) {
      perp.set(0, 0, 1);
    }
    perp.normalize();
    const bendDistance = Math.min(distance * 0.25, 240);
    const alternator = index % 2 === 0 ? 1 : -1;
    perp.multiplyScalar(bendDistance * alternator);
    const controlOffset = perp.clone().multiplyScalar(0.6);
    const control1 = start.clone().add(controlOffset);
    const control2 = end.clone().sub(controlOffset);
    [start, end, control1, control2].forEach((point) => {
      point.y = 1.2;
    });
    const curve = new THREE.CubicBezierCurve3(start, control1, control2, end);
    const points = curve.getPoints(40);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const opacity = THREE.MathUtils.clamp(0.18 + (link.weight || 0) / 90, 0.18, 0.78);
    const material = new THREE.LineBasicMaterial({
      color: 0x6f8bff,
      transparent: true,
      opacity,
    });
    const line = new THREE.Line(geometry, material);
    scene!.add(line);
  });
}

function computeAnchorPoint(origin: THREE.Vector3, towards: THREE.Vector3) {
  const dir = towards.clone().sub(origin);
  if (dir.lengthSq() === 0) {
    dir.set(1, 0, 0);
  }
  dir.normalize();
  const offset = FILE_NODE_RADIUS * 0.8 + 28;
  const anchor = origin.clone().add(dir.multiplyScalar(offset));
  anchor.y = 1.2;
  return anchor;
}

function handleResize() {
  if (!renderer || !camera || !canvasContainer.value) return;
  const container = canvasContainer.value;
  const width = container.clientWidth || container.parentElement?.clientWidth || 800;
  const height = container.clientHeight || 540;
  const nextCamera = createTopCamera(width, height);
  nextCamera.position.copy(camera.position);
  nextCamera.lookAt(controls?.target || defaultTarget);
  camera = nextCamera;
  controls?.update();
  renderer.setSize(width, height);
}

function animate() {
  if (!renderer || !scene || !camera) return;
  animationId = requestAnimationFrame(animate);
  const now = performance.now();
  const delta = (now - lastFrameTime) / 1000;
  lastFrameTime = now;
  applyMovement(delta);
  renderer.render(scene, camera);
  if (performance.now() - lastInteraction > inactivityResetMs) {
    lastInteraction = performance.now();
    resetView();
  }
}

function markInteraction() {
  lastInteraction = performance.now();
}

function handleKeyDown(event: KeyboardEvent) {
  switch (event.key.toLowerCase()) {
    case 'w':
    case 'arrowup':
      movementState.forward = true;
      break;
    case 's':
    case 'arrowdown':
      movementState.back = true;
      break;
    case 'a':
    case 'arrowleft':
      movementState.left = true;
      break;
    case 'd':
    case 'arrowright':
      movementState.right = true;
      break;
    default:
      return;
  }
  event.preventDefault();
  markInteraction();
}

function handleKeyUp(event: KeyboardEvent) {
  switch (event.key.toLowerCase()) {
    case 'w':
    case 'arrowup':
      movementState.forward = false;
      break;
    case 's':
    case 'arrowdown':
      movementState.back = false;
      break;
    case 'a':
    case 'arrowleft':
      movementState.left = false;
      break;
    case 'd':
    case 'arrowright':
      movementState.right = false;
      break;
    default:
      return;
  }
  event.preventDefault();
  markInteraction();
}

function onCanvasWheel(event: WheelEvent) {
  if (!camera || !controls) return;
  event.preventDefault();
  const delta = event.deltaY;
  nudgeZoom(delta > 0 ? 1 : -1);
  markInteraction();
}

function startPad(direction: 'forward' | 'back' | 'left' | 'right') {
  movementState[direction] = true;
  markInteraction();
}

function stopPad(direction: 'forward' | 'back' | 'left' | 'right') {
  movementState[direction] = false;
}

function nudgeZoom(direction: number) {
  const next = THREE.MathUtils.clamp(zoomLevel.value + direction * ZOOM_STEP, ZOOM_MIN, ZOOM_MAX);
  animateZoomTo(next);
}

function applyMovement(delta: number) {
  if (!camera || !controls) return;
  const dx =
    (movementState.right ? 1 : 0) -
    (movementState.left ? 1 : 0);
  const dz =
    (movementState.back ? 1 : 0) -
    (movementState.forward ? 1 : 0);
  if (!dx && !dz) return;
  const speed = CAMERA_SPEED * delta;
  const move = new THREE.Vector3(dx, 0, dz).normalize().multiplyScalar(speed);
  camera.position.add(move);
  controls.target.add(move);
  clampCameraState();
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
    const node = resolveNodeFromObject(hit);
    if (node) {
      focusOnNode(node.id);
    }
  }
}

async function focusOnNode(nodeId: string, zoomFactor = 0.35, duration = 700) {
  const mesh = nodeMeshes.get(nodeId);
  if (!mesh || !camera || !controls) return;
  const target = mesh.position.clone();
  const zoomTarget = THREE.MathUtils.clamp(1.9 - zoomFactor * 2.4, ZOOM_MIN, 2.1);
  animateZoomTo(zoomTarget);
  const offsetHeight = 220 + 520 * zoomFactor;
  const offset = new THREE.Vector3(0, offsetHeight, 0);
  const direction = target.clone().sub(new THREE.Vector3(0, 0, 0));
  direction.y = 0;
  if (direction.lengthSq() === 0) {
    direction.set(1, 0, 0);
  }
  direction.normalize().multiplyScalar(120 + 240 * zoomFactor);
  const destination = target.clone().add(offset).add(direction);
  const boundedTarget = clampVectorToBounds(target.clone(), false);
  boundedTarget.y = 0;
  await animateCamera(clampVectorToBounds(destination, true), boundedTarget, duration);
}

async function animateCamera(position: THREE.Vector3, target: THREE.Vector3, duration = 600) {
  if (!camera || !controls) return;
  const boundedPosition = clampVectorToBounds(position.clone(), true);
  const boundedTarget = clampVectorToBounds(target.clone(), false);
  boundedTarget.y = 0;
  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const start = performance.now();
  return new Promise<void>((resolve) => {
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = t * (2 - t);
      camera!.position.lerpVectors(startPos, boundedPosition, eased);
      controls!.target.lerpVectors(startTarget, boundedTarget, eased);
      clampCameraState();
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        clampCameraState();
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
    await waitForGraphSettled();
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
  clampCameraState();
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
      currentFileName.value = t('codegraph.scanning');
    } else {
      currentFileName.value = fileNames.value[fileIndex % fileNames.value.length];
      fileIndex += 1;
    }
    if (loadingPhase.value === 'loading') {
      loadingLabel.value = currentFileName.value
        ? t('codegraph.loading_file').replace('{file}', currentFileName.value)
        : t('codegraph.scanning');
      pushRecentFile(currentFileName.value);
    }
  }, 25);
  startProgressLoop();
}

function stopLoadingTicker() {
  if (building.value) {
    return;
  }
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
    loadingLabel.value = currentFileName.value || t('codegraph.scanning');
    buildingLabel.value = t('codegraph.preparing_build');
    recentFiles.value = [];
    stopBuildingLabelTicker();
  } else if (state === 'building') {
    buildingLabel.value = BUILDING_MESSAGES[buildingIndex % BUILDING_MESSAGES.length];
    startBuildingLabelTicker();
  } else if (state === 'done') {
    stopBuildingLabelTicker();
    buildingLabel.value = t('codegraph.up_to_date');
    loadingLabel.value = t('codegraph.idle');
    currentFileName.value = '';
    loadingProgress.value = 1;
    stopProgressLoop();
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
    loadingLabel.value = currentFileName.value || t('codegraph.scanning');
  } else if (loadingPhase.value === 'building') {
    const pct = Math.min(elapsed / BUILD_EXPECTED_MS, 1);
    loadingProgress.value = 0.6 + pct * 0.38;
  }
}

function pushRecentFile(name: string) {
  const cleaned = (name || '').trim();
  if (!cleaned) return;
  if (recentFiles.value[0] === cleaned) return;
  recentFiles.value = [cleaned, ...recentFiles.value.filter((entry) => entry !== cleaned)].slice(0, 6);
}

function createTopCamera(width: number, height: number) {
  const frustumSize = 2000;
  const aspect = width / height;
  const camera = new THREE.OrthographicCamera(
    (-frustumSize * aspect) / 2,
    (frustumSize * aspect) / 2,
    frustumSize / 2,
    -frustumSize / 2,
    10,
    6000,
  );
  camera.zoom = zoomLevel.value;
  camera.updateProjectionMatrix();
  return camera;
}

function animateZoomTo(target: number) {
  if (!camera) return;
  if (Math.abs(target - zoomLevel.value) < 0.001) return;
  zoomFrom = zoomLevel.value;
  zoomTo = target;
  zoomStartTime = performance.now();
  if (zoomAnimation) {
    cancelAnimationFrame(zoomAnimation);
  }
  const tick = () => {
    if (!camera) return;
    const elapsed = performance.now() - zoomStartTime;
    const t = Math.min(1, elapsed / ZOOM_DURATION_MS);
    const eased = t * (2 - t);
    const value = THREE.MathUtils.lerp(zoomFrom, zoomTo, eased);
    setCameraZoom(value);
    if (t < 1) {
      zoomAnimation = requestAnimationFrame(tick);
    } else {
      zoomAnimation = null;
    }
  };
  zoomAnimation = requestAnimationFrame(tick);
}

function setCameraZoom(value: number) {
  if (!camera) return;
  zoomLevel.value = value;
  camera.zoom = value;
  camera.updateProjectionMatrix();
}

function clampVectorToBounds(vector: THREE.Vector3, clampY = true) {
  vector.x = THREE.MathUtils.clamp(vector.x, -CAMERA_LIMITS.x, CAMERA_LIMITS.x);
  vector.z = THREE.MathUtils.clamp(vector.z, -CAMERA_LIMITS.z, CAMERA_LIMITS.z);
  if (clampY) {
    vector.y = THREE.MathUtils.clamp(vector.y, CAMERA_MIN_HEIGHT, CAMERA_MAX_HEIGHT);
  }
  return vector;
}

function clampCameraState() {
  if (!camera || !controls) return;
  clampVectorToBounds(camera.position, true);
  const target = controls.target.clone();
  clampVectorToBounds(target, false);
  target.y = 0;
  controls.target.copy(target);
  controls.update();
}

function announceGraphReady() {
  graphSettledAt = performance.now();
  const event = new CustomEvent(GRAPH_READY_EVENT, { detail: { nodes: nodes.value.length } });
  window.dispatchEvent(event);
}

async function waitForGraphSettled() {
  if (!graphReady.value) {
    await new Promise<void>((resolve) => {
      const handler = () => resolve();
      window.addEventListener(GRAPH_READY_EVENT, handler as EventListener, { once: true });
    });
  }
  const elapsed = performance.now() - graphSettledAt;
  const delayMs = Math.max(0, 140 - elapsed);
  if (delayMs > 0) {
    await wait(delayMs);
  }
  await nextFrame();
  await nextFrame();
}

function wait(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}

function nextFrame() {
  return new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

function describeGraphError(error: any): string {
  if (error?.code === 'ECONNABORTED') {
    const timeout = error?.config?.timeout ? `${error.config.timeout}ms` : 'the request';
    return t('codegraph.timeout').replace('{timeout}', timeout);
  }
  if (error?.response) {
    const status = error.response.status;
    if (status === 403) {
      return t('codegraph.error_session');
    }
    if (status === 404) {
      return t('codegraph.error_404');
    }
    return t('codegraph.error_status').replace('{status}', String(status));
  }
  if (error?.request) {
    return t('codegraph.error_unreachable');
  }
  const message = typeof error?.message === 'string' ? error.message : t('codegraph.error_unknown');
  return t('codegraph.error_request').replace('{message}', message);
}

function resolveNodeFromObject(object: THREE.Object3D | null): GraphNodePayload | null {
  let current: THREE.Object3D | null = object;
  while (current) {
    if (current.userData?.node) {
      return current.userData.node as GraphNodePayload;
    }
    current = (current.parent as THREE.Object3D) || null;
  }
  return null;
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
    if (!loading.value) {
      setPhase('done');
      stopLoadingTicker();
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
  window.addEventListener('keydown', handleKeyDown);
  window.addEventListener('keyup', handleKeyUp);
});

watch(building, async (value, previous) => {
  if (value) {
    buildingCompletionPending.value = true;
    if (!loadingTimer) {
      startLoadingTicker();
    }
    setPhase('building');
  } else {
    if (buildingCompletionPending.value && previous) {
      buildingCompletionPending.value = false;
      setPhase('done');
      stopLoadingTicker();
      await refreshGraph(false);
    } else if (!loading.value) {
      setPhase('done');
      stopLoadingTicker();
    }
  }
});

watch(loading, (value) => {
  if (!value && !building.value) {
    setPhase('done');
    stopLoadingTicker();
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
  window.removeEventListener('keydown', handleKeyDown);
  window.removeEventListener('keyup', handleKeyUp);
  if (renderer) {
    renderer.dispose();
    renderer = null;
  }
  if (zoomAnimation) {
    cancelAnimationFrame(zoomAnimation);
    zoomAnimation = null;
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
  width: 100%;
  min-height: 0;
  flex: 1 1 auto;
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

.loading-files {
  margin-top: 0.35rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.72rem;
  color: rgba(255, 255, 255, 0.6);
}

.loading-files span {
  background: rgba(255, 255, 255, 0.06);
  padding: 0.1rem 0.4rem;
  border-radius: 6px;
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
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.canvas-host {
  position: relative;
  width: 100%;
  min-height: 540px;
  flex: 1 1 auto;
  height: clamp(480px, 60vh, 1000px);
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

.hint-error {
  color: #f87171;
  font-style: normal;
  font-weight: 600;
}

.canvas-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  justify-content: flex-end;
  align-items: flex-end;
  pointer-events: none;
  padding: 1rem;
}

.pad-wrapper {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.6rem;
  pointer-events: auto;
}

.dpad-grid {
  display: grid;
  grid-template-areas:
    "cell1 cell2 cell3"
    "cell4 cell5 cell6"
    "cell7 cell8 cell9";
  gap: 0.25rem;
  padding: 0.45rem;
  border-radius: 12px;
  background: rgba(9, 16, 33, 0.82);
  border: 1px solid rgba(124, 182, 255, 0.45);
  box-shadow: 0 18px 38px rgba(0, 0, 0, 0.45);
}

.dpad-grid .pad-cell,
.dpad-grid .pad-button {
  width: 46px;
  height: 46px;
}

.pad-cell.empty {
  border-radius: 10px;
  background: rgba(14, 24, 44, 0.65);
  pointer-events: none;
}

.pad-button {
  border-radius: 10px;
  border: 1px solid rgba(158, 196, 255, 0.4);
  background: linear-gradient(145deg, rgba(32, 58, 104, 0.95), rgba(20, 37, 68, 0.95));
  color: #eff6ff;
  font-size: 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease;
}

.pad-button:hover {
  transform: translateY(-1px);
  border-color: rgba(148, 214, 255, 0.9);
  background: rgba(52, 98, 170, 0.95);
}

.pad-button.up {
  grid-area: cell2;
}

.pad-button.down {
  grid-area: cell8;
}

.pad-button.left {
  grid-area: cell4;
}

.pad-button.right {
  grid-area: cell6;
}

.pad-button.center {
  grid-area: cell5;
  background: rgba(74, 128, 210, 0.95);
  border-color: rgba(168, 216, 255, 0.85);
}

.zoom-pad {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  border-radius: 10px;
  background: rgba(6, 12, 24, 0.85);
  border: 1px solid rgba(122, 181, 255, 0.35);
  padding: 0.25rem 0.5rem;
  box-shadow: 0 12px 26px rgba(0, 0, 0, 0.4);
}

.zoom-pad button {
  width: 36px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid rgba(158, 196, 255, 0.35);
  background: rgba(26, 46, 86, 0.95);
  color: #f8fbff;
  cursor: pointer;
}

.zoom-pad span {
  font-size: 0.9rem;
  color: rgba(230, 243, 255, 0.9);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
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
