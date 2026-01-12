<template>
  <div ref="canvasContainer" class="organism-canvas"></div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface GraphNode {
  id: string;
  label?: string;
  group?: string;
  status?: string;
  exposure?: number;
  value?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  weight?: number;
  kind?: string;
}

const props = defineProps<{
  graph?: {
    nodes?: GraphNode[];
    edges?: GraphEdge[];
  };
  labelScale?: number;
}>();

const canvasContainer = ref<HTMLDivElement | null>(null);

let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let renderer: THREE.WebGLRenderer | null = null;
let graphGroup: THREE.Group | null = null;
let frameHandle: number | null = null;
let controls: OrbitControls | null = null;

let resetTimeout: number | null = null;
let isZooming = false;
let zoomStartY = 0;
let zoomTarget = 0;
let currentZoom = 0;
let lastInteraction = 0;

const theme = {
  background: 0x050b12,
  nodeColors: new Map<string, number>([
    ['ok', 0x38bdf8],
    ['idle', 0x64748b],
    ['soft', 0x60a5fa],
    ['engaged', 0x22d3ee],
    ['strong', 0x34d399],
    ['cautious', 0xfacc15],
    ['halted', 0xef4444],
    ['long', 0x22d3ee],
    ['short', 0xf97316],
    ['watch', 0x818cf8],
    ['position', 0x22c55e],
    ['event', 0xf472b6],
    ['busy', 0x38bdf8],
    ['graph', 0x7c3aed],
  ]),
  edgeColors: new Map<string, number>([
    ['signal', 0x1e293b],
    ['operational', 0x1d4ed8],
    ['exposure', 0x0ea5e9],
    ['position', 0x22c55e],
    ['decision', 0xf97316],
    ['neuro', 0x6366f1],
    ['queue', 0x2563eb],
    ['ghost', 0x7c3aed],
    ['state', 0x14b8a6],
    ['session', 0x0f766e],
    ['holding', 0x0ea5e9],
    ['finance', 0xfacc15],
    ['trade', 0x38bdf8],
    ['feedback', 0xf59e0b],
    ['metric', 0x8b5cf6],
    ['discovery', 0x22d3ee],
    ['vote', 0xfbbf24],
    ['memory', 0x10b981],
  ]),
};

const groupPalette = new Map<string, number>([
  ['system', 0x38bdf8],
  ['module', 0x7c3aed],
  ['asset', 0x22c55e],
  ['event', 0xf472b6],
  ['session', 0x2dd4bf],
  ['holding', 0xf97316],
  ['native', 0x0ea5e9],
  ['finance', 0xfacc15],
  ['ghost', 0xdb2777],
  ['queue', 0x2563eb],
  ['feedback', 0xf59e0b],
  ['metric', 0x8b5cf6],
  ['discovery', 0x22d3ee],
  ['vote', 0xfbbf24],
  ['transition', 0xa855f7],
  ['brain', 0x38bdf8],
]);

const crossGroupColor = 0xfcd34d;

function blendColors(a: number, b: number, ratio = 0.5) {
  const colorA = new THREE.Color(a);
  colorA.lerp(new THREE.Color(b), ratio);
  return colorA.getHex();
}

function getGroupColor(group?: string) {
  if (!group) return groupPalette.get('system') ?? 0x38bdf8;
  return groupPalette.get(group) ?? groupPalette.get('asset') ?? 0x22c55e;
}

function resolveNodeColor(node: GraphNode): number {
  const base = new THREE.Color(getGroupColor(node.group));
  const status = (node.status || 'ok').toLowerCase();
  if (['halted', 'cautious', 'warn', 'error'].includes(status)) {
    base.lerp(new THREE.Color(0xef4444), 0.35);
  } else if (['strong', 'engaged'].includes(status)) {
    base.lerp(new THREE.Color(0x4ade80), 0.2);
  } else if (status === 'soft') {
    base.lerp(new THREE.Color(0x60a5fa), 0.15);
  }
  return base.getHex();
}

function resolveEdgeColor(edge: GraphEdge, groups: Map<string, string>): number {
  const sourceGroup = groups.get(edge.source);
  const targetGroup = groups.get(edge.target);
  if (sourceGroup && targetGroup) {
    if (sourceGroup === targetGroup) {
      return getGroupColor(sourceGroup);
    }
    return blendColors(getGroupColor(sourceGroup), getGroupColor(targetGroup), 0.45);
  }
  const kind = edge.kind || 'signal';
  return theme.edgeColors.get(kind) ?? crossGroupColor;
}

function cleanupScene() {
  if (graphGroup && scene) {
    scene.remove(graphGroup);
    graphGroup.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((mat) => mat.dispose());
      } else if (mesh.material) {
        mesh.material.dispose();
      }
    });
  }
  graphGroup = null;
}

function layoutNodes(nodes: GraphNode[]): Map<string, THREE.Vector3> {
  const groups = new Map<string, GraphNode[]>();
  nodes.forEach((node) => {
    const key = node.group || 'asset';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(node);
  });

  const config: Record<string, { radius: number; height: number; start: number }> = {
    system: { radius: 42, height: 36, start: 0 },
    module: { radius: 72, height: 18, start: Math.PI / 6 },
    asset: { radius: 128, height: -14, start: Math.PI / 12 },
    event: { radius: 96, height: 28, start: Math.PI / 4 },
    session: { radius: 56, height: 64, start: Math.PI / 6 },
    holding: { radius: 188, height: -28, start: Math.PI / 5 },
    native: { radius: 140, height: -6, start: Math.PI / 3 },
    finance: { radius: 72, height: 58, start: Math.PI / 3 },
    ghost: { radius: 150, height: -22, start: Math.PI / 8 },
    ghost_trade: { radius: 210, height: -46, start: Math.PI / 2.8 },
    live_trade: { radius: 236, height: -62, start: Math.PI / 2.2 },
    feedback: { radius: 108, height: 72, start: Math.PI / 2.6 },
    metric: { radius: 132, height: 56, start: Math.PI / 2.1 },
    discovery: { radius: 248, height: 24, start: Math.PI / 5 },
    vote: { radius: 86, height: 78, start: Math.PI / 2.4 },
    queue: { radius: 120, height: -32, start: Math.PI / 1.8 },
  };

  const positions = new Map<string, THREE.Vector3>();
  groups.forEach((nodesInGroup, key) => {
    const base = config[key] || { radius: 140, height: 0, start: Math.PI / 5 };
    const step = (Math.PI * 2) / Math.max(nodesInGroup.length, 1);
    nodesInGroup.forEach((node, idx) => {
      const angle = base.start + step * idx;
      const radius = base.radius * (0.82 + Math.random() * 0.3);
      const x = radius * Math.cos(angle);
      const z = radius * Math.sin(angle);
      const elevation = base.height + (node.value || 0) * 2.4;
      positions.set(node.id, new THREE.Vector3(x, elevation, z));
    });
  });
  return positions;
}

function buildGraph(graph?: { nodes?: GraphNode[]; edges?: GraphEdge[] }) {
  if (!scene) return;
  cleanupScene();
  graphGroup = new THREE.Group();

  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  const positions = layoutNodes(nodes);
  const nodeGroups = new Map<string, string>();
  nodes.forEach((node) => nodeGroups.set(node.id, node.group || 'asset'));

  const labelScale = Number(props.labelScale ?? 1);

  nodes.forEach((node) => {
    const position = positions.get(node.id) ?? new THREE.Vector3(0, 0, 0);
    const radius = Math.max(0.18, Math.min(0.7, Math.abs(node.value || node.exposure || 0.35)));
    const geometry = new THREE.SphereGeometry(radius, 18, 18);
    const nodeColor = resolveNodeColor(node);
    const energy = Math.min(1.2, Math.max(0.28, Math.log1p(Math.abs(Number(node.exposure || node.value || 0.35))) * 0.18 + 0.25));
    const material = new THREE.MeshStandardMaterial({
      color: nodeColor,
      emissive: nodeColor,
      emissiveIntensity: 0.25 + energy * 0.55,
      metalness: 0.25,
      roughness: 0.35,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    graphGroup!.add(mesh);

    const glowGeometry = new THREE.SphereGeometry(radius * (1.6 + energy * 0.4), 16, 16);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: nodeColor,
      transparent: true,
      opacity: 0.08 + energy * 0.1,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    glow.position.copy(position);
    graphGroup!.add(glow);

    if (node.label) {
      const label = createTextSprite(node.label, labelScale);
      label.position.copy(position.clone().add(new THREE.Vector3(0, radius + 4.5 * labelScale, 0)));
      graphGroup!.add(label);
    }
  });

  edges.forEach((edge) => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (!from || !to) return;
    const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
    const material = new THREE.LineBasicMaterial({
      color: resolveEdgeColor(edge, nodeGroups),
      transparent: true,
      opacity: Math.min(0.8, Math.max(0.18, (edge.weight || 0.28) * 1.25)),
    });
    const line = new THREE.Line(geometry, material);
    graphGroup!.add(line);
  });

  scene.add(graphGroup);
}

function createTextSprite(text: string, scale: number) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  const fontSize = 24;
  ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;
  const textWidth = ctx.measureText(text).width;
  canvas.width = textWidth + 28;
  canvas.height = fontSize * 2;
  ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;
  ctx.fillStyle = 'rgba(10, 23, 38, 0.8)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#dbeafe';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 14, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  const scaleFactor = 0.026 * (Number.isFinite(scale) ? scale : 1);
  sprite.scale.set(canvas.width * scaleFactor, canvas.height * scaleFactor, 1);
  return sprite;
}

function scheduleReset() {
  if (resetTimeout) window.clearTimeout(resetTimeout);
  resetTimeout = window.setTimeout(() => {
    zoomTarget = 0;
    resetTimeout = null;
  }, 1200);
}

function updateZoom() {
  if (!camera) return;
  const now = performance.now();
  currentZoom += (zoomTarget - currentZoom) * 0.12;
  const targetY = 220 - currentZoom;
  const targetZ = 260 - currentZoom * 1.35;
  camera.position.y += (targetY - camera.position.y) * 0.085;
  camera.position.z += (targetZ - camera.position.z) * 0.085;
  camera.lookAt(0, 0, 0);
  if (!isZooming && now - lastInteraction > 220 && Math.abs(currentZoom) < 0.01) {
    zoomTarget = 0;
  }
}

function resetCamera(force = false) {
  if (!camera) return;
  if (force) {
    currentZoom = 0;
    zoomTarget = 0;
    camera.position.set(0, 220, 260);
    camera.lookAt(0, 0, 0);
    return;
  }
  zoomTarget = 0;
  scheduleReset();
}

function animate() {
  if (!renderer || !scene || !camera) return;
  frameHandle = requestAnimationFrame(animate);
  updateZoom();
  renderer.render(scene, camera);
}

function onResize() {
  if (!renderer || !camera || !canvasContainer.value) return;
  const { clientWidth, clientHeight } = canvasContainer.value;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
}

function onPointerDown(event: MouseEvent) {
  isZooming = true;
  lastInteraction = performance.now();
  zoomStartY = event.clientY;
  zoomTarget = currentZoom;
  if (resetTimeout) {
    window.clearTimeout(resetTimeout);
    resetTimeout = null;
  }
}

function onPointerMove(event: MouseEvent) {
  if (!isZooming) return;
  lastInteraction = performance.now();
  const delta = (zoomStartY - event.clientY) * 0.002;
  zoomTarget = Math.max(-140, Math.min(140, currentZoom + delta * 46));
}

function onPointerUp() {
  if (!isZooming) return;
  isZooming = false;
  scheduleReset();
}

function onTouchStart(event: TouchEvent) {
  if (event.touches.length !== 1) return;
  isZooming = true;
  lastInteraction = performance.now();
  zoomStartY = event.touches[0].clientY;
  zoomTarget = currentZoom;
  event.preventDefault();
  if (resetTimeout) {
    window.clearTimeout(resetTimeout);
    resetTimeout = null;
  }
}

function onTouchMove(event: TouchEvent) {
  if (!isZooming || event.touches.length !== 1) return;
  lastInteraction = performance.now();
  const delta = (zoomStartY - event.touches[0].clientY) * 0.002;
  zoomTarget = Math.max(-140, Math.min(140, currentZoom + delta * 46));
  event.preventDefault();
}

function onTouchEnd(event: TouchEvent) {
  if (!isZooming) return;
  isZooming = false;
  scheduleReset();
  event.preventDefault();
}

function addInteractionHandlers(element: HTMLElement) {
  element.addEventListener('mousedown', onPointerDown);
  window.addEventListener('mousemove', onPointerMove);
  window.addEventListener('mouseup', onPointerUp);
  element.addEventListener('mouseleave', onPointerUp);
  element.addEventListener('touchstart', onTouchStart, { passive: false });
  window.addEventListener('touchmove', onTouchMove, { passive: false });
  window.addEventListener('touchend', onTouchEnd);
}

function removeInteractionHandlers() {
  const element = renderer?.domElement;
  if (!element) return;
  element.removeEventListener('mousedown', onPointerDown);
  window.removeEventListener('mousemove', onPointerMove);
  window.removeEventListener('mouseup', onPointerUp);
  element.removeEventListener('mouseleave', onPointerUp);
  element.removeEventListener('touchstart', onTouchStart);
  window.removeEventListener('touchmove', onTouchMove);
  window.removeEventListener('touchend', onTouchEnd);
}

function initScene() {
  const container = canvasContainer.value;
  if (!container) return;

  const { clientWidth, clientHeight } = container;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(theme.background);

  camera = new THREE.PerspectiveCamera(60, clientWidth / clientHeight, 0.1, 800);
  camera.position.set(0, 220, 260);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(clientWidth, clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const ambient = new THREE.AmbientLight(0x94a3b8, 0.7);
  scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0x60a5fa, 0.9);
  keyLight.position.set(20, 30, 10);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.5);
  rimLight.position.set(-25, -15, -10);
  scene.add(rimLight);

  container.appendChild(renderer.domElement);
  window.addEventListener('resize', onResize, { passive: true });

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.enableRotate = false;

  addInteractionHandlers(renderer.domElement);
  animate();
}

function disposeRenderer() {
  if (frameHandle !== null) {
    cancelAnimationFrame(frameHandle);
    frameHandle = null;
  }
  window.removeEventListener('resize', onResize);
  removeInteractionHandlers();
  if (renderer && renderer.domElement.parentElement) {
    renderer.domElement.parentElement.removeChild(renderer.domElement);
  }
  renderer?.dispose();
  renderer = null;
  scene = null;
  camera = null;
  if (controls) {
    controls.dispose();
    controls = null;
  }
}

onMounted(() => {
  initScene();
  buildGraph(props.graph);
  resetCamera(true);
});

onBeforeUnmount(() => {
  cleanupScene();
  disposeRenderer();
});

watch(
  () => props.graph,
  (graph) => {
    if (!scene) return;
    buildGraph(graph);
    resetCamera(true);
  },
  { deep: true },
);

watch(
  () => props.labelScale,
  () => {
    if (!scene) return;
    buildGraph(props.graph);
  },
);
</script>

<style scoped>
.organism-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 420px;
  border-radius: 18px;
  overflow: hidden;
  background: radial-gradient(circle at top, #0f172a, #020617 70%);
  box-shadow: 0 24px 60px rgba(8, 47, 73, 0.35);
  border: 1px solid rgba(59, 130, 246, 0.12);
}

.organism-canvas canvas {
  display: block;
}
</style>
