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
  domain?: string;
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
let particleSystem: THREE.Points | null = null;
let particleTrail: THREE.Points | null = null;
let particleGeometry: THREE.BufferGeometry | null = null;
let particleVelocities: Float32Array | null = null;
let particleBounds = 240;
let lastFrame = 0;

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

const domainPalette = new Map<string, number>([
  ['core', 0x7dd3fc],
  ['memory', 0x22d3ee],
  ['execution', 0xf472b6],
  ['market', 0x34d399],
  ['treasury', 0xf59e0b],
  ['ghost', 0xf43f5e],
  ['live', 0x0ea5e9],
  ['session', 0x93c5fd],
  ['scout', 0x3b82f6],
  ['neural', 0xa855f7],
  ['feedback', 0xfacc15],
]);

const groupPalette = new Map<string, number>([
  ['system', domainPalette.get('core') ?? 0x38bdf8],
  ['module', domainPalette.get('core') ?? 0x7c3aed],
  ['asset', domainPalette.get('market') ?? 0x22c55e],
  ['event', domainPalette.get('execution') ?? 0xf472b6],
  ['session', domainPalette.get('session') ?? 0x2dd4bf],
  ['holding', domainPalette.get('treasury') ?? 0xf97316],
  ['native', domainPalette.get('treasury') ?? 0x0ea5e9],
  ['finance', domainPalette.get('treasury') ?? 0xfacc15],
  ['ghost', domainPalette.get('ghost') ?? 0xdb2777],
  ['ghost_trade', domainPalette.get('ghost') ?? 0xdb2777],
  ['live_trade', domainPalette.get('live') ?? 0x0ea5e9],
  ['queue', domainPalette.get('execution') ?? 0x2563eb],
  ['feedback', domainPalette.get('feedback') ?? 0xf59e0b],
  ['metric', domainPalette.get('memory') ?? 0x8b5cf6],
  ['discovery', domainPalette.get('scout') ?? 0x22d3ee],
  ['vote', domainPalette.get('core') ?? 0xfbbf24],
  ['transition', domainPalette.get('core') ?? 0xa855f7],
  ['brain', domainPalette.get('core') ?? 0x38bdf8],
  ['window', domainPalette.get('memory') ?? 0x22d3ee],
  ['window_series', domainPalette.get('memory') ?? 0x22d3ee],
]);

const domainLayout: Record<
  string,
  { center: THREE.Vector3; radius: number; spread: number; start: number; wobble: number }
> = {
  core: { center: new THREE.Vector3(0, 64, 0), radius: 68, spread: 22, start: 0, wobble: 0.6 },
  memory: { center: new THREE.Vector3(-110, 46, -40), radius: 86, spread: 22, start: Math.PI / 3, wobble: 0.9 },
  execution: { center: new THREE.Vector3(110, 24, 44), radius: 88, spread: 28, start: Math.PI / 7, wobble: 0.4 },
  market: { center: new THREE.Vector3(42, -24, -120), radius: 136, spread: 26, start: Math.PI / 1.6, wobble: 0.55 },
  treasury: { center: new THREE.Vector3(-36, -34, 126), radius: 104, spread: 24, start: Math.PI / 2.2, wobble: 0.6 },
  ghost: { center: new THREE.Vector3(-168, -18, 12), radius: 74, spread: 18, start: Math.PI / 9, wobble: 0.35 },
  live: { center: new THREE.Vector3(168, -28, -12), radius: 78, spread: 18, start: Math.PI / 1.3, wobble: 0.35 },
  session: { center: new THREE.Vector3(0, 18, 142), radius: 82, spread: 14, start: Math.PI / 12, wobble: 0.4 },
  scout: { center: new THREE.Vector3(0, 32, -176), radius: 92, spread: 16, start: Math.PI / 2.4, wobble: 0.55 },
  neural: { center: new THREE.Vector3(0, 118, 0), radius: 52, spread: 20, start: Math.PI / 5, wobble: 0.7 },
  feedback: { center: new THREE.Vector3(-76, 26, 96), radius: 60, spread: 16, start: Math.PI / 1.8, wobble: 0.5 },
};

const crossGroupColor = 0xfcd34d;
const interDomainColor = 0xff7a18;
const fieryColor = 0xff6b1a;
const fieryTrailColor = 0xffd166;

function blendColors(a: number, b: number, ratio = 0.5) {
  const colorA = new THREE.Color(a);
  colorA.lerp(new THREE.Color(b), ratio);
  return colorA.getHex();
}

function resolveDomain(node: GraphNode): string {
  const explicit = (node.domain || '').toLowerCase();
  if (explicit) return explicit;
  const group = (node.group || '').toLowerCase();
  if (group.startsWith('graph')) return 'neural';
  if (group.startsWith('ghost_trade') || group === 'ghost') return 'ghost';
  if (group.startsWith('live_trade')) return 'live';
  if (group.startsWith('window')) return 'memory';
  if (group === 'system' || group === 'module' || group === 'vote' || group === 'transition' || group === 'brain') {
    return 'core';
  }
  if (group === 'metric' || group === 'memory') return 'memory';
  if (group === 'queue' || group === 'event') return 'execution';
  if (group === 'session') return 'session';
  if (group === 'discovery') return 'scout';
  if (group === 'feedback') return 'feedback';
  if (group === 'holding' || group === 'native' || group === 'finance') return 'treasury';
  return 'market';
}

function getDomainColor(domain: string) {
  return domainPalette.get(domain) ?? groupPalette.get('asset') ?? 0x22c55e;
}

function getGroupColor(group?: string, domain?: string) {
  if (domain) return getDomainColor(domain);
  if (!group) return groupPalette.get('system') ?? 0x38bdf8;
  return groupPalette.get(group) ?? getDomainColor(resolveDomain({ id: '', group }));
}

function resolveNodeColor(node: GraphNode, domain?: string): number {
  const base = new THREE.Color(getGroupColor(node.group, domain ?? resolveDomain(node)));
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

function resolveEdgeColor(edge: GraphEdge, groups: Map<string, string>, domains: Map<string, string>): number {
  const sourceGroup = groups.get(edge.source);
  const targetGroup = groups.get(edge.target);
  const sourceDomain = domains.get(edge.source);
  const targetDomain = domains.get(edge.target);
  if (sourceDomain && targetDomain) {
    if (sourceDomain === targetDomain) {
      return getDomainColor(sourceDomain);
    }
    return interDomainColor;
  }
  if (sourceGroup && targetGroup) {
    if (sourceGroup === targetGroup) {
      return getGroupColor(sourceGroup, sourceDomain);
    }
    return blendColors(getGroupColor(sourceGroup, sourceDomain), getGroupColor(targetGroup, targetDomain), 0.45);
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
  particleSystem = null;
  particleTrail = null;
  particleGeometry = null;
  particleVelocities = null;
  graphGroup = null;
}

function layoutNodes(nodes: GraphNode[]): { positions: Map<string, THREE.Vector3>; domains: Map<string, string> } {
  const domains = new Map<string, string>();
  const domainBuckets = new Map<string, GraphNode[]>();
  nodes.forEach((node) => {
    const domain = resolveDomain(node);
    domains.set(node.id, domain);
    if (!domainBuckets.has(domain)) domainBuckets.set(domain, []);
    domainBuckets.get(domain)!.push(node);
  });

  const positions = new Map<string, THREE.Vector3>();
  const fallback = { center: new THREE.Vector3(0, 0, 0), radius: 140, spread: 16, start: Math.PI / 5, wobble: 0.5 };

  domainBuckets.forEach((nodesInDomain, key) => {
    const base = domainLayout[key] || fallback;
    const step = (Math.PI * 2) / Math.max(nodesInDomain.length, 4);
    nodesInDomain.forEach((node, idx) => {
      const angle = base.start + step * idx;
      const jitter = 0.78 + Math.random() * 0.35;
      const radius = base.radius * jitter;
      const x = base.center.x + radius * Math.cos(angle);
      const z = base.center.z + radius * Math.sin(angle);
      const elevation = base.center.y + Math.sin(angle * base.wobble) * base.spread + (node.value || node.exposure || 0) * 3;
      positions.set(node.id, new THREE.Vector3(x, elevation, z));
    });
  });

  return { positions, domains };
}

function computeUsdWeight(nodes: GraphNode[]) {
  return nodes.reduce((sum, node) => {
    const group = (node.group || '').toLowerCase();
    const base = Math.abs(Number(node.value ?? node.exposure ?? 0.2));
    const multiplier =
      group.startsWith('finance') || group.startsWith('holding') || group.startsWith('native')
        ? 2.4
        : group.includes('trade') || group === 'ghost'
          ? 1.8
          : group === 'asset'
            ? 1.3
            : 1.0;
    return sum + base * multiplier + 0.08;
  }, 0);
}

function buildParticleField(nodes: GraphNode[], positions: Map<string, THREE.Vector3>) {
  if (!graphGroup) return;
  const usdWeight = computeUsdWeight(nodes);
  const count = Math.min(1600, Math.max(160, Math.floor(usdWeight * 140)));
  particleBounds = Math.min(280, Math.max(180, Math.sqrt(usdWeight + positions.size) * 24));

  const geometry = new THREE.BufferGeometry();
  const posArray = new Float32Array(count * 3);
  particleVelocities = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const angle = Math.random() * Math.PI * 2;
    const radius = particleBounds * (0.45 + Math.random() * 0.75);
    const yScatter = (Math.random() * 2 - 1) * (particleBounds * 0.35);
    posArray[i * 3] = Math.cos(angle) * radius;
    posArray[i * 3 + 1] = yScatter;
    posArray[i * 3 + 2] = Math.sin(angle) * radius;
    particleVelocities[i * 3] = (Math.random() * 2 - 1) * 0.6;
    particleVelocities[i * 3 + 1] = (Math.random() * 2 - 1) * 0.4;
    particleVelocities[i * 3 + 2] = (Math.random() * 2 - 1) * 0.6;
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
  particleGeometry = geometry;

  const coreMaterial = new THREE.PointsMaterial({
    color: fieryColor,
    size: 2.2,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.72,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const trailMaterial = new THREE.PointsMaterial({
    color: fieryTrailColor,
    size: 3.8,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.22,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  particleSystem = new THREE.Points(geometry, coreMaterial);
  particleTrail = new THREE.Points(geometry, trailMaterial);
  graphGroup.add(particleTrail);
  graphGroup.add(particleSystem);
}

function buildGraph(graph?: { nodes?: GraphNode[]; edges?: GraphEdge[] }) {
  if (!scene) return;
  cleanupScene();
  graphGroup = new THREE.Group();
  lastFrame = performance.now();

  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  const { positions, domains } = layoutNodes(nodes);
  const nodeGroups = new Map<string, string>();
  nodes.forEach((node) => nodeGroups.set(node.id, node.group || 'asset'));

  const labelScale = Number(props.labelScale ?? 1);

  nodes.forEach((node) => {
    const position = positions.get(node.id) ?? new THREE.Vector3(0, 0, 0);
    const radius = Math.max(0.18, Math.min(0.7, Math.abs(node.value || node.exposure || 0.35)));
    const geometry = new THREE.SphereGeometry(radius, 18, 18);
    const nodeColor = resolveNodeColor(node, domains.get(node.id));
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
    const interDomain = domains.get(edge.source) && domains.get(edge.target) && domains.get(edge.source) !== domains.get(edge.target);
    const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
    const material = new THREE.LineBasicMaterial({
      color: resolveEdgeColor(edge, nodeGroups, domains),
      transparent: true,
      opacity: Math.min(0.9, Math.max(0.18, (edge.weight || 0.28) * 1.25) + (interDomain ? 0.12 : 0)),
    });
    const line = new THREE.Line(geometry, material);
    graphGroup!.add(line);
  });

  buildParticleField(nodes, positions);
  scene.add(graphGroup);
}

function updateParticles(delta: number, now: number) {
  if (!particleGeometry || !particleVelocities) return;
  const positionsAttr = particleGeometry.getAttribute('position') as THREE.BufferAttribute;
  const count = positionsAttr.count;
  for (let i = 0; i < count; i += 1) {
    const idx = i * 3;
    let x = positionsAttr.getX(i) + particleVelocities[idx] * delta * 60;
    let y = positionsAttr.getY(i) + particleVelocities[idx + 1] * delta * 50;
    let z = positionsAttr.getZ(i) + particleVelocities[idx + 2] * delta * 60;
    if (Math.abs(x) > particleBounds) {
      x = Math.sign(x) * particleBounds;
      particleVelocities[idx] *= -0.9;
    }
    if (Math.abs(y) > particleBounds * 0.6) {
      y = Math.sign(y) * particleBounds * 0.6;
      particleVelocities[idx + 1] *= -0.92;
    }
    if (Math.abs(z) > particleBounds) {
      z = Math.sign(z) * particleBounds;
      particleVelocities[idx + 2] *= -0.9;
    }
    const bounce = Math.sin(now * 0.002 + i * 0.35) * 0.65;
    positionsAttr.setX(i, x + bounce * 0.25);
    positionsAttr.setY(i, y + bounce * 0.35);
    positionsAttr.setZ(i, z);
  }
  positionsAttr.needsUpdate = true;
  if (particleSystem) {
    particleSystem.rotation.y += delta * 0.08;
    particleSystem.position.y = Math.sin(now * 0.0015) * 1.6;
  }
  if (particleTrail) {
    particleTrail.rotation.y -= delta * 0.04;
    const mat = particleTrail.material as THREE.PointsMaterial;
    mat.opacity = 0.16 + Math.sin(now * 0.002) * 0.06;
  }
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
  const now = performance.now();
  if (!lastFrame) {
    lastFrame = now;
  }
  const delta = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  frameHandle = requestAnimationFrame(animate);
  updateZoom();
  updateParticles(delta, now);
  if (graphGroup) {
    graphGroup.rotation.y += delta * 0.12;
    graphGroup.position.y = Math.sin(now * 0.0011) * 2.4;
  }
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
  background:
    radial-gradient(circle at 20% 18%, rgba(56, 189, 248, 0.16), transparent 32%),
    radial-gradient(circle at 82% 24%, rgba(248, 113, 113, 0.1), transparent 36%),
    radial-gradient(circle at 50% 78%, rgba(52, 211, 153, 0.08), transparent 42%),
    radial-gradient(circle at top, #0f172a, #020617 70%);
  box-shadow: 0 24px 60px rgba(8, 47, 73, 0.35);
  border: 1px solid rgba(59, 130, 246, 0.12);
}

.organism-canvas canvas {
  display: block;
}

.organism-canvas::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 50% 45%, rgba(147, 197, 253, 0.08), transparent 55%),
    linear-gradient(120deg, rgba(15, 23, 42, 0) 40%, rgba(14, 165, 233, 0.12) 50%, rgba(15, 23, 42, 0) 60%);
  pointer-events: none;
  mix-blend-mode: screen;
}
</style>
