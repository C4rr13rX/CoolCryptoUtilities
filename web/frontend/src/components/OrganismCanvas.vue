<template>
  <div ref="canvasContainer" class="organism-canvas">
    <div v-if="playbackActive" class="playback-hud">
      <span class="playback-label">▶ {{ playbackLabel }}</span>
      <button class="playback-stop" @click="stopPlayback">■</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Props / emits
// ---------------------------------------------------------------------------

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

interface Snapshot {
  ts: number;
  label?: string;
  nodeValues: Record<string, number>;
}

const props = defineProps<{
  graph?: { nodes?: GraphNode[]; edges?: GraphEdge[] };
  labelScale?: number;
}>();

const emit = defineEmits<{
  (e: 'tradeEvent', payload: { symbol: string; size: number; direction: 'enter' | 'exit' }): void;
}>();

// ---------------------------------------------------------------------------
// Three.js refs
// ---------------------------------------------------------------------------

const canvasContainer = ref<HTMLDivElement | null>(null);

let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let renderer: THREE.WebGLRenderer | null = null;
let graphGroup: THREE.Group | null = null;
let frameHandle: number | null = null;
let lastFrame = 0;

// Per-node fire emitters: nodeId → {points, geo, velocities, basePos}
type FireEmitter = {
  points: THREE.Points;
  geo: THREE.BufferGeometry;
  pos: Float32Array;
  vel: Float32Array;
  life: Float32Array;
  baseX: number;
  baseZ: number;
  count: number;
};
const fireEmitters = new Map<string, FireEmitter>();

// Snapshot store + playback
const snapshots: Snapshot[] = [];
let playbackTimer: number | null = null;
let playbackIdx = 0;
const playbackActive = ref(false);
const playbackLabel = ref('');

// Node mesh registry for snapshot-driven resize
const nodeMeshes = new Map<string, THREE.Mesh>();

// Trade burst registry: [{pos, vel, geo, pts, life}] for one-shot bursts
type TradeBurst = {
  pts: THREE.Points;
  geo: THREE.BufferGeometry;
  vel: Float32Array;
  life: Float32Array;
  age: number;
};
const tradeBursts: TradeBurst[] = [];

// ---------------------------------------------------------------------------
// Brain layout — flat 2D top-down (x, z plane) arranged like a brain
// ---------------------------------------------------------------------------

const domainLayout2D: Record<string, { cx: number; cz: number; radius: number }> = {
  // Frontal lobes — core system logic, session, neural
  core:      { cx:   0,   cz: -90,  radius: 70 },
  neural:    { cx:   0,   cz: -50,  radius: 40 },
  session:   { cx:   0,   cz: -175, radius: 55 },
  // Motor / execution — upper lateral
  execution: { cx:   0,   cz: -130, radius: 80 },
  // Temporal lobes — market data, scouts
  market:    { cx: 190,   cz:  -20, radius: 95 },
  scout:     { cx:-190,   cz:  -20, radius: 85 },
  // Parietal — memory, ghost/live trading
  memory:    { cx: -90,   cz:   60, radius: 70 },
  ghost:     { cx:-160,   cz:  100, radius: 55 },
  live:      { cx: 160,   cz:  100, radius: 55 },
  // Occipital / cerebellum — treasury, feedback
  treasury:  { cx:  40,   cz:  155, radius: 80 },
  feedback:  { cx: -40,   cz:  200, radius: 50 },
};

// Domain color palette
const domainColors: Record<string, number> = {
  core:      0x7dd3fc,
  neural:    0xa855f7,
  session:   0x93c5fd,
  execution: 0xf472b6,
  market:    0x34d399,
  scout:     0x3b82f6,
  memory:    0x22d3ee,
  ghost:     0xf43f5e,
  live:      0x0ea5e9,
  treasury:  0xf59e0b,
  feedback:  0xfacc15,
};

function resolveDomain(node: GraphNode): string {
  const explicit = (node.domain || '').toLowerCase();
  if (explicit && domainLayout2D[explicit]) return explicit;
  const g = (node.group || '').toLowerCase();
  if (g.startsWith('graph') || g === 'brain') return 'neural';
  if (g === 'ghost_trade' || g === 'ghost') return 'ghost';
  if (g === 'live_trade') return 'live';
  if (g.startsWith('window')) return 'memory';
  if (g === 'system' || g === 'module' || g === 'vote' || g === 'transition') return 'core';
  if (g === 'metric' || g === 'memory') return 'memory';
  if (g === 'queue' || g === 'event') return 'execution';
  if (g === 'session') return 'session';
  if (g === 'discovery') return 'scout';
  if (g === 'feedback') return 'feedback';
  if (g === 'holding' || g === 'native' || g === 'finance') return 'treasury';
  return 'market';
}

function getDomainColor(domain: string): number {
  return domainColors[domain] ?? 0x22c55e;
}

// ---------------------------------------------------------------------------
// Layout: flat top-down positions (y=0 plane)
// ---------------------------------------------------------------------------

function layoutNodes(nodes: GraphNode[]): { positions: Map<string, THREE.Vector3>; domains: Map<string, string> } {
  const domains = new Map<string, string>();
  const buckets = new Map<string, GraphNode[]>();
  for (const n of nodes) {
    const d = resolveDomain(n);
    domains.set(n.id, d);
    if (!buckets.has(d)) buckets.set(d, []);
    buckets.get(d)!.push(n);
  }

  const positions = new Map<string, THREE.Vector3>();
  const fallback = { cx: 0, cz: 0, radius: 100 };

  buckets.forEach((group, domain) => {
    const layout = domainLayout2D[domain] ?? fallback;
    const step = (Math.PI * 2) / Math.max(group.length, 1);
    group.forEach((node, idx) => {
      const angle = (idx / Math.max(group.length, 1)) * Math.PI * 2;
      const jitter = 0.65 + Math.random() * 0.5;
      const r = layout.radius * jitter;
      const x = layout.cx + r * Math.cos(angle);
      const z = layout.cz + r * Math.sin(angle);
      // y = 0 for flat brain look. Small value-based lift for "active" nodes.
      const valueLift = Math.min(8, Math.log1p(Math.abs(node.value ?? node.exposure ?? 0)) * 2);
      positions.set(node.id, new THREE.Vector3(x, valueLift, z));
    });
  });

  return { positions, domains };
}

// ---------------------------------------------------------------------------
// Fire particle emitter per node
// ---------------------------------------------------------------------------

function nodeParticleCount(node: GraphNode): number {
  const v = Math.abs(node.value ?? node.exposure ?? 0.1);
  return Math.max(8, Math.min(200, Math.floor(Math.log1p(v) * 30 + 10)));
}

function buildFireEmitter(node: GraphNode, x: number, z: number): FireEmitter {
  const count = nodeParticleCount(node);
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  const vel = new Float32Array(count * 3);
  const life = new Float32Array(count);

  const domainHex = getDomainColor(resolveDomain(node));
  const baseColor = new THREE.Color(domainHex);

  // Generate per-particle colors biased toward fire (orange→yellow→white)
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    // Init spread around node
    pos[i * 3]     = x + (Math.random() - 0.5) * 4;
    pos[i * 3 + 1] = Math.random() * 2;
    pos[i * 3 + 2] = z + (Math.random() - 0.5) * 4;
    // Velocity: upward drift with random horizontal scatter
    vel[i * 3]     = (Math.random() - 0.5) * 0.4;
    vel[i * 3 + 1] = 0.8 + Math.random() * 1.6;  // upward flame
    vel[i * 3 + 2] = (Math.random() - 0.5) * 0.4;
    life[i] = Math.random(); // staggered starts

    // Color: interpolate from domain color → orange → yellow at tip
    const t = Math.random();
    const c = baseColor.clone().lerp(new THREE.Color(0xff6b00), 0.4 + t * 0.5);
    c.lerp(new THREE.Color(0xffd700), t * 0.3);
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }

  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 2.4,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.75,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const pts = new THREE.Points(geo, mat);
  return { points: pts, geo, pos, vel, life, baseX: x, baseZ: z, count };
}

function updateFireEmitter(emitter: FireEmitter, delta: number, now: number): void {
  const { pos, vel, life, baseX, baseZ, count, geo } = emitter;
  const maxHeight = 18 + Math.sin(now * 0.001) * 3; // flame flicker

  for (let i = 0; i < count; i++) {
    life[i] += delta * (0.4 + Math.random() * 0.3);
    if (life[i] > 1.0) {
      // Reset: birth new particle at base
      pos[i * 3]     = baseX + (Math.random() - 0.5) * 5;
      pos[i * 3 + 1] = 0;
      pos[i * 3 + 2] = baseZ + (Math.random() - 0.5) * 5;
      vel[i * 3]     = (Math.random() - 0.5) * 0.5;
      vel[i * 3 + 1] = 0.8 + Math.random() * 2.0;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.5;
      life[i] = 0;
    }
    // Move particle
    pos[i * 3]     += vel[i * 3] * delta * 30;
    pos[i * 3 + 1] += vel[i * 3 + 1] * delta * 30;
    pos[i * 3 + 2] += vel[i * 3 + 2] * delta * 30;
    // Gravity-like drag on horizontal velocity
    vel[i * 3]     *= 0.98;
    vel[i * 3 + 2] *= 0.98;
    // Cap height
    if (pos[i * 3 + 1] > maxHeight) {
      life[i] = 1.0; // will reset next frame
    }
  }
  (geo.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;

  // Fade opacity by life
  const mat = emitter.points.material as THREE.PointsMaterial;
  mat.opacity = 0.65 + 0.15 * Math.sin(now * 0.003);
}

// ---------------------------------------------------------------------------
// Trade burst — one-shot particle explosion at a node position
// ---------------------------------------------------------------------------

function spawnTradeBurst(x: number, y: number, z: number, size: number, direction: 'enter' | 'exit'): void {
  if (!graphGroup) return;
  const count = Math.min(120, Math.max(20, Math.floor(size * 60)));
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  const vel = new Float32Array(count * 3);
  const life = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  const enterColor = new THREE.Color(0x22d3ee); // cyan for buy
  const exitColor = new THREE.Color(0xf97316);  // orange for sell

  for (let i = 0; i < count; i++) {
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI;
    const speed = 2.0 + Math.random() * 4.0;
    vel[i * 3]     = Math.sin(phi) * Math.cos(theta) * speed;
    vel[i * 3 + 1] = Math.cos(phi) * speed * 0.6;
    vel[i * 3 + 2] = Math.sin(phi) * Math.sin(theta) * speed;
    life[i] = 0;
    const c = direction === 'enter' ? enterColor.clone() : exitColor.clone();
    c.lerp(new THREE.Color(0xffffff), Math.random() * 0.3);
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 3.5, vertexColors: true, sizeAttenuation: true,
    transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const pts = new THREE.Points(geo, mat);
  graphGroup.add(pts);
  tradeBursts.push({ pts, geo, vel, life, age: 0 });
}

function updateTradeBursts(delta: number): void {
  for (let b = tradeBursts.length - 1; b >= 0; b--) {
    const burst = tradeBursts[b];
    burst.age += delta;
    const posAttr = burst.geo.getAttribute('position') as THREE.BufferAttribute;
    const life = burst.life;
    const vel = burst.vel;
    let alive = 0;
    for (let i = 0; i < posAttr.count; i++) {
      life[i] += delta * 0.8;
      if (life[i] >= 1.0) continue;
      alive++;
      posAttr.setX(i, posAttr.getX(i) + vel[i * 3] * delta * 30);
      posAttr.setY(i, posAttr.getY(i) + vel[i * 3 + 1] * delta * 30);
      posAttr.setZ(i, posAttr.getZ(i) + vel[i * 3 + 2] * delta * 30);
      vel[i * 3 + 1] -= 2.0 * delta; // gravity
    }
    posAttr.needsUpdate = true;
    const mat = burst.pts.material as THREE.PointsMaterial;
    mat.opacity = Math.max(0, 0.9 * (1 - burst.age * 1.2));

    if (alive === 0 || burst.age > 2.0) {
      graphGroup?.remove(burst.pts);
      burst.geo.dispose();
      tradeBursts.splice(b, 1);
    }
  }
}

// ---------------------------------------------------------------------------
// Scene build
// ---------------------------------------------------------------------------

function cleanupScene(): void {
  fireEmitters.forEach(({ points, geo }) => {
    graphGroup?.remove(points);
    geo.dispose();
  });
  fireEmitters.clear();
  nodeMeshes.clear();
  tradeBursts.forEach(({ pts, geo }) => { graphGroup?.remove(pts); geo.dispose(); });
  tradeBursts.length = 0;

  if (graphGroup && scene) {
    scene.remove(graphGroup);
    graphGroup.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) mesh.material.forEach((m) => m.dispose());
      else if (mesh.material) (mesh.material as THREE.Material).dispose();
    });
  }
  graphGroup = null;
}

function buildGraph(graph?: { nodes?: GraphNode[]; edges?: GraphEdge[] }): void {
  if (!scene) return;
  cleanupScene();
  graphGroup = new THREE.Group();
  lastFrame = performance.now();

  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  const { positions, domains } = layoutNodes(nodes);

  const labelScale = Number(props.labelScale ?? 1);

  // Draw domain background blobs (brain regions)
  const drawnDomains = new Set<string>();
  nodes.forEach((n) => drawnDomains.add(domains.get(n.id)!));
  drawnDomains.forEach((domain) => {
    const layout = domainLayout2D[domain];
    if (!layout) return;
    const geo = new THREE.CircleGeometry(layout.radius * 1.15, 48);
    const col = new THREE.Color(getDomainColor(domain));
    const mat = new THREE.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.045,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(layout.cx, -0.5, layout.cz);
    graphGroup!.add(mesh);
    // Domain ring outline
    const ringGeo = new THREE.RingGeometry(layout.radius * 1.12, layout.radius * 1.17, 48);
    const ringMat = new THREE.MeshBasicMaterial({
      color: col, transparent: true, opacity: 0.15,
      side: THREE.DoubleSide, depthWrite: false,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(layout.cx, -0.2, layout.cz);
    graphGroup!.add(ring);
  });

  // Draw edges
  nodes.forEach((n) => {
    domains.set(n.id, resolveDomain(n));
  });
  edges.forEach((edge) => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (!from || !to) return;
    const srcDomain = domains.get(edge.source) ?? 'core';
    const tgtDomain = domains.get(edge.target) ?? 'core';
    const col = srcDomain === tgtDomain
      ? new THREE.Color(getDomainColor(srcDomain))
      : new THREE.Color(0xff7a18);
    const geo = new THREE.BufferGeometry().setFromPoints([from, to]);
    const mat = new THREE.LineBasicMaterial({
      color: col,
      transparent: true,
      opacity: Math.min(0.55, Math.max(0.08, (edge.weight ?? 0.25) * 0.9)),
    });
    graphGroup!.add(new THREE.Line(geo, mat));
  });

  // Draw nodes + fire emitters
  nodes.forEach((node) => {
    const pos = positions.get(node.id) ?? new THREE.Vector3();
    const domain = domains.get(node.id) ?? 'market';
    const col = new THREE.Color(getDomainColor(domain));

    // Status tint
    const status = (node.status ?? 'ok').toLowerCase();
    if (['halted', 'warn', 'error'].includes(status)) col.lerp(new THREE.Color(0xef4444), 0.4);
    else if (['strong', 'engaged'].includes(status)) col.lerp(new THREE.Color(0x4ade80), 0.25);

    const v = Math.abs(node.value ?? node.exposure ?? 0.2);
    const radius = Math.max(2.5, Math.min(9.0, Math.log1p(v) * 2.5 + 2.5));

    // Core sphere
    const geo = new THREE.SphereGeometry(radius, 20, 20);
    const mat = new THREE.MeshStandardMaterial({
      color: col,
      emissive: col,
      emissiveIntensity: 0.6,
      metalness: 0.2,
      roughness: 0.4,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    graphGroup!.add(mesh);
    nodeMeshes.set(node.id, mesh);

    // Glow halo (flat circle on ground plane)
    const haloGeo = new THREE.CircleGeometry(radius * 2.2, 32);
    const haloMat = new THREE.MeshBasicMaterial({
      color: col, transparent: true, opacity: 0.12,
      side: THREE.DoubleSide, depthWrite: false,
    });
    const halo = new THREE.Mesh(haloGeo, haloMat);
    halo.rotation.x = -Math.PI / 2;
    halo.position.set(pos.x, 0.1, pos.z);
    graphGroup!.add(halo);

    // Label
    if (node.label) {
      const sprite = createTextSprite(node.label, labelScale);
      sprite.position.set(pos.x, pos.y + radius + 5 * labelScale, pos.z);
      graphGroup!.add(sprite);
    }

    // Fire emitter
    const emitter = buildFireEmitter(node, pos.x, pos.z);
    graphGroup!.add(emitter.points);
    fireEmitters.set(node.id, emitter);
  });

  scene.add(graphGroup);
}

// ---------------------------------------------------------------------------
// Snapshot store + playback
// ---------------------------------------------------------------------------

function addSnapshot(nodeValues: Record<string, number>, label?: string): void {
  snapshots.push({ ts: Date.now(), label, nodeValues: { ...nodeValues } });
  // Keep last 200 snapshots
  if (snapshots.length > 200) snapshots.splice(0, snapshots.length - 200);
}

function applySnapshot(snapshot: Snapshot): void {
  playbackLabel.value = snapshot.label
    ? snapshot.label
    : new Date(snapshot.ts).toLocaleTimeString();

  for (const [nodeId, value] of Object.entries(snapshot.nodeValues)) {
    const mesh = nodeMeshes.get(nodeId);
    const emitter = fireEmitters.get(nodeId);
    if (!mesh) continue;
    // Scale sphere radius to match value
    const newRadius = Math.max(2.5, Math.min(9.0, Math.log1p(Math.abs(value)) * 2.5 + 2.5));
    mesh.scale.setScalar(newRadius / 5.0); // normalize against default radius of 5
    // Update emitter count by re-scaling material size
    if (emitter) {
      const mat = emitter.points.material as THREE.PointsMaterial;
      mat.size = Math.max(1.0, Math.min(4.5, 1.5 + Math.log1p(Math.abs(value)) * 0.6));
    }
  }
}

function playSnapshots(intervalMs = 800): void {
  if (snapshots.length < 2) return;
  playbackIdx = 0;
  playbackActive.value = true;
  applySnapshot(snapshots[0]);

  playbackTimer = window.setInterval(() => {
    playbackIdx++;
    if (playbackIdx >= snapshots.length) {
      stopPlayback();
      return;
    }
    applySnapshot(snapshots[playbackIdx]);
  }, intervalMs);
}

function stopPlayback(): void {
  if (playbackTimer !== null) {
    window.clearInterval(playbackTimer);
    playbackTimer = null;
  }
  playbackActive.value = false;
}

// ---------------------------------------------------------------------------
// Trade event — called externally to trigger a particle burst
// ---------------------------------------------------------------------------

function onTradeEvent(symbol: string, size: number, direction: 'enter' | 'exit'): void {
  // Find the node matching this symbol
  const nodes = props.graph?.nodes ?? [];
  const node = nodes.find((n) =>
    n.id.toLowerCase().includes(symbol.toLowerCase()) ||
    (n.label ?? '').toLowerCase().includes(symbol.toLowerCase())
  );
  if (!node) return;
  const emitter = fireEmitters.get(node.id);
  const mesh = nodeMeshes.get(node.id);
  if (!emitter && !mesh) return;
  const x = emitter?.baseX ?? mesh?.position.x ?? 0;
  const z = emitter?.baseZ ?? mesh?.position.z ?? 0;
  const y = mesh?.position.y ?? 0;
  spawnTradeBurst(x, y + 5, z, size, direction);
}

// ---------------------------------------------------------------------------
// Animation
// ---------------------------------------------------------------------------

function animate(): void {
  if (!renderer || !scene || !camera) return;
  const now = performance.now();
  if (!lastFrame) lastFrame = now;
  const delta = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  frameHandle = requestAnimationFrame(animate);

  // Update all fire emitters
  fireEmitters.forEach((emitter) => updateFireEmitter(emitter, delta, now));

  // Update trade bursts
  updateTradeBursts(delta);

  // Very slow axial rotation on the whole brain (gives life without breaking top-down feel)
  if (graphGroup) {
    graphGroup.rotation.y += delta * 0.008;
  }

  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Text sprite
// ---------------------------------------------------------------------------

function createTextSprite(text: string, scale: number): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  const fontSize = 22;
  ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;
  const textWidth = ctx.measureText(text).width;
  canvas.width = textWidth + 24;
  canvas.height = fontSize * 1.8;
  ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;
  ctx.fillStyle = 'rgba(5, 11, 18, 0.78)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#e0f2fe';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 12, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(mat);
  const f = 0.024 * (Number.isFinite(scale) ? scale : 1);
  sprite.scale.set(canvas.width * f, canvas.height * f, 1);
  return sprite;
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

function onResize(): void {
  if (!renderer || !camera || !canvasContainer.value) return;
  const { clientWidth, clientHeight } = canvasContainer.value;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function initScene(): void {
  const container = canvasContainer.value;
  if (!container) return;
  const { clientWidth, clientHeight } = container;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050b12);
  scene.fog = new THREE.FogExp2(0x050b12, 0.0008);

  // Top-down bird's-eye camera
  camera = new THREE.PerspectiveCamera(50, clientWidth / clientHeight, 0.5, 1200);
  camera.position.set(0, 480, 0);
  camera.lookAt(0, 0, 0);
  // Tilt very slightly forward so nodes are fully visible from top
  camera.position.z = 40;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(clientWidth, clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Lighting
  scene.add(new THREE.AmbientLight(0xc0d8f0, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 0.6);
  key.position.set(10, 100, 20);
  scene.add(key);

  window.addEventListener('resize', onResize, { passive: true });
  animate();
}

function disposeRenderer(): void {
  if (frameHandle !== null) { cancelAnimationFrame(frameHandle); frameHandle = null; }
  window.removeEventListener('resize', onResize);
  stopPlayback();
  if (renderer?.domElement.parentElement) {
    renderer.domElement.parentElement.removeChild(renderer.domElement);
  }
  renderer?.dispose();
  renderer = null; scene = null; camera = null;
}

// ---------------------------------------------------------------------------
// Lifecycle + reactivity
// ---------------------------------------------------------------------------

onMounted(() => {
  initScene();
  buildGraph(props.graph);
});

onBeforeUnmount(() => {
  cleanupScene();
  disposeRenderer();
});

watch(() => props.graph, (g) => {
  if (!scene) return;
  buildGraph(g);
}, { deep: true });

watch(() => props.labelScale, () => {
  if (!scene) return;
  buildGraph(props.graph);
});

// Expose external API
defineExpose({ addSnapshot, playSnapshots, stopPlayback, onTradeEvent });
</script>

<style scoped>
.organism-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 440px;
  border-radius: 18px;
  overflow: hidden;
  background: radial-gradient(ellipse at 50% 30%, rgba(56, 189, 248, 0.10), transparent 60%),
              radial-gradient(ellipse at 50% 80%, rgba(168, 85, 247, 0.07), transparent 55%),
              #050b12;
  box-shadow: 0 24px 60px rgba(8, 47, 73, 0.4), 0 0 0 1px rgba(59, 130, 246, 0.10);
}

.organism-canvas canvas { display: block; }

.playback-hud {
  position: absolute;
  top: 12px;
  left: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(5, 11, 18, 0.75);
  border: 1px solid rgba(96, 165, 250, 0.25);
  border-radius: 8px;
  padding: 4px 10px;
  backdrop-filter: blur(6px);
  z-index: 10;
}

.playback-label {
  font-size: 12px;
  font-weight: 600;
  color: #7dd3fc;
  letter-spacing: 0.03em;
}

.playback-stop {
  background: none;
  border: none;
  color: #f87171;
  font-size: 14px;
  cursor: pointer;
  padding: 0 2px;
  line-height: 1;
}

.organism-canvas::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at 50% 0%, rgba(147, 197, 253, 0.06), transparent 50%);
  pointer-events: none;
}
</style>
