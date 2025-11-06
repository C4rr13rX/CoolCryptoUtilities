<template>
  <div ref="canvasContainer" class="organism-canvas"></div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';

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
}>();

const canvasContainer = ref<HTMLDivElement | null>(null);

let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let renderer: THREE.WebGLRenderer | null = null;
let graphGroup: THREE.Group | null = null;
let frameHandle: number | null = null;

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
  ]),
};

function resolveNodeColor(node: GraphNode): number {
  const status = node.status || 'ok';
  return theme.nodeColors.get(status) ?? 0x60a5fa;
}

function resolveEdgeColor(edge: GraphEdge): number {
  const kind = edge.kind || 'signal';
  return theme.edgeColors.get(kind) ?? 0x374151;
}

function cleanupScene() {
  if (graphGroup && scene) {
    scene.remove(graphGroup);
    graphGroup.traverse((child) => {
      if ((child as THREE.Mesh).geometry) {
        (child as THREE.Mesh).geometry.dispose();
      }
      if ((child as THREE.Mesh).material) {
        const material = (child as THREE.Mesh).material;
        if (Array.isArray(material)) {
          material.forEach((mat) => mat.dispose());
        } else {
          material.dispose();
        }
      }
    });
  }
  graphGroup = null;
}

function layoutNodes(nodes: GraphNode[]): Map<string, THREE.Vector3> {
  const groups = new Map<string, GraphNode[]>();
  nodes.forEach((node) => {
    const groupKey = node.group || 'asset';
    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey)!.push(node);
  });

  const layoutConfig: Record<string, { radius: number; height: number; start: number }> = {
    system: { radius: 6, height: 2, start: 0 },
    module: { radius: 10, height: 0.5, start: Math.PI / 6 },
    asset: { radius: 16, height: -0.5, start: Math.PI / 12 },
    event: { radius: 12, height: 2.2, start: Math.PI / 4 },
  };

  const result = new Map<string, THREE.Vector3>();
  groups.forEach((groupNodes, groupKey) => {
    let config = layoutConfig[groupKey];
    if (!config) {
      if (groupKey.startsWith('graph')) {
        config = { radius: 22, height: -2.5, start: Math.PI / 8 };
      } else {
        config = { radius: 16, height: 0, start: Math.PI / 5 };
      }
    }
    const { radius, height, start } = config;
    const step = (Math.PI * 2) / Math.max(groupNodes.length, 1);
    groupNodes.forEach((node, index) => {
      const angle = start + step * index;
      const x = radius * Math.cos(angle);
      const z = radius * Math.sin(angle);
      const y = height + (node.value || 0) * 0.6;
      result.set(node.id, new THREE.Vector3(x, y, z));
    });
  });
  return result;
}

function buildGraph(graph?: { nodes?: GraphNode[]; edges?: GraphEdge[] }) {
  if (!scene) return;
  cleanupScene();
  graphGroup = new THREE.Group();

  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  const positions = layoutNodes(nodes);

  nodes.forEach((node) => {
    const position = positions.get(node.id) ?? new THREE.Vector3(0, 0, 0);
    const radius = Math.max(0.6, Math.min(1.6, Math.abs(node.value || node.exposure || 0.8)));
    const geometry = new THREE.SphereGeometry(radius, 20, 20);
    const material = new THREE.MeshStandardMaterial({
      color: resolveNodeColor(node),
      emissive: resolveNodeColor(node) * 0.4,
      emissiveIntensity: 0.1,
      metalness: 0.15,
      roughness: 0.35,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData = { node };
    graphGroup!.add(mesh);

    if (node.label) {
      const label = createTextSprite(node.label);
      label.position.copy(position.clone().add(new THREE.Vector3(0, radius + 0.6, 0)));
      graphGroup!.add(label);
    }
  });

  edges.forEach((edge) => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (!from || !to) {
      return;
    }
    const points = [from, to];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: resolveEdgeColor(edge),
      transparent: true,
      opacity: Math.min(0.9, Math.max(0.15, (edge.weight || 0.3) * 1.2)),
    });
    const line = new THREE.Line(geometry, material);
    graphGroup!.add(line);
  });

  scene.add(graphGroup);
}

function createTextSprite(text: string) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d')!;
  const fontSize = 32;
  context.font = `600 ${fontSize}px 'Inter', sans-serif`;
  const textWidth = context.measureText(text).width;
  canvas.width = textWidth + 32;
  canvas.height = fontSize * 2;
  context.font = `600 ${fontSize}px 'Inter', sans-serif`;
  context.fillStyle = 'rgba(10, 23, 38, 0.8)';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#dbeafe';
  context.textBaseline = 'middle';
  context.fillText(text, 16, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  const scaleFactor = 0.04;
  sprite.scale.set(canvas.width * scaleFactor, canvas.height * scaleFactor, 1);
  return sprite;
}

function animate() {
  if (!renderer || !scene || !camera) return;
  frameHandle = requestAnimationFrame(animate);
  if (graphGroup) {
    graphGroup.rotation.y += 0.002;
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

function initScene() {
  const container = canvasContainer.value;
  if (!container) return;

  const { clientWidth, clientHeight } = container;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(theme.background);

  camera = new THREE.PerspectiveCamera(50, clientWidth / clientHeight, 0.1, 500);
  camera.position.set(0, 14, 34);

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
  animate();
}

function disposeRenderer() {
  if (frameHandle !== null) {
    cancelAnimationFrame(frameHandle);
    frameHandle = null;
  }
  window.removeEventListener('resize', onResize);
  if (renderer && renderer.domElement.parentElement) {
    renderer.domElement.parentElement.removeChild(renderer.domElement);
  }
  renderer?.dispose();
  renderer = null;
  scene = null;
  camera = null;
}

onMounted(() => {
  initScene();
  buildGraph(props.graph);
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
  },
  { deep: true },
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
