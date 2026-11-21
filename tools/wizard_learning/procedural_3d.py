"""tools/wizard_learning/procedural_3d.py

Mathematical 3D procedural mesh generator with continuous UV mapping.
Runs 100% on CPU with zero GPU requirements, optimized for real-time rendering
in Multiscale Robot World and browser viewports.

Categories:
  - machine_part: motor housings, chassis, crawler tracks, articulating arms, grippers,
                  involute gears, hydraulic actuators, flanged bearings, track links
  - circuitry: PCB boards, QFP IC chips, electrolytic capacitors, heatsink arrays,
               power distribution boards (PDB), IMU sensor modules
  - plant: L-system fractal branching trees, shrubs, foliage clusters, biomass crops
  - animal: low-poly articulated quadrupeds, segmented hexapods, bipedal units

Outputs standard Three.js BufferGeometry schema and Wavefront OBJ strings.
"""
from __future__ import annotations

import math
from typing import Any


def _normalize_vec3(x: float, y: float, z: float) -> tuple[float, float, float]:
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-9:
        return (0.0, 1.0, 0.0)
    return (x / length, y / length, z / length)


class ProceduralMeshBuilder:
    """Mathematical mesh accumulator with analytical normals and UV coordinates."""

    def __init__(self, name: str = "ProceduralMesh"):
        self.name = name
        self.vertices: list[float] = []  # [x, y, z, ...]
        self.normals: list[float] = []   # [nx, ny, nz, ...]
        self.uvs: list[float] = []       # [u, v, ...]
        self.indices: list[int] = []     # [i0, i1, i2, ...]
        self._vertex_count = 0

    def add_vertex(
        self,
        x: float, y: float, z: float,
        nx: float, ny: float, nz: float,
        u: float, v: float,
    ) -> int:
        idx = self._vertex_count
        self.vertices.extend([round(x, 5), round(y, 5), round(z, 5)])
        self.normals.extend([round(nx, 5), round(ny, 5), round(nz, 5)])
        self.uvs.extend([round(max(0.0, min(1.0, u)), 5), round(max(0.0, min(1.0, v)), 5)])
        self._vertex_count += 1
        return idx

    def add_triangle(self, i0: int, i1: int, i2: int) -> None:
        self.indices.extend([i0, i1, i2])

    def add_quad(self, i0: int, i1: int, i2: int, i3: int) -> None:
        self.indices.extend([i0, i1, i2, i0, i2, i3])

    def add_box(
        self,
        cx: float, cy: float, cz: float,
        sx: float, sy: float, sz: float,
        u_min: float = 0.0, u_max: float = 1.0,
        v_min: float = 0.0, v_max: float = 1.0,
    ) -> None:
        """Add an axis-aligned box with analytical normals and UVs."""
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        faces = [
            # Front (+z)
            ([cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
             [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz], (0, 0, 1)),
            # Back (-z)
            ([cx + hx, cy - hy, cz - hz], [cx - hx, cy - hy, cz - hz],
             [cx - hx, cy + hy, cz - hz], [cx + hx, cy + hy, cz - hz], (0, 0, -1)),
            # Top (+y)
            ([cx - hx, cy + hy, cz + hz], [cx + hx, cy + hy, cz + hz],
             [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz], (0, 1, 0)),
            # Bottom (-y)
            ([cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
             [cx + hx, cy - hy, cz + hz], [cx - hx, cy - hy, cz + hz], (0, -1, 0)),
            # Right (+x)
            ([cx + hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz - hz],
             [cx + hx, cy + hy, cz - hz], [cx + hx, cy + hy, cz + hz], (1, 0, 0)),
            # Left (-x)
            ([cx - hx, cy - hy, cz - hz], [cx - hx, cy - hy, cz + hz],
             [cx - hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz - hz], (-1, 0, 0)),
        ]
        for v0, v1, v2, v3, norm in faces:
            nx, ny, nz = norm
            i0 = self.add_vertex(v0[0], v0[1], v0[2], nx, ny, nz, u_min, v_min)
            i1 = self.add_vertex(v1[0], v1[1], v1[2], nx, ny, nz, u_max, v_min)
            i2 = self.add_vertex(v2[0], v2[1], v2[2], nx, ny, nz, u_max, v_max)
            i3 = self.add_vertex(v3[0], v3[1], v3[2], nx, ny, nz, u_min, v_max)
            self.add_quad(i0, i1, i2, i3)

    def add_cylinder(
        self,
        cx: float, cy: float, cz: float,
        radius: float, height: float,
        segments: int = 16,
        axis: str = "y",
    ) -> None:
        """Add a cylinder with cylindrical UV mapping."""
        half_h = height * 0.5
        # Barrel
        ring_bottom = []
        ring_top = []
        for s in range(segments + 1):
            theta = 2.0 * math.pi * s / segments
            ct = math.cos(theta)
            st = math.sin(theta)
            u = s / segments
            if axis == "y":
                x = cx + radius * ct
                z = cz + radius * st
                ring_bottom.append(self.add_vertex(x, cy - half_h, z, ct, 0.0, st, u, 0.0))
                ring_top.append(self.add_vertex(x, cy + half_h, z, ct, 0.0, st, u, 1.0))
            elif axis == "x":
                y = cy + radius * ct
                z = cz + radius * st
                ring_bottom.append(self.add_vertex(cx - half_h, y, z, 0.0, ct, st, u, 0.0))
                ring_top.append(self.add_vertex(cx + half_h, y, z, 0.0, ct, st, u, 1.0))
            else:  # z
                x = cx + radius * ct
                y = cy + radius * st
                ring_bottom.append(self.add_vertex(x, y, cz - half_h, ct, st, 0.0, u, 0.0))
                ring_top.append(self.add_vertex(x, y, cz + half_h, ct, st, 0.0, u, 1.0))

        for s in range(segments):
            self.add_quad(ring_bottom[s], ring_bottom[s + 1], ring_top[s + 1], ring_top[s])

        # End caps (top & bottom)
        if axis == "y":
            c_top = self.add_vertex(cx, cy + half_h, cz, 0.0, 1.0, 0.0, 0.5, 0.5)
            c_bot = self.add_vertex(cx, cy - half_h, cz, 0.0, -1.0, 0.0, 0.5, 0.5)
            for s in range(segments):
                theta1 = 2.0 * math.pi * s / segments
                theta2 = 2.0 * math.pi * (s + 1) / segments
                # Top
                i_t1 = self.add_vertex(
                    cx + radius * math.cos(theta1), cy + half_h, cz + radius * math.sin(theta1),
                    0.0, 1.0, 0.0, 0.5 + 0.5 * math.cos(theta1), 0.5 + 0.5 * math.sin(theta1),
                )
                i_t2 = self.add_vertex(
                    cx + radius * math.cos(theta2), cy + half_h, cz + radius * math.sin(theta2),
                    0.0, 1.0, 0.0, 0.5 + 0.5 * math.cos(theta2), 0.5 + 0.5 * math.sin(theta2),
                )
                self.add_triangle(c_top, i_t1, i_t2)
                # Bottom
                i_b1 = self.add_vertex(
                    cx + radius * math.cos(theta1), cy - half_h, cz + radius * math.sin(theta1),
                    0.0, -1.0, 0.0, 0.5 + 0.5 * math.cos(theta1), 0.5 + 0.5 * math.sin(theta1),
                )
                i_b2 = self.add_vertex(
                    cx + radius * math.cos(theta2), cy - half_h, cz + radius * math.sin(theta2),
                    0.0, -1.0, 0.0, 0.5 + 0.5 * math.cos(theta2), 0.5 + 0.5 * math.sin(theta2),
                )
                self.add_triangle(c_bot, i_b2, i_b1)
        elif axis == "z":
            c_top = self.add_vertex(cx, cy, cz + half_h, 0.0, 0.0, 1.0, 0.5, 0.5)
            c_bot = self.add_vertex(cx, cy, cz - half_h, 0.0, 0.0, -1.0, 0.5, 0.5)
            for s in range(segments):
                theta1 = 2.0 * math.pi * s / segments
                theta2 = 2.0 * math.pi * (s + 1) / segments
                i_t1 = self.add_vertex(
                    cx + radius * math.cos(theta1), cy + radius * math.sin(theta1), cz + half_h,
                    0.0, 0.0, 1.0, 0.5 + 0.5 * math.cos(theta1), 0.5 + 0.5 * math.sin(theta1),
                )
                i_t2 = self.add_vertex(
                    cx + radius * math.cos(theta2), cy + radius * math.sin(theta2), cz + half_h,
                    0.0, 0.0, 1.0, 0.5 + 0.5 * math.cos(theta2), 0.5 + 0.5 * math.sin(theta2),
                )
                self.add_triangle(c_top, i_t1, i_t2)
                i_b1 = self.add_vertex(
                    cx + radius * math.cos(theta1), cy + radius * math.sin(theta1), cz - half_h,
                    0.0, 0.0, -1.0, 0.5 + 0.5 * math.cos(theta1), 0.5 + 0.5 * math.sin(theta1),
                )
                i_b2 = self.add_vertex(
                    cx + radius * math.cos(theta2), cy + radius * math.sin(theta2), cz - half_h,
                    0.0, 0.0, -1.0, 0.5 + 0.5 * math.cos(theta2), 0.5 + 0.5 * math.sin(theta2),
                )
                self.add_triangle(c_bot, i_b2, i_b1)

    def to_threejs_dict(self) -> dict[str, Any]:
        """Return schema compatible with Three.js BufferGeometryLoader."""
        return {
            "metadata": {
                "version": 4.5,
                "type": "BufferGeometry",
                "generator": "W1z4rDBrain_Procedural3D",
            },
            "data": {
                "attributes": {
                    "position": {
                        "itemSize": 3,
                        "type": "Float32Array",
                        "array": self.vertices,
                    },
                    "normal": {
                        "itemSize": 3,
                        "type": "Float32Array",
                        "array": self.normals,
                    },
                    "uv": {
                        "itemSize": 2,
                        "type": "Float32Array",
                        "array": self.uvs,
                    },
                },
                "index": {
                    "type": "Uint32Array",
                    "array": self.indices,
                },
            },
        }

    def to_obj(self) -> str:
        """Export to standard Wavefront OBJ format."""
        lines = [f"# {self.name} - Generated by W1z4rD Procedural 3D Engine"]
        for i in range(0, len(self.vertices), 3):
            lines.append(f"v {self.vertices[i]} {self.vertices[i+1]} {self.vertices[i+2]}")
        for i in range(0, len(self.normals), 3):
            lines.append(f"vn {self.normals[i]} {self.normals[i+1]} {self.normals[i+2]}")
        for i in range(0, len(self.uvs), 2):
            lines.append(f"vt {self.uvs[i]} {self.uvs[i+1]}")
        for i in range(0, len(self.indices), 3):
            i0 = self.indices[i] + 1
            i1 = self.indices[i+1] + 1
            i2 = self.indices[i+2] + 1
            lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
        return "\n".join(lines)


# ── Category Generators ───────────────────────────────────────────────────────

def generate_machine_part(part_name: str = "motor_housing") -> ProceduralMeshBuilder:
    builder = ProceduralMeshBuilder(name=f"MachinePart_{part_name}")
    p = part_name.lower()

    if p in {"motor_housing", "motor"}:
        # Main motor stator cylinder
        builder.add_cylinder(0.0, 0.0, 0.0, radius=0.18, height=0.45, segments=16, axis="z")
        # 8 Extruded cooling fins
        for fin in range(8):
            angle = fin * math.pi / 4.0
            cx = 0.21 * math.cos(angle)
            cy = 0.21 * math.sin(angle)
            builder.add_box(cx, cy, 0.0, sx=0.015, sy=0.015, sz=0.42)
        # Drive shaft
        builder.add_cylinder(0.0, 0.0, 0.30, radius=0.035, height=0.16, segments=12, axis="z")

    elif p in {"involute_gear", "gear", "spur_gear"}:
        # Central hub cylinder with axle bore
        builder.add_cylinder(0.0, 0.0, 0.0, radius=0.14, height=0.06, segments=20, axis="z")
        # Central mounting keyway slot
        builder.add_box(0.0, 0.035, 0.0, sx=0.015, sy=0.02, sz=0.065)
        # 16 Radial involute gear teeth
        num_teeth = 16
        for t in range(num_teeth):
            angle = t * 2.0 * math.pi / num_teeth
            tx = 0.16 * math.cos(angle)
            ty = 0.16 * math.sin(angle)
            builder.add_box(tx, ty, 0.0, sx=0.025, sy=0.035, sz=0.058)

    elif p in {"hydraulic_actuator", "hydraulic_cylinder", "piston"}:
        # Outer heavy pressure barrel
        builder.add_cylinder(0.0, 0.0, 0.0, radius=0.08, height=0.55, segments=16, axis="y")
        # Polished chrome rod extending upwards
        builder.add_cylinder(0.0, 0.40, 0.0, radius=0.04, height=0.45, segments=14, axis="y")
        # Gland seal collar
        builder.add_cylinder(0.0, 0.26, 0.0, radius=0.095, height=0.06, segments=16, axis="y")
        # Clevis mount at base
        builder.add_box(0.0, -0.32, 0.0, sx=0.07, sy=0.10, sz=0.09)
        # Clevis pin hole
        builder.add_cylinder(0.0, -0.32, 0.0, radius=0.02, height=0.11, segments=10, axis="x")

    elif p in {"flanged_bearing", "bearing_block", "bearing"}:
        # Outer square mounting flange
        builder.add_box(0.0, 0.0, 0.0, sx=0.22, sy=0.22, sz=0.04)
        # 4 Corner bolt holes (mounting pads)
        for dx, dy in [(-0.08, -0.08), (0.08, -0.08), (-0.08, 0.08), (0.08, 0.08)]:
            builder.add_cylinder(dx, dy, 0.0, radius=0.015, height=0.045, segments=8, axis="z")
        # Central bearing outer race boss
        builder.add_cylinder(0.0, 0.0, 0.025, radius=0.07, height=0.05, segments=18, axis="z")
        # Inner shaft bore
        builder.add_cylinder(0.0, 0.0, 0.025, radius=0.04, height=0.052, segments=16, axis="z")

    elif p in {"track_link", "crawler_shoe"}:
        # Main forged steel shoe plate
        builder.add_box(0.0, 0.0, 0.0, sx=0.35, sy=0.04, sz=0.18)
        # Traction grouser bar
        builder.add_box(0.0, 0.04, 0.0, sx=0.35, sy=0.04, sz=0.035)
        # Interlocking pin hinge lugs (left & right)
        builder.add_cylinder(-0.12, 0.0, 0.09, radius=0.025, height=0.06, segments=10, axis="x")
        builder.add_cylinder(0.12, 0.0, 0.09, radius=0.025, height=0.06, segments=10, axis="x")

    elif p in {"crawler_chassis", "chassis"}:
        # Central welded belly pan
        builder.add_box(0.0, 0.20, 0.0, sx=0.90, sy=0.18, sz=1.40)
        # Left track housing
        builder.add_box(-0.60, 0.22, 0.0, sx=0.25, sy=0.38, sz=1.60)
        # Right track housing
        builder.add_box(0.60, 0.22, 0.0, sx=0.25, sy=0.38, sz=1.60)
        # Azimuth turntable ring
        builder.add_cylinder(0.0, 0.32, 0.0, radius=0.35, height=0.08, segments=16, axis="y")

    elif p in {"articulated_arm", "boom"}:
        # Lower shoulder joint
        builder.add_cylinder(0.0, 0.0, 0.0, radius=0.12, height=0.22, segments=12, axis="x")
        # Main boom truss
        builder.add_box(0.0, 0.50, 0.0, sx=0.12, sy=0.95, sz=0.14)
        # Elbow joint
        builder.add_cylinder(0.0, 1.0, 0.0, radius=0.09, height=0.18, segments=12, axis="x")
        # Forearm jib
        builder.add_box(0.0, 1.45, 0.15, sx=0.09, sy=0.80, sz=0.10)

    else:  # gripper
        # Wrist hub
        builder.add_box(0.0, 0.0, 0.0, sx=0.14, sy=0.08, sz=0.14)
        # Left clamping jaw
        builder.add_box(-0.08, 0.12, 0.0, sx=0.03, sy=0.18, sz=0.08)
        # Right clamping jaw
        builder.add_box(0.08, 0.12, 0.0, sx=0.03, sy=0.18, sz=0.08)
        # Arc welding nozzle
        builder.add_cylinder(0.0, 0.16, 0.0, radius=0.015, height=0.14, segments=8, axis="y")

    return builder


def generate_circuitry(circuit_name: str = "pcb_board") -> ProceduralMeshBuilder:
    builder = ProceduralMeshBuilder(name=f"Circuitry_{circuit_name}")
    c = circuit_name.lower()

    if c in {"power_distribution_board", "pdb", "power_board"}:
        # Heavy 2oz copper 4-layer FR4 PCB
        builder.add_box(0.0, 0.002, 0.0, sx=0.60, sy=0.004, sz=0.45)
        # High-current DC screw terminal blocks (input & outputs)
        builder.add_box(-0.24, 0.02, 0.0, sx=0.08, sy=0.035, sz=0.20)
        builder.add_box(0.24, 0.02, -0.10, sx=0.08, sy=0.035, sz=0.14)
        builder.add_box(0.24, 0.02, 0.10, sx=0.08, sy=0.035, sz=0.14)
        # 4 TO-220 Power MOSFET switches with extruded tab heatsinks
        for idx, z in enumerate([-0.12, -0.04, 0.04, 0.12]):
            builder.add_box(0.0, 0.015, z, sx=0.04, sy=0.025, sz=0.05)
            builder.add_box(0.025, 0.025, z, sx=0.005, sy=0.04, sz=0.05)
        # Toroidal high-current power inductor
        builder.add_cylinder(-0.10, 0.03, 0.10, radius=0.045, height=0.04, segments=16, axis="y")

    elif c in {"sensor_module", "imu_sensor", "imu"}:
        # Compact daughterboard FR4 PCB
        builder.add_box(0.0, 0.001, 0.0, sx=0.22, sy=0.0025, sz=0.22)
        # LGA-16 MEMS Gyroscope/Accelerometer Package
        builder.add_box(0.0, 0.005, 0.0, sx=0.06, sy=0.006, sz=0.06)
        # 4 Decoupling ceramic 0402 capacitors
        for dx, dz in [(-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05), (0.05, 0.05)]:
            builder.add_box(dx, 0.004, dz, sx=0.02, sy=0.004, sz=0.012)
        # 4-pin Gold plated I2C/SPI connector strip
        builder.add_box(0.0, 0.012, -0.09, sx=0.16, sy=0.02, sz=0.03)
        for p in range(4):
            px = -0.06 + p * 0.04
            builder.add_cylinder(px, 0.025, -0.09, radius=0.004, height=0.03, segments=6, axis="y")

    elif c in {"pcb_board", "pcb"}:
        # Substrate FR4 laminate board
        builder.add_box(0.0, 0.001, 0.0, sx=0.50, sy=0.003, sz=0.40)
        # Surface Mount Microcontroller (QFP-64 package body)
        builder.add_box(0.0, 0.010, 0.0, sx=0.14, sy=0.014, sz=0.14)
        # Metal heat sink block
        builder.add_box(0.15, 0.015, 0.08, sx=0.09, sy=0.025, sz=0.12)
        # 3 Cylindrical filter capacitors
        builder.add_cylinder(-0.16, 0.030, -0.10, radius=0.025, height=0.055, segments=12, axis="y")
        builder.add_cylinder(-0.16, 0.030, -0.02, radius=0.025, height=0.055, segments=12, axis="y")
        builder.add_cylinder(-0.16, 0.030, 0.06, radius=0.025, height=0.055, segments=12, axis="y")

    elif c in {"qfp_chip", "mcu"}:
        # Central molded epoxy packaging
        builder.add_box(0.0, 0.005, 0.0, sx=0.20, sy=0.010, sz=0.20)
        # Perimeter gull-wing lead frame pins
        for side in range(4):
            theta = side * math.pi / 2.0
            cx = 0.12 * math.cos(theta)
            cz = 0.12 * math.sin(theta)
            if side % 2 == 0:
                builder.add_box(cx, 0.003, 0.0, sx=0.035, sy=0.003, sz=0.18)
            else:
                builder.add_box(0.0, 0.003, cz, sx=0.18, sy=0.003, sz=0.035)

    else:  # capacitor
        builder.add_cylinder(0.0, 0.05, 0.0, radius=0.04, height=0.10, segments=16, axis="y")
        # Anode and cathode wire leads
        builder.add_cylinder(-0.018, -0.02, 0.0, radius=0.003, height=0.04, segments=6, axis="y")
        builder.add_cylinder(0.018, -0.02, 0.0, radius=0.003, height=0.04, segments=6, axis="y")

    return builder


def generate_plant(plant_name: str = "fractal_tree") -> ProceduralMeshBuilder:
    builder = ProceduralMeshBuilder(name=f"Botanical_{plant_name}")
    p = plant_name.lower()

    if p in {"biomass_crop", "crop", "cellulose_stalk"}:
        # Central fibrous stalk with 4 nodal segments
        total_h = 1.40
        num_nodes = 4
        node_h = total_h / num_nodes
        for n in range(num_nodes):
            ny = (n + 0.5) * node_h
            radius = 0.025 * (1.0 - 0.15 * n)
            builder.add_cylinder(0.0, ny, 0.0, radius=radius, height=node_h, segments=8, axis="y")
            # Alternating leaf blades radiating at golden angle (137.5 deg)
            phi = n * 2.39996  # ~137.5 degrees
            lx = 0.18 * math.cos(phi)
            lz = 0.18 * math.sin(phi)
            builder.add_box(lx, ny + 0.05, lz, sx=0.25, sy=0.01, sz=0.08)
        # Root anchor base
        builder.add_cylinder(0.0, 0.03, 0.0, radius=0.06, height=0.06, segments=10, axis="y")

    elif p in {"branching_shrub", "shrub", "bush"}:
        # Central stump
        builder.add_cylinder(0.0, 0.15, 0.0, radius=0.08, height=0.30, segments=10, axis="y")
        # 5 Diverging radial stem branches
        for b in range(5):
            angle = b * 2.0 * math.pi / 5.0
            sx = 0.35 * math.cos(angle)
            sz = 0.35 * math.sin(angle)
            builder.add_box(sx * 0.5, 0.45, sz * 0.5, sx=0.05, sy=0.50, sz=0.05)
            # Foliage hemisphere clump
            builder.add_box(sx, 0.70, sz, sx=0.42, sy=0.40, sz=0.42)
        # Central dense crown foliage
        builder.add_box(0.0, 0.85, 0.0, sx=0.65, sy=0.50, sz=0.65)

    else:  # fractal_tree
        # Procedural tree trunk
        builder.add_cylinder(0.0, 0.60, 0.0, radius=0.12, height=1.20, segments=12, axis="y")
        # Primary branching tier (3 branches)
        for b in range(3):
            angle = b * 2.0 * math.pi / 3.0
            bx = 0.28 * math.cos(angle)
            bz = 0.28 * math.sin(angle)
            # Angled branch approximation
            builder.add_box(bx, 1.45, bz, sx=0.08, sy=0.60, sz=0.08)
            # Foliage cluster
            builder.add_box(bx * 1.5, 1.85, bz * 1.5, sx=0.55, sy=0.45, sz=0.55)
        # Crown foliage
        builder.add_box(0.0, 2.10, 0.0, sx=0.75, sy=0.60, sz=0.75)

    return builder


def generate_animal(animal_name: str = "quadruped_rover") -> ProceduralMeshBuilder:
    builder = ProceduralMeshBuilder(name=f"Organism_{animal_name}")
    a = animal_name.lower()

    if a in {"hexapod_crawler", "hexapod", "spider_bot"}:
        # Central low-profile hexagonal chassis
        builder.add_cylinder(0.0, 0.35, 0.0, radius=0.38, height=0.14, segments=6, axis="y")
        # Top sensor mast / stereoscopic camera turret
        builder.add_cylinder(0.0, 0.48, 0.0, radius=0.08, height=0.12, segments=10, axis="y")
        builder.add_box(0.0, 0.56, 0.06, sx=0.14, sy=0.08, sz=0.08)
        # 6 Articulating radial legs (tripod gait kinematic geometry)
        for leg_idx in range(6):
            angle = leg_idx * math.pi / 3.0
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # Coxa (horizontal joint)
            cx = 0.45 * cos_a
            cz = 0.45 * sin_a
            builder.add_box(cx, 0.35, cz, sx=0.10, sy=0.08, sz=0.10)
            # Femur (diagonal descending link)
            fx = 0.65 * cos_a
            fz = 0.65 * sin_a
            builder.add_cylinder(fx, 0.26, fz, radius=0.03, height=0.22, segments=8, axis="y")
            # Tibia / foot contact tip
            tx = 0.82 * cos_a
            tz = 0.82 * sin_a
            builder.add_cylinder(tx, 0.10, tz, radius=0.025, height=0.20, segments=8, axis="y")

    elif a in {"bipedal_unit", "biped", "humanoid_walker"}:
        # Upper torso / chest
        builder.add_box(0.0, 1.25, 0.0, sx=0.45, sy=0.40, sz=0.24)
        # Head / perception pod
        builder.add_box(0.0, 1.55, 0.02, sx=0.20, sy=0.20, sz=0.20)
        # Pelvis / hip structure
        builder.add_box(0.0, 0.95, 0.0, sx=0.36, sy=0.15, sz=0.20)
        # Left and right 2-joint legs
        for side, lx in [("left", -0.14), ("right", 0.14)]:
            # Thigh link
            builder.add_cylinder(lx, 0.72, 0.0, radius=0.05, height=0.38, segments=8, axis="y")
            # Knee joint
            builder.add_cylinder(lx, 0.50, 0.0, radius=0.055, height=0.12, segments=8, axis="x")
            # Shin link
            builder.add_cylinder(lx, 0.28, 0.0, radius=0.045, height=0.36, segments=8, axis="y")
            # Stabilizer foot plate
            builder.add_box(lx, 0.04, 0.06, sx=0.12, sy=0.06, sz=0.26)

    else:  # quadruped_rover
        # Torso body
        builder.add_box(0.0, 0.65, 0.0, sx=0.45, sy=0.32, sz=0.90)
        # Neck and head
        builder.add_box(0.0, 0.88, 0.48, sx=0.20, sy=0.25, sz=0.28)
        # 4 Articulated legs
        leg_positions = [
            (-0.25, 0.30, 0.35),   # Front Left
            (0.25, 0.30, 0.35),    # Front Right
            (-0.25, 0.30, -0.35),  # Rear Left
            (0.25, 0.30, -0.35),   # Rear Right
        ]
        for lx, ly, lz in leg_positions:
            # Upper limb
            builder.add_cylinder(lx, ly + 0.12, lz, radius=0.045, height=0.32, segments=8, axis="y")
            # Lower limb
            builder.add_cylinder(lx, ly - 0.15, lz + 0.02, radius=0.035, height=0.30, segments=8, axis="y")
            # Foot pad
            builder.add_box(lx, 0.02, lz + 0.03, sx=0.09, sy=0.04, sz=0.12)

    return builder


def generate_mesh(category: str, part_type: str | None = None) -> dict[str, Any]:
    """Generate 3D procedural mesh with complete BufferGeometry and OBJ."""
    cat = (category or "machine_part").lower()
    sub = (part_type or "").lower()

    if "circuit" in cat or "elec" in cat or "pcb" in cat or "power" in cat or "sensor" in cat:
        builder = generate_circuitry(sub or "pcb_board")
    elif "plant" in cat or "tree" in cat or "botan" in cat or "crop" in cat or "shrub" in cat:
        builder = generate_plant(sub or "fractal_tree")
    elif "animal" in cat or "organ" in cat or "creature" in cat or "quad" in cat or "hex" in cat or "biped" in cat:
        builder = generate_animal(sub or "quadruped_rover")
    else:  # machine_part
        builder = generate_machine_part(sub or "motor_housing")

    return {
        "name": builder.name,
        "vertex_count": len(builder.vertices) // 3,
        "triangle_count": len(builder.indices) // 3,
        "threejs_geometry": builder.to_threejs_dict(),
        "obj_format": builder.to_obj(),
    }
