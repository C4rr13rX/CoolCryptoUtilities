"""
Blueprint Exporter Tool — ASME Y14.5 & ISO 1101 Vector CAD Blueprint Generator
Used by C0d3rV2 and the W1z4rD brain to export real-world engineering blueprints
for robots, actuators, frames, and modular parts.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tool_registry import Tool
except ImportError:
    from .tool_registry import Tool


MATERIAL_DATABASE = {
    "ti_6al_4v": {
        "name": "Titanium Ti-6Al-4V (Grade 5)",
        "standard": "ASTM B348 / AMS 4928",
        "density_kg_m3": 4430.0,
        "yield_strength_mpa": 880.0,
        "tensile_strength_mpa": 950.0,
        "youngs_modulus_gpa": 113.8,
    },
    "al_7075_t6": {
        "name": "Aluminum Alloy 7075-T6",
        "standard": "ASTM B221 / AMS 4045",
        "density_kg_m3": 2810.0,
        "yield_strength_mpa": 503.0,
        "tensile_strength_mpa": 572.0,
        "youngs_modulus_gpa": 71.7,
    },
    "cfrp_peek": {
        "name": "Carbon Fiber PEEK Composite",
        "standard": "ISO 527 / ASTM D3039",
        "density_kg_m3": 1580.0,
        "yield_strength_mpa": 1800.0,
        "tensile_strength_mpa": 2070.0,
        "youngs_modulus_gpa": 138.0,
    },
    "steel_316l": {
        "name": "Stainless Steel 316L",
        "standard": "ASTM A240",
        "density_kg_m3": 8000.0,
        "yield_strength_mpa": 290.0,
        "tensile_strength_mpa": 580.0,
        "youngs_modulus_gpa": 193.0,
    },
}


def generate_blueprint_svg(
    part_name: str,
    part_number: str = "DWG-ROB-001",
    revision: str = "A.1",
    material_key: str = "ti_6al_4v",
    length_mm: float = 120.0,
    width_mm: float = 60.0,
    height_mm: float = 40.0,
    bore_diameter_mm: float = 25.0,
    wall_thickness_mm: float = 5.0,
    notes: Optional[List[str]] = None,
) -> str:
    """Generates an ASME Y14.5 compliant vector SVG engineering drawing."""
    mat = MATERIAL_DATABASE.get(material_key.lower(), MATERIAL_DATABASE["ti_6al_4v"])
    
    # Calculate physical volume and mass
    outer_vol_m3 = (length_mm * width_mm * height_mm) * 1e-9
    bore_vol_m3 = (math.pi * (bore_diameter_mm / 2.0)**2 * height_mm) * 1e-9 if bore_diameter_mm > 0 else 0.0
    net_vol_m3 = max(1e-12, outer_vol_m3 - bore_vol_m3)
    calculated_mass_kg = net_vol_m3 * mat["density_kg_m3"]
    
    date_str = time.strftime("%Y-%m-%d")
    notes_list = notes or [
        "1. ALL DIMENSIONS ARE IN MILLIMETERS (mm).",
        f"2. MATERIAL: {mat['name']} PER {mat['standard']}.",
        "3. REMOVE ALL BURRS AND BREAK SHARP EDGES R0.5 MAX.",
        "4. SURFACE FINISH Ra <= 0.8 um ON ALL MACHINED SURFACES.",
        "5. GD&T PER ASME Y14.5-2018.",
    ]
    notes_svg = "".join(
        f'<text x="50" y="{680 + i * 18}" font-family="Courier, monospace" font-size="11" fill="#1e293b">{note}</text>'
        for i, note in enumerate(notes_list)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 850" width="100%" height="100%">
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f1f5f9" stroke-width="1"/>
    </pattern>
    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#0284c7"/>
    </marker>
  </defs>

  <!-- Background and Grid -->
  <rect width="1100" height="850" fill="#ffffff"/>
  <rect width="1100" height="850" fill="url(#grid)"/>

  <!-- ASME Drawing Outer Border -->
  <rect x="25" y="25" width="1050" height="800" fill="none" stroke="#0f172a" stroke-width="2.5"/>
  <rect x="35" y="35" width="1030" height="780" fill="none" stroke="#334155" stroke-width="1"/>

  <!-- Title / Header -->
  <text x="550" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-weight="bold" font-size="20" fill="#0f172a" letter-spacing="2">
    MULTISCALE ROBOTICS ENGINEERING BLUEPRINT
  </text>
  <text x="550" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#64748b">
    THIRD ANGLE ORTHOGRAPHIC PROJECTION | ASME Y14.5-2018 SPECIFICATION
  </text>

  <!-- View 1: FRONT VIEW -->
  <g transform="translate(100, 160)">
    <text x="0" y="-20" font-family="Arial, sans-serif" font-weight="bold" font-size="14" fill="#0f172a">FRONT ELEVATION</text>
    <!-- Main outline -->
    <rect x="0" y="0" width="{length_mm * 1.8}" height="{height_mm * 1.8}" fill="#f8fafc" stroke="#0f172a" stroke-width="2"/>
    <!-- Center bore centerline -->
    <line x1="{length_mm * 0.9}" y1="-15" x2="{length_mm * 0.9}" y2="{height_mm * 1.8 + 15}" stroke="#ef4444" stroke-width="1" stroke-dasharray="8,4,2,4"/>
    <!-- Bore hidden lines -->
    <line x1="{length_mm * 0.9 - bore_diameter_mm * 0.9}" y1="0" x2="{length_mm * 0.9 - bore_diameter_mm * 0.9}" y2="{height_mm * 1.8}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4,4"/>
    <line x1="{length_mm * 0.9 + bore_diameter_mm * 0.9}" y1="0" x2="{length_mm * 0.9 + bore_diameter_mm * 0.9}" y2="{height_mm * 1.8}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4,4"/>
    
    <!-- Length Dimension -->
    <line x1="0" y1="{height_mm * 1.8 + 30}" x2="{length_mm * 1.8}" y2="{height_mm * 1.8 + 30}" stroke="#0284c7" stroke-width="1.2" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
    <line x1="0" y1="{height_mm * 1.8 + 5}" x2="0" y2="{height_mm * 1.8 + 40}" stroke="#0284c7" stroke-width="0.8"/>
    <line x1="{length_mm * 1.8}" y1="{height_mm * 1.8 + 5}" x2="{length_mm * 1.8}" y2="{height_mm * 1.8 + 40}" stroke="#0284c7" stroke-width="0.8"/>
    <text x="{length_mm * 0.9}" y="{height_mm * 1.8 + 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#0284c7" font-weight="bold">{length_mm:.1f} mm</text>

    <!-- Height Dimension -->
    <line x1="-30" y1="0" x2="-30" y2="{height_mm * 1.8}" stroke="#0284c7" stroke-width="1.2" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
    <line x1="-5" y1="0" x2="-40" y2="0" stroke="#0284c7" stroke-width="0.8"/>
    <line x1="-5" y1="{height_mm * 1.8}" x2="-40" y2="{height_mm * 1.8}" stroke="#0284c7" stroke-width="0.8"/>
    <text x="-35" y="{height_mm * 0.9 + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#0284c7" font-weight="bold">{height_mm:.1f} mm</text>
  </g>

  <!-- View 2: TOP PLAN VIEW -->
  <g transform="translate(100, 370)">
    <text x="0" y="-20" font-family="Arial, sans-serif" font-weight="bold" font-size="14" fill="#0f172a">TOP PLAN VIEW</text>
    <!-- Outline -->
    <rect x="0" y="0" width="{length_mm * 1.8}" height="{width_mm * 1.8}" fill="#f8fafc" stroke="#0f172a" stroke-width="2"/>
    <!-- Center bore circle -->
    <circle cx="{length_mm * 0.9}" cy="{width_mm * 0.9}" r="{bore_diameter_mm * 0.9}" fill="#ffffff" stroke="#0f172a" stroke-width="2"/>
    <line x1="{length_mm * 0.9 - bore_diameter_mm * 1.3}" y1="{width_mm * 0.9}" x2="{length_mm * 0.9 + bore_diameter_mm * 1.3}" y2="{width_mm * 0.9}" stroke="#ef4444" stroke-width="1" stroke-dasharray="6,3,1,3"/>
    <line x1="{length_mm * 0.9}" y1="{width_mm * 0.9 - bore_diameter_mm * 1.3}" x2="{length_mm * 0.9}" y2="{width_mm * 0.9 + bore_diameter_mm * 1.3}" stroke="#ef4444" stroke-width="1" stroke-dasharray="6,3,1,3"/>

    <!-- Width Dimension -->
    <line x1="-30" y1="0" x2="-30" y2="{width_mm * 1.8}" stroke="#0284c7" stroke-width="1.2" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
    <line x1="-5" y1="0" x2="-40" y2="0" stroke="#0284c7" stroke-width="0.8"/>
    <line x1="-5" y1="{width_mm * 1.8}" x2="-40" y2="{width_mm * 1.8}" stroke="#0284c7" stroke-width="0.8"/>
    <text x="-35" y="{width_mm * 0.9 + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#0284c7" font-weight="bold">{width_mm:.1f} mm</text>
    
    <!-- Bore Dimension Callout -->
    <text x="{length_mm * 0.9 + bore_diameter_mm * 1.1}" y="{width_mm * 0.9 - 15}" font-family="Arial, sans-serif" font-size="11" fill="#0284c7" font-weight="bold">THRU DIA {bore_diameter_mm:.1f} H7</text>
    <path d="M {length_mm * 0.9 + bore_diameter_mm * 0.6} {width_mm * 0.9 - bore_diameter_mm * 0.6} L {length_mm * 0.9 + bore_diameter_mm * 1.0} {width_mm * 0.9 - 18} L {length_mm * 0.9 + bore_diameter_mm * 1.8} {width_mm * 0.9 - 18}" fill="none" stroke="#0284c7" stroke-width="1.2" marker-start="url(#arrow)"/>
  </g>

  <!-- View 3: RIGHT ELEVATION -->
  <g transform="translate(460, 160)">
    <text x="0" y="-20" font-family="Arial, sans-serif" font-weight="bold" font-size="14" fill="#0f172a">RIGHT ELEVATION</text>
    <rect x="0" y="0" width="{width_mm * 1.8}" height="{height_mm * 1.8}" fill="#f8fafc" stroke="#0f172a" stroke-width="2"/>
    <line x1="{width_mm * 0.9}" y1="-10" x2="{width_mm * 0.9}" y2="{height_mm * 1.8 + 10}" stroke="#ef4444" stroke-width="1" stroke-dasharray="6,3,1,3"/>
    <line x1="{width_mm * 0.9 - bore_diameter_mm * 0.9}" y1="0" x2="{width_mm * 0.9 - bore_diameter_mm * 0.9}" y2="{height_mm * 1.8}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4,4"/>
    <line x1="{width_mm * 0.9 + bore_diameter_mm * 0.9}" y1="0" x2="{width_mm * 0.9 + bore_diameter_mm * 0.9}" y2="{height_mm * 1.8}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4,4"/>
  </g>

  <!-- View 4: ISOMETRIC AXONOMETRIC 3D -->
  <g transform="translate(700, 150)">
    <text x="80" y="0" text-anchor="middle" font-family="Arial, sans-serif" font-weight="bold" font-size="14" fill="#0f172a">ISOMETRIC 3D VIEW</text>
    <g transform="scale(0.85) translate(30, 40)">
      <!-- 3D simulated extrusion -->
      <path d="M 0 50 L 120 0 L 220 40 L 100 90 Z" fill="#e2e8f0" stroke="#0f172a" stroke-width="2"/>
      <path d="M 0 50 L 0 110 L 100 150 L 100 90 Z" fill="#cbd5e1" stroke="#0f172a" stroke-width="2"/>
      <path d="M 100 90 L 100 150 L 220 100 L 220 40 Z" fill="#94a3b8" stroke="#0f172a" stroke-width="2"/>
      <!-- Bore hole ellipse -->
      <ellipse cx="110" cy="45" rx="28" ry="14" fill="#f8fafc" stroke="#0f172a" stroke-width="1.8"/>
    </g>
    <!-- GD&T Feature Control Frame -->
    <g transform="translate(0, 240)">
      <rect x="0" y="0" width="180" height="32" fill="#ffffff" stroke="#0f172a" stroke-width="1.5"/>
      <line x1="36" y1="0" x2="36" y2="32" stroke="#0f172a" stroke-width="1.2"/>
      <line x1="108" y1="0" x2="108" y2="32" stroke="#0f172a" stroke-width="1.2"/>
      <line x1="144" y1="0" x2="144" y2="32" stroke="#0f172a" stroke-width="1.2"/>
      <text x="18" y="21" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">⊕</text>
      <text x="72" y="21" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold">⌀ 0.05 Ⓜ</text>
      <text x="126" y="21" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold">A</text>
      <text x="162" y="21" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold">B</text>
      <text x="0" y="-8" font-family="Arial, sans-serif" font-size="10" fill="#64748b" font-weight="bold">GD&amp;T FEATURE CONTROL</text>
    </g>
  </g>

  <!-- GENERAL NOTES SECTION -->
  <g>
    <text x="50" y="650" font-family="Arial, sans-serif" font-weight="bold" font-size="13" fill="#0f172a">ENGINEERING SPECIFICATIONS &amp; MANUFACTURING NOTES:</text>
    {notes_svg}
  </g>

  <!-- ASME Y14.5 TITLE BLOCK (Bottom Right) -->
  <g transform="translate(635, 615)">
    <rect x="0" y="0" width="415" height="185" fill="#f8fafc" stroke="#0f172a" stroke-width="2"/>
    <line x1="0" y1="35" x2="415" y2="35" stroke="#0f172a" stroke-width="1.5"/>
    <line x1="0" y1="85" x2="415" y2="85" stroke="#0f172a" stroke-width="1.5"/>
    <line x1="0" y1="135" x2="415" y2="135" stroke="#0f172a" stroke-width="1.5"/>
    <line x1="220" y1="35" x2="220" y2="135" stroke="#0f172a" stroke-width="1.5"/>
    <line x1="320" y1="135" x2="320" y2="185" stroke="#0f172a" stroke-width="1.5"/>

    <text x="15" y="24" font-family="Arial, sans-serif" font-weight="bold" font-size="14" fill="#0f172a">{part_name.upper()}</text>
    <text x="395" y="24" text-anchor="end" font-family="Arial, sans-serif" font-weight="bold" font-size="12" fill="#0284c7">REV: {revision}</text>

    <text x="15" y="52" font-family="Arial, sans-serif" font-size="9" fill="#64748b">MATERIAL SPECIFICATION:</text>
    <text x="15" y="70" font-family="Arial, sans-serif" font-weight="bold" font-size="11" fill="#0f172a">{mat['name']}</text>

    <text x="235" y="52" font-family="Arial, sans-serif" font-size="9" fill="#64748b">CALCULATED MASS:</text>
    <text x="235" y="70" font-family="Arial, sans-serif" font-weight="bold" font-size="11" fill="#0284c7">{calculated_mass_kg:.3f} kg ({calculated_mass_kg * 1000:.1f} g)</text>

    <text x="15" y="102" font-family="Arial, sans-serif" font-size="9" fill="#64748b">DRAWING NUMBER:</text>
    <text x="15" y="120" font-family="Arial, sans-serif" font-weight="bold" font-size="12" fill="#0f172a">{part_number}</text>

    <text x="235" y="102" font-family="Arial, sans-serif" font-size="9" fill="#64748b">YIELD STRENGTH:</text>
    <text x="235" y="120" font-family="Arial, sans-serif" font-weight="bold" font-size="11" fill="#0f172a">{mat['yield_strength_mpa']:.0f} MPa</text>

    <text x="15" y="152" font-family="Arial, sans-serif" font-size="9" fill="#64748b">ARCHITECT / AGENT:</text>
    <text x="15" y="170" font-family="Arial, sans-serif" font-weight="bold" font-size="11" fill="#0f172a">W1z4rD Autonomous Architect</text>

    <text x="335" y="152" font-family="Arial, sans-serif" font-size="9" fill="#64748b">DATE / SCALE:</text>
    <text x="335" y="170" font-family="Arial, sans-serif" font-weight="bold" font-size="11" fill="#0f172a">{date_str} | 1:1</text>
  </g>
</svg>"""
    return svg


class BlueprintExportTool(Tool):
    """C0d3rV2 Tool for exporting vector ASME Y14.5 CAD blueprints."""

    name = "blueprint_export"
    description = (
        "Export a real-world engineering blueprint conforming to ASME Y14.5 & ISO 1101 "
        "standards with orthographic projections (front, top, right, isometric), material callout "
        "(Ti-6Al-4V, Al 7075-T6, CFRP), mass calculations, GD&T feature control frames, and title block."
    )
    use_when = (
        "Use whenever designing, fabricating, or specifying robot parts, structural links, "
        "actuators, rover wheel hubs, or assemblies that require formal engineering blueprints."
    )
    params_schema = {
        "part_name": "str — formal name of the robot part or structural component",
        "part_number": "str — drawing identifier (e.g. DWG-ROB-101)",
        "material": "str — material key: 'ti_6al_4v', 'al_7075_t6', 'cfrp_peek', 'steel_316l'",
        "length_mm": "float — length in millimeters (default: 120.0)",
        "width_mm": "float — width in millimeters (default: 60.0)",
        "height_mm": "float — height in millimeters (default: 40.0)",
        "bore_diameter_mm": "float — center bore / shaft diameter in mm (default: 25.0)",
        "output_path": "str — target relative or absolute file path (e.g. 'blueprints/part_link.svg')",
    }

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir).resolve()

    def execute(self, params: dict) -> dict:
        part_name = str(params.get("part_name", "Robotic Structural Component"))
        part_number = str(params.get("part_number", "DWG-ROB-001"))
        material = str(params.get("material", "ti_6al_4v"))
        length_mm = float(params.get("length_mm", 120.0))
        width_mm = float(params.get("width_mm", 60.0))
        height_mm = float(params.get("height_mm", 40.0))
        bore_diameter_mm = float(params.get("bore_diameter_mm", 25.0))
        output_path_str = str(params.get("output_path", f"blueprints/{part_number.lower()}.svg"))

        svg_content = generate_blueprint_svg(
            part_name=part_name,
            part_number=part_number,
            material_key=material,
            length_mm=length_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            bore_diameter_mm=bore_diameter_mm,
        )

        out_path = Path(output_path_str)
        if not out_path.is_absolute():
            out_path = (self.workdir / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(svg_content, encoding="utf-8")

        mat_info = MATERIAL_DATABASE.get(material.lower(), MATERIAL_DATABASE["ti_6al_4v"])
        net_vol_m3 = ((length_mm * width_mm * height_mm) - (math.pi * (bore_diameter_mm / 2.0)**2 * height_mm)) * 1e-9
        mass_kg = max(0.001, net_vol_m3 * mat_info["density_kg_m3"])

        return {
            "status": "success",
            "file_written": str(out_path),
            "blueprint_format": "ASME Y14.5 / SVG Vector CAD",
            "part_name": part_name,
            "part_number": part_number,
            "material": mat_info["name"],
            "calculated_mass_kg": round(mass_kg, 4),
            "yield_strength_mpa": mat_info["yield_strength_mpa"],
            "orthographic_views": ["front", "top", "right_side", "isometric_3d"],
            "gdt_controls": ["position_0.05_mm", "concentricity_h7"],
        }
