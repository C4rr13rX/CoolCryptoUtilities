# BrandDozer Windows CLI (Executable)

This repo can be packaged into a Windows executable that runs the BrandDozer CLI
from PowerShell with arguments.

## Build (PowerShell)

```powershell
.\scripts\build_brandozer_windows.ps1
```

Output:

- `dist\brandozer\brandozer.exe`

## Add to PATH

```powershell
.\scripts\install_brandozer_windows.ps1
```

If Codex CLI is missing and you want the installer to try winget/choco:

```powershell
.\scripts\install_brandozer_windows.ps1 -ForceInstallCodex
```

Open a new PowerShell window after installation.

## Usage

```powershell
brandozer start "Your prompt here" --team-mode solo --codex-model gpt-5.2-codex --codex-reasoning medium
brandozer runs --limit 5
```

## Notes

- Requires the Codex CLI on PATH (`codex`).
- Writes runtime data under `runtime\branddozer\...`.
- Installer attempts to locate `codex.exe` in common locations and add it to PATH; if not found, it prints guidance.
- `-ForceInstallCodex` will try `winget` or `choco` if available, then re-check PATH.
- Installer runs a self-check (`brandozer --help`) and prints a resolution hint on failure.
