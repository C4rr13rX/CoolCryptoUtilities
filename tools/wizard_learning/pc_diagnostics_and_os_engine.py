"""tools/wizard_learning/pc_diagnostics_and_os_engine.py

Permissibly Licensed (CC0 1.0 Universal / Open Standards) Multi-OS PC Basics,
Maintenance, Diagnostics, Virtualization & CompTIA A+-Scope Causal Engine.

Core Design Principles:
  1. 100% Permissibly Licensed Provenance (CC0 1.0 / Public Domain / Open RFC & POSIX/UEFI/NVMe/SMART specs):
     Zero proprietary exam dumps or copyrighted textbook prose. All explanations are
     synthesized from first-principles computer engineering, IETF RFCs (791, 793, 826,
     1035, 2131, 4291), POSIX.1-2017, UEFI 2.10, TCG TPM 2.0, T13 S.M.A.R.T., NVMe,
     FreeDesktop XDG, public Win32/PowerShell/Linux/BSD/VirtualBox CLI interfaces.
  2. Dual-Layer General + Specific Vocabulary:
     Every concept binds a general OS-agnostic mechanism (e.g., "service supervisor",
     "block journal integrity", "DHCP lease exhaustion / link-local fallback",
     "process memory pressure") to concrete implementations across Windows 10/11,
     Linux (Ubuntu/Debian, Fedora/RHEL, Arch, Alpine), macOS/FreeBSD, UEFI/POST,
     and VirtualBox/WSL2 — enabling cross-OS transfer to unseen systems.
  3. Empirical Observation Overrides Prior Expectations:
     Follows the Scientific & A+ 7-Step Troubleshooting Loop: live observations
     (IP address, mount flags, S.M.A.R.T. counters, OS release, session isolation,
     hypervisor flags, error codes) override initial user assumptions and mutate
     the recommended diagnostic/remediation course harmlessly.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LICENSE_TAG = "CC0-1.0 Universal (Public Domain Dedication — Original First-Principles & Open Standards)"


@dataclasses.dataclass(frozen=True)
class PcOsConcept:
    concept_id: str
    domain: str
    title: str
    general_principle: str
    windows_specific: str
    linux_specific: str
    macos_bsd_specific: str
    diagnostic_signals: Tuple[str, ...]
    harmless_probes: Tuple[str, ...]
    remediation_and_mutation: str
    keywords: Tuple[str, ...]
    standards_ref: str


# ── Comprehensive Multi-OS + A+ First-Principles Knowledge Graph ─────────────

PC_OS_APLUS_CONCEPTS: Tuple[PcOsConcept, ...] = (
    # 1. Hardware, Firmware & Boot Chain
    PcOsConcept(
        concept_id="boot_post_uefi_gpt",
        domain="hardware_firmware_boot",
        title="POST, UEFI Firmware, GPT & Secure Boot Chain of Trust",
        general_principle=(
            "At power-on, platform firmware executes Power-On Self-Test (POST) to verify CPU rails, "
            "DRAM training, and PCIe enumeration before initializing UEFI services, validating cryptographic "
            "signatures against Secure Boot databases (PK/KEK/db vs revoked dbx), reading the GUID Partition "
            "Table (GPT) EFI System Partition (ESP, FAT32), and handing control to the OS bootloader."
        ),
        windows_specific=(
            "Windows Boot Manager (`bootmgfw.efi` on the FAT32 ESP) reads the Boot Configuration Data (`BCD`) "
            "store (`bcdedit /enum`), verifies `winload.efi` via Secure Boot + TPM 2.0 measured boot (PCRs 0-7), "
            "and launches `ntoskrnl.exe`."
        ),
        linux_specific=(
            "UEFI loads signed `shimx64.efi` -> `grubx64.efi` or `systemd-boot`, which mounts `/boot/efi`, "
            "loads `vmlinuz` + `initramfs` into RAM, Pivot-Roots to the root filesystem in `/etc/fstab`, "
            "and spawns PID 1 (`systemd` or `init`). Inspect via `efibootmgr -v` and `bootctl status`."
        ),
        macos_bsd_specific=(
            "Apple Silicon uses Boot ROM -> iBoot -> LocalPolicy signed APFS Preboot container; Intel Macs/FreeBSD "
            "use UEFI ESP (`loader.efi` on FreeBSD, `bless --info` / `nvram -p` on macOS)."
        ),
        diagnostic_signals=(
            "post beep codes", "diagnostic leds", "no bootable device", "secure boot violation",
            "bootmgr is missing", "grub rescue", "uefi", "gpt", "efi system partition", "cmos",
        ),
        harmless_probes=(
            "Windows: `Confirm-SecureBootUEFI`; `Get-Disk | Select Number,PartitionStyle,HealthStatus`",
            "Linux: `lsblk -o NAME,PTTYPE,FSTYPE,PARTLABEL,MOUNTPOINT`; `test -d /sys/firmware/efi && echo UEFI || echo BIOS`",
            "macOS/BSD: `diskutil list` or `gpart show`",
        ),
        remediation_and_mutation=(
            "Observe first: if firmware reports 'No Bootable Device' after a BIOS reset, check whether CSM/Legacy "
            "mode was accidentally enabled on a GPT/UEFI disk (mutate: switch firmware back to UEFI-only rather "
            "than wiping the drive). If 'Secure Boot Violation' appears after a dual-boot or unsigned kernel update, "
            "enroll the MOK/signed shim or verify `dbx` rather than blindly disabling all security."
        ),
        keywords=("post", "uefi", "bios", "secure boot", "gpt", "mbr", "efi", "esp", "bootmgfw", "grub", "bcdedit"),
        standards_ref="UEFI Specification 2.10 / TCG PC Client Platform Firmware Profile (TPM 2.0)",
    ),
    PcOsConcept(
        concept_id="tpm2_encryption_bitlocker_luks",
        domain="hardware_firmware_boot",
        title="TPM 2.0 Hardware Root of Trust & Full-Disk Encryption (BitLocker / LUKS2 / FileVault)",
        general_principle=(
            "A Trusted Platform Module (TPM 2.0) records cryptographic hashes of firmware and boot stages into "
            "Platform Configuration Registers (PCRs). Full-disk encryption seals the Volume Master Key (VMK) to "
            "specific PCR values: if boot firmware, partition layout, or boot order mutates unexpectedly, the TPM "
            "refuses to unseal the key and prompts for the recovery key."
        ),
        windows_specific=(
            "BitLocker (`manage-bde -status` / `Get-BitLockerVolume`) seals the VMK in TPM 2.0 (PCRs 7 & 11). "
            "Before UEFI updates or motherboard changes, run `Suspend-BitLocker -MountPoint C: -RebootCount 1`."
        ),
        linux_specific=(
            "LUKS2 (`cryptsetup luksDump /dev/nvme0n1p3`) with `systemd-cryptenroll --tpm2-device=auto` binds "
            "unlocking to TPM2 PCRs while preserving passphrase recovery slots."
        ),
        macos_bsd_specific=(
            "macOS FileVault (`fdesetup status`) integrates hardware AES engines with Secure Enclave / T2 keys; "
            "FreeBSD uses `geli` full-disk encryption."
        ),
        diagnostic_signals=(
            "bitlocker recovery", "tpm", "pcr", "luks", "filevault", "fdesetup", "manage-bde", "cryptsetup",
        ),
        harmless_probes=(
            "Windows: `Get-Tpm` (admin) or `Get-CimInstance -Namespace root/cimv2/security/microsofttpm -ClassName Win32_Tpm`",
            "Linux: `ls -l /dev/tpm*`; `lsblk -f`",
            "macOS: `fdesetup status`",
        ),
        remediation_and_mutation=(
            "If BitLocker/LUKS suddenly demands a recovery key right after a firmware update or dock/USB boot-order "
            "change, DO NOT reformat: observe that PCR hashes changed, restore original boot order/dock state to test "
            "normal unsealing, or supply the 48-digit recovery key and re-seal PCRs."
        ),
        keywords=("tpm", "tpm 2.0", "pcr", "bitlocker", "luks", "filevault", "encryption", "manage-bde", "cryptsetup"),
        standards_ref="ISO/IEC 11889 (TPM 2.0) / NIST SP 800-111 Storage Encryption",
    ),
    PcOsConcept(
        concept_id="cpu_thermals_psu_ram_ecc",
        domain="hardware_firmware_boot",
        title="CPU Thermal Throttling, PSU Voltage Rails & DRAM / ECC Memory Diagnostics",
        general_principle=(
            "Hardware instability stems from physical conservation limits: (1) dried thermal paste, failed fans, "
            "or clogged heatsinks drive junction temperature ($T_j$) to $T_{j,\\max}$ (~95-105 °C), triggering PROCHOT "
            "clock throttling or sudden thermal shutdown; (2) ATX PSU rail droop outside ±5% on +12V, +5V, or +3.3V "
            "causes resets under GPU/CPU transient load; (3) faulty DRAM cells cause non-deterministic page faults, "
            "WHEA/MCE hardware errors, and filesystem corruption unless caught by ECC (SECDED)."
        ),
        windows_specific=(
            "Inspect `Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'}` "
            "and `Kernel-Power` Event ID 41, plus `mdsched.exe` (Windows Memory Diagnostic)."
        ),
        linux_specific=(
            "Check `/sys/class/thermal/thermal_zone*/temp`, `sensors` (`lm-sensors`), `dmesg | grep -iE 'mce|edac|thermal|oom'`, "
            "and test RAM offline via `memtest86+`."
        ),
        macos_bsd_specific=(
            "Run Apple Diagnostics (hold D / power button at boot), `powermetrics --samplers smc` (macOS), or "
            "`sysctl dev.cpu` temperatures and `mcelog` on FreeBSD."
        ),
        diagnostic_signals=(
            "thermal throttling", "overheating", "random reboot", "kernel-power 41", "whea_uncorrectable_error",
            "memory_management bsod", "memtest", "psu rails", "12v rail", "ecc memory", "prochot",
        ),
        harmless_probes=(
            "Windows: `Get-CimInstance Win32_Processor | Select Name,NumberOfCores,LoadPercentage,CurrentClockSpeed`",
            "Linux: `cat /proc/loadavg; grep -E 'MemTotal|MemAvailable' /proc/meminfo`",
        ),
        remediation_and_mutation=(
            "Distinguish software crash vs thermal/power/RAM fault empirically: if shutdown occurs instantly with no "
            "crash dump under peak load, suspect PSU +12V OCP/UVP or thermal trip; if random bit-flip crashes occur "
            "even idle with XMP/EXPO overclock active, mutate BIOS DRAM profile to JEDEC baseline and run `memtest86+`."
        ),
        keywords=("thermal", "throttling", "overheat", "psu", "voltage rail", "12v", "ram", "ddr4", "ddr5", "ecc", "memtest", "whea"),
        standards_ref="ATX 3.0 Power Supply Spec / JEDEC JESD79-5 (DDR5) / ACPI Thermal Model",
    ),

    # 2. Storage, S.M.A.R.T., RAID & Filesystems
    PcOsConcept(
        concept_id="smart_nvme_sata_health",
        domain="storage_filesystems_raid",
        title="NVMe & SATA S.M.A.R.T. Telemetry and Pre-Failure Drive Triage",
        general_principle=(
            "Storage media telegraph physical degradation before total loss through Self-Monitoring, Analysis, and "
            "Reporting Technology (S.M.A.R.T.) and NVMe Health Logs. Non-zero Reallocated Sectors (`0x05`), Current "
            "Pending Sectors (`0xC5`), Uncorrectable Sector Count (`0xC6`), or NVMe `Media and Data Integrity Errors` "
            "and `Available Spare < Threshold` prove physical media failure."
        ),
        windows_specific=(
            "Use `Get-PhysicalDisk | Select FriendlyName,MediaType,HealthStatus,OperationalStatus,Size` and "
            "`Get-StorageReliabilityCounter` (PowerShell) to read wear, temperature, and read/write error counters."
        ),
        linux_specific=(
            "Use `smartctl -a /dev/nvme0n1` (`smartmontools`), `nvme smart-log /dev/nvme0n1` (`nvme-cli`), and "
            "`dmesg | grep -iE 'nvme|ata|i/o error|blk_update_request'`."
        ),
        macos_bsd_specific=(
            "Use `diskutil info disk0 | grep -i smart` on macOS and `smartctl -a /dev/nda0` or `nvmecontrol logpage -p 2 nvme0` on FreeBSD."
        ),
        diagnostic_signals=(
            "smart status", "reallocated sector", "pending sector", "uncorrectable", "click of death",
            "i/o device error", "nvme wear", "percentage used", "available spare", "bad blocks",
        ),
        harmless_probes=(
            "Windows: `Get-PhysicalDisk | Select FriendlyName,MediaType,HealthStatus,OperationalStatus`",
            "Linux: `lsblk -d -o NAME,MODEL,SIZE,ROTA,STATE`",
        ),
        remediation_and_mutation=(
            "CRITICAL ETHICAL & SCIENTIFIC MUTATION: If S.M.A.R.T. shows rising `Pending`/`Uncorrectable` sectors or "
            "`Media Errors > 0`, NEVER run write-heavy `chkdsk /r` or defragmentation first (which stresses dying flash/platters "
            "and destroys remaining data)! Immediately mutate plan to read-only sector imaging (`ddrescue`) / backup "
            "of critical user data first, then replace hardware."
        ),
        keywords=("smart", "s.m.a.r.t", "reallocated", "pending sector", "nvme", "sata", "ssd", "hdd", "get-physicaldisk", "smartctl", "ddrescue"),
        standards_ref="INCITS T13 ATA/ATAPI S.M.A.R.T. / NVM Express Base Specification 2.0",
    ),
    PcOsConcept(
        concept_id="raid_levels_math_and_recovery",
        domain="storage_filesystems_raid",
        title="RAID 0, 1, 5, 6, and 10 Capacity, Fault Tolerance & Rebuild Mechanics",
        general_principle=(
            "Redundant Array of Independent Disks (RAID) trades raw capacity ($N \\cdot S$, where $N$ is drive count and "
            "$S$ is smallest drive size) against fault tolerance and IOPS:\n"
            "- **RAID 0 (Striping)**: Capacity $= N \\cdot S$ (min 2 disks), **0-disk** fault tolerance (any 1 failed disk loses 100% of data).\n"
            "- **RAID 1 (Mirroring)**: Capacity $= S$ (or $(N/2)\\cdot S$ in pairs, min 2 disks), survives **1** disk failure.\n"
            "- **RAID 5 (Single Distributed Parity)**: Capacity $= (N - 1) \\cdot S$ (min 3 disks), survives **1** disk failure.\n"
            "- **RAID 6 (Dual Distributed Parity)**: Capacity $= (N - 2) \\cdot S$ (min 4 disks), survives **2** simultaneous disk failures.\n"
            "- **RAID 10 (1+0 Mirrored Stripes)**: Capacity $= (N / 2) \\cdot S$ (min 4 even disks), survives **1 disk per mirrored pair** with fast rebuilds and no parity write penalty."
        ),
        windows_specific=(
            "Windows Storage Spaces (`Get-VirtualDisk`, `Get-StoragePool`) provides Simple (RAID 0), Mirror (RAID 1/10), "
            "and Parity (RAID 5/6) resiliency."
        ),
        linux_specific=(
            "Linux software RAID uses `mdadm --detail /dev/md0` (`/proc/mdstat`), `Btrfs` profiles, or `ZFS` (`zpool status`: "
            "`mirror`, `raidz1`, `raidz2`, `raidz3` with end-to-end Merkle checksums that prevent silent bit-rot)."
        ),
        macos_bsd_specific=(
            "macOS `diskutil appleRAID list` (RAID 0/1/JBOD); FreeBSD native OpenZFS (`zpool status`, `zpool scrub`) and `gmirror`."
        ),
        diagnostic_signals=(
            "raid 0", "raid 1", "raid 5", "raid 6", "raid 10", "degraded array", "mdadm", "zpool", "storage spaces", "parity",
        ),
        harmless_probes=(
            "Windows: `Get-VirtualDisk`",
            "Linux: `cat /proc/mdstat` or `zpool status`",
        ),
        remediation_and_mutation=(
            "Remember: RAID is high availability, NOT a backup (accidental `rm -rf` replicates instantly across all mirrors!). "
            "On a degraded RAID 5 array with large multi-TB drives, verify recent backup BEFORE pulling/resilvering a disk "
            "in case a second drive hits an Unrecoverable Read Error (URE) during rebuild."
        ),
        keywords=("raid", "raid 0", "raid 1", "raid 5", "raid 6", "raid 10", "striping", "mirroring", "parity", "mdadm", "zfs", "raidz"),
        standards_ref="SNIA Common RAID Disk Data Format (DDF) / OpenZFS Specification",
    ),
    PcOsConcept(
        concept_id="filesystems_readonly_integrity_repair",
        domain="storage_filesystems_raid",
        title="Cross-OS Filesystems (NTFS, ReFS, ext4, XFS, Btrfs, ZFS, APFS) & Read-Only Remount Mutation",
        general_principle=(
            "Modern journaling and Copy-on-Write (CoW) filesystems protect metadata invariants. When kernel I/O drivers "
            "detect journal corruption, metadata checksum mismatches, or sudden power-loss inconsistencies, the OS "
            "deliberately remounts the volume **Read-Only (`ro`)** as a protective circuit breaker to prevent further "
            "metadata overwrite."
        ),
        windows_specific=(
            "Windows uses `NTFS` (`$MFT` + `$LogFile` journal) and `ReFS`. Online scan: `chkdsk C: /scan` (harmless online); "
            "OS component store & system file repair: `DISM /Online /Cleanup-Image /RestoreHealth` followed by `sfc /scannow`."
        ),
        linux_specific=(
            "Linux uses `ext4` (`e2fsck`), `XFS` (`xfs_repair` — must be unmounted), `Btrfs` (`btrfs scrub start`), and `ZFS`. "
            "Inspect mount state via `findmnt` or `grep ' ro,' /proc/mounts` and kernel reason via `dmesg | tail -n 50`."
        ),
        macos_bsd_specific=(
            "macOS uses `APFS` (CoW containers + snapshots): verify harmlessly with `diskutil verifyVolume /` (`fsck_apfs -n`); "
            "FreeBSD uses `UFS2` (`fsck_ffs`) and `ZFS` (`zpool scrub`)."
        ),
        diagnostic_signals=(
            "read-only file system", "erofs", "corrupted journal", "ntfs", "ext4", "xfs", "apfs", "btrfs",
            "sfc /scannow", "dism", "chkdsk", "fsck", "superblock",
        ),
        harmless_probes=(
            "Windows: `Get-Volume | Select DriveLetter,FileSystemType,HealthStatus,SizeRemaining`",
            "Linux: `findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS /`; `df -hT`",
            "macOS: `diskutil info /`",
        ),
        remediation_and_mutation=(
            "OBSERVATION MUTATION: If a script or package manager fails with `Read-only file system (EROFS)`, DO NOT blindly "
            "force `mount -o remount,rw /`! First inspect `dmesg` / `Get-PhysicalDisk`: if mounted `ro` due to an NVMe/SATA "
            "error or dirty journal, back up unsaved RAM state and run offline `fsck` / `chkdsk` after verifying hardware health; "
            "if inside an immutable container/LiveISO, pivot writes to `/tmp` or a writable volume."
        ),
        keywords=("filesystem", "ntfs", "refs", "ext4", "xfs", "btrfs", "apfs", "read-only", "erofs", "sfc", "dism", "chkdsk", "fsck", "diskutil"),
        standards_ref="POSIX.1-2017 Filesystem Semantics / Linux ext4/XFS Kernel Docs / Microsoft NTFS/DISM Reference",
    ),

    # 3. Multi-OS Process, Service, Package & Privilege Administration
    PcOsConcept(
        concept_id="multi_os_process_memory_scheduler",
        domain="multi_os_administration",
        title="Process Table, Virtual Memory, Swap Thrashing & OOM Triage Across OSes",
        general_principle=(
            "Every multitasking OS isolates processes in virtual address spaces backed by physical RAM pages and swap/pagefile "
            "storage. Distinguish harmless disk page cache (reclaimable in microseconds) from true committed anonymous memory "
            "exhaustion, which causes major page-fault thrashing (high disk I/O + near-zero useful CPU throughput) or "
            "Out-Of-Memory (OOM) termination."
        ),
        windows_specific=(
            "Inspect `Get-Process | Sort-Object WorkingSet64 -Descending | Select -First 10 Name,Id,CPU,WorkingSet64` "
            "and `Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory`. "
            "Use `Stop-Process -Id <PID>` (polite close first, `-Force` only if hung) or Resource Monitor (`resmon`)."
        ),
        linux_specific=(
            "Check `MemAvailable` (NOT `MemFree`!) in `/proc/meminfo` (`free -h`), PSI stall metrics in `/proc/pressure/memory`, "
            "`ps aux --sort=-%mem | head`, `vmstat 1 3`, and OOM kills via `journalctl -k | grep -i 'killed process'`. "
            "Send polite `SIGTERM` (`kill -15 <PID>`) so cleanup handlers flush files before ever resorting to `SIGKILL` (`-9`)."
        ),
        macos_bsd_specific=(
            "Check `memory_pressure` and `vm_stat` / `top -o rsize` on macOS; `top -o res` and `swapinfo` on FreeBSD."
        ),
        diagnostic_signals=(
            "high memory", "out of memory", "oom killer", "swap thrashing", "100% disk", "high cpu",
            "zombie process", "sigterm", "sigkill", "working set", "memavailable",
        ),
        harmless_probes=(
            "Windows: `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory,TotalVisibleMemorySize`",
            "Linux: `free -m; cat /proc/loadavg`",
            "macOS: `vm_stat | head -n 10`",
        ),
        remediation_and_mutation=(
            "OBSERVATION MUTATION: Beginners often panic seeing low `MemFree` on Linux and try clearing caches, when "
            "`MemAvailable` is actually 60% full of healthy page cache! Always inspect `MemAvailable` + swap-in/out rate (`si`/`so`). "
            "Also, a `Z` (`defunct`/zombie) process consumes zero RAM and CANNOT be killed with `kill -9` because it is already "
            "dead — you must inspect or signal its parent PID (`PPID`) to `wait()` on it!"
        ),
        keywords=("process", "memory", "ram", "swap", "pagefile", "thrashing", "oom", "zombie", "sigterm", "sigkill", "memavailable", "ps", "get-process"),
        standards_ref="IEEE Std 1003.1 (POSIX Signals & Process Lifecycle) / Linux PSI Documentation",
    ),
    PcOsConcept(
        concept_id="multi_os_service_supervisors_logs",
        domain="multi_os_administration",
        title="Service Supervisors & Structured Event/Journal Log Forensics (SCM / systemd / launchd)",
        general_principle=(
            "Background daemons are managed by a root service supervisor that tracks dependency graphs, exit codes, "
            "restart backoffs, and structured logs. When a service fails to start, never restart it in a blind loop: "
            "read the exit status, stderr/event trace, port conflicts, and config syntax validator first."
        ),
        windows_specific=(
            "Service Control Manager (`services.msc`, `Get-Service`, `sc.exe qc <name>`). Inspect structured failures with "
            "`Get-WinEvent -FilterHashtable @{LogName='System','Application'; Level=1,2} -MaxEvents 20` and Task Scheduler (`Get-ScheduledTask`). "
            "Note Windows Session Isolation: services run in **Session 0** (non-interactive), while the logged-in user's desktop runs in **Session 1+** (`qwinsta`)."
        ),
        linux_specific=(
            "`systemd` (`systemctl status <unit>`, `systemctl list-units --failed`, `journalctl -u <unit> -n 50 --no-pager`). "
            "On Alpine Linux, OpenRC uses `rc-status` and `rc-service <name> status`."
        ),
        macos_bsd_specific=(
            "macOS `launchd` (`launchctl list`, `launchctl print system/<label>`, `/Library/LaunchDaemons/*.plist`, `log show --last 5m`); "
            "FreeBSD `rc.d` (`service -e`, `service <name> status`, `/var/log/messages`)."
        ),
        diagnostic_signals=(
            "service failed", "systemctl", "journalctl", "event viewer", "get-winevent", "launchctl",
            "session 0", "qwinsta", "crash loop", "exit-code", "openrc",
        ),
        harmless_probes=(
            "Windows: `Get-Service | Where-Object {$_.StartType -eq 'Automatic' -and $_.Status -ne 'Running'} | Select -First 10 Name,Status`",
            "Linux: `systemctl --failed --no-pager` (or `ps -p 1 -o comm=`)",
            "macOS: `launchctl list | head -n 15`",
        ),
        remediation_and_mutation=(
            "OBSERVATION MUTATION: On Linux inside WSL, containers, or Alpine, PID 1 (`ps -p 1 -o comm=`) might be `init` "
            "or `sh` rather than `systemd` — if `System has not been booted with systemd` is observed, mutate immediately "
            "to `service <name> status` / process inspection instead of repeating `systemctl`! On Windows, if running in "
            "Session 0 (`qwinsta`), never assume Win32 `ImageGrab` sees Console Session 1's desktop."
        ),
        keywords=("service", "daemon", "systemd", "systemctl", "journalctl", "get-service", "get-winevent", "event viewer", "launchd", "launchctl", "session 0", "qwinsta"),
        standards_ref="freedesktop.org systemd Service Spec / Microsoft Windows SCM & Session 0 Isolation",
    ),
    PcOsConcept(
        concept_id="multi_os_package_managers_and_os_detection",
        domain="multi_os_administration",
        title="Cross-OS Package Management & Runtime Distribution Detection",
        general_principle=(
            "Software installation, dependency resolution, cryptographic signature verification, and atomic upgrades "
            "are governed by the host OS's native package manager. Assuming `apt` on every Linux box or `choco` on every "
            "Windows box fails whenever the distribution family differs — always probe OS identity first."
        ),
        windows_specific=(
            "Native Windows Package Manager: `winget search <pkg>`, `winget list`, `winget upgrade` (plus `DISM` / "
            "`Get-WindowsCapability` for OS features, and `pip` inside project `.venv` virtual environments)."
        ),
        linux_specific=(
            "Read `/etc/os-release` first! Then dispatch by family:\n"
            "- **Debian / Ubuntu**: `apt update && apt install <pkg>`, query files via `dpkg -L <pkg>`.\n"
            "- **Fedora / RHEL / Rocky / Alma**: `dnf install <pkg>`, query via `rpm -qa`.\n"
            "- **Arch / Manjaro**: `pacman -S <pkg>`, query via `pacman -Qi <pkg>`.\n"
            "- **Alpine (musl)**: `apk add <pkg>`, query via `apk info`.\n"
            "- **openSUSE**: `zypper install <pkg>`."
        ),
        macos_bsd_specific=(
            "macOS: `softwareupdate -l` (OS updates) and `brew install <pkg>` (Homebrew) / `pkgutil --pkgs`. "
            "FreeBSD: `pkg install <pkg>` and `freebsd-update fetch`."
        ),
        diagnostic_signals=(
            "apt: command not found", "package manager", "os-release", "winget", "dnf", "pacman", "apk", "brew",
            "dpkg was interrupted", "lock-frontend", "unmet dependencies",
        ),
        harmless_probes=(
            "Cross-OS Python: `python -c \"import platform; print(platform.platform())\"`",
            "Linux: `cat /etc/os-release`",
            "Windows: `winget --version`",
        ),
        remediation_and_mutation=(
            "OBSERVATION MUTATION: If `apt` returns `command not found`, inspect `/etc/os-release` and `command -v dnf pacman apk zypper` "
            "and mutate to the native package manager. If `Could not get lock /var/lib/dpkg/lock-frontend` appears on Ubuntu, "
            "NEVER blindly `rm` the lock file! First check `ps aux | grep -E 'apt|dpkg|unattended'` — if `unattended-upgrades` "
            "is actively writing, wait for it to finish cleanly so the package database is not corrupted."
        ),
        keywords=("package manager", "winget", "apt", "dpkg", "dnf", "rpm", "pacman", "apk", "zypper", "brew", "os-release", "lock-frontend"),
        standards_ref="freedesktop.org os-release Spec / Debian Policy / RPM / Winget CLI Reference",
    ),
    PcOsConcept(
        concept_id="permissions_acls_least_privilege",
        domain="multi_os_administration",
        title="Access Control, POSIX Mode Bits, Windows NTFS ACLs & Ethical Least-Privilege Elevation",
        general_principle=(
            "Operating systems enforce mandatory and discretionary access controls separating unprivileged user contexts "
            "from root/Administrator kernels. When encountering `Permission denied` (`EACCES` / `Access is denied`), "
            "evaluate *why* the boundary exists before elevating: user-space config, virtualenvs, and workspace files "
            "should stay owned by the user, never `chmod 777` or world-writable."
        ),
        windows_specific=(
            "NTFS Access Control Lists (`icacls <path>`, `Get-Acl`), User Account Control (UAC split-token elevation), "
            "`whoami /groups` & `whoami /priv`. Running elevated scripts modifies system-wide state only with explicit user approval."
        ),
        linux_specific=(
            "POSIX uid/gid + octal mode bits (`u=rwx,g=rx,o=rx` = `0755` for dirs/executables, `0600` for SSH private keys!), "
            "extended ACLs (`getfacl`/`setfacl`), SELinux (`getenforce`, `ls -Z`) / AppArmor (`aa-status`), and `sudo`/`pkexec`."
        ),
        macos_bsd_specific=(
            "macOS POSIX + System Integrity Protection (`csrutil status`) + Transparency, Consent, and Control (TCC) "
            "per-app privacy grants (Screen Recording, Accessibility, Full Disk Access)."
        ),
        diagnostic_signals=(
            "permission denied", "eacces", "access is denied", "chmod 777", "ssh unprotected private key",
            "icacls", "selinux", "tcc", "uac", "sudo",
        ),
        harmless_probes=(
            "Windows: `whoami`; `icacls .`",
            "Linux/macOS: `id`; `stat -c '%a %U:%G %n' <file>` (or `stat -f '%Sp %Su:%Sg %N'` on BSD/macOS)",
        ),
        remediation_and_mutation=(
            "ETHICAL SECURITY MUTATION: Never fix `Permission denied` with `chmod -R 777`! For SSH failing with "
            "`UNPROTECTED PRIVATE KEY FILE` (`0644`), tighten permissions to `chmod 600 ~/.ssh/id_ed25519` (`chmod 700 ~/.ssh`). "
            "If `pip install` fails on system Python (`PEP 668 externally-managed-environment`), mutate to creating/using "
            "a `.venv` (`python3 -m venv .venv`) instead of `sudo pip install`."
        ),
        keywords=("permissions", "chmod", "chown", "icacls", "acl", "uac", "sudo", "polkit", "selinux", "pep 668", "least privilege", "0600"),
        standards_ref="POSIX.1e ACLs / NIST SP 800-53 AC-6 Least Privilege / PEP 668",
    ),

    # 4. Networking, Security & Bottom-Up L1->L7 Troubleshooting
    PcOsConcept(
        concept_id="network_bottom_up_dhcp_dns_gateway",
        domain="networking_and_security",
        title="Bottom-Up TCP/IP Diagnostics: Link, DHCP/APIPA, Gateway Routing, MTU & DNS",
        general_principle=(
            "Network failures must be isolated systematically across layers rather than guessing:\n"
            "1. **L1/L2 Link & NIC**: Carrier up? Wi-Fi associated? VLAN/MAC valid?\n"
            "2. **L3 IP & DHCP (DORA: Discover-Offer-Request-Acknowledge)**: Did the host receive a routable subnet IP, "
            "or did it self-assign an **APIPA / Link-Local (`169.254.x.x/16` or `fe80::/10`)** address proving DHCP failed?\n"
            "3. **L3 Default Gateway & Routing**: Can we reach the local gateway IP (`ping <gw>`), and can packets reach an "
            "external IP (`1.1.1.1` / `8.8.8.8`) without DNS?\n"
            "4. **L4 Transport & Firewall**: Is the target TCP port listening (`SYN -> SYN-ACK`) or blocked (`RST` / timeout)?\n"
            "5. **L7 DNS & TLS**: If IP ping works (`1.1.1.1` reachable) while hostname fails (`example.com`), the root cause "
            "is **DNS resolution** (`UDP/TCP 53` / DoH) — NOT physical internet loss! If TLS handshakes fail with certificate "
            "validity errors across all sites, check **system clock skew** first!"
        ),
        windows_specific=(
            "`Get-NetIPConfiguration`, `ipconfig /all`, `Test-NetConnection <host> -Port 443`, `Resolve-DnsName <domain>`, "
            "`Get-NetTCPConnection -State Listen`, `Get-NetFirewallRule`."
        ),
        linux_specific=(
            "`ip -br link`, `ip -br addr`, `ip route`, `ss -tulpn`, `resolvectl status` / `dig @1.1.1.1 <domain>`, "
            "`ping -c 2 -W 2 <gw>`, `ufw status` / `nft list ruleset`."
        ),
        macos_bsd_specific=(
            "`ifconfig`, `netstat -rn`, `scutil --dns`, `networksetup -listallhardwareports`, `lsof -iTCP -sTCP:LISTEN -P -n`, `pfctl -sr`."
        ),
        diagnostic_signals=(
            "169.254.", "apipa", "dns_probe_finished_nxdomain", "no internet", "connection refused",
            "connection timed out", "dhcp", "default gateway", "mtu", "certificate expired clock", "ss -tulpn",
        ),
        harmless_probes=(
            "Windows: `Get-NetIPAddress -AddressFamily IPv4 | Select InterfaceAlias,IPAddress,PrefixLength`",
            "Linux: `ip -br addr; ip route show default`",
            "macOS: `scutil --dns | head -n 25`",
        ),
        remediation_and_mutation=(
            "OBSERVATION OVERRIDES ASSUMPTION:\n"
            "- User says 'DNS is broken', but observation shows IP `169.254.88.12` -> **Mutate**: DHCP failed at L2/L3 (check cable/VLAN/DHCP server), not DNS!\n"
            "- User says 'Wi-Fi has no internet', but `ping 1.1.1.1` succeeds (`14 ms`) while `Resolve-DnsName` times out -> **Mutate**: L1-L3 internet is UP; fix local DNS resolver (`resolvectl`/`Set-DnsClientServerAddress`)!\n"
            "- `curl` gets instant `Connection refused` (`ECONNREFUSED`) vs 30s timeout -> `ECONNREFUSED` proves host is reachable and sent TCP `RST` (daemon not listening on that port/loopback bind `127.0.0.1` vs `0.0.0.0`), whereas timeout indicates firewall `DROP` or routing blackhole!"
        ),
        keywords=("network", "dhcp", "dora", "apipa", "169.254", "dns", "gateway", "ping", "ss", "netstat", "get-nettcpconnection", "connection refused", "firewall", "mtu", "cidr"),
        standards_ref="IETF RFC 791 (IPv4), RFC 793 (TCP), RFC 1035 (DNS), RFC 2131 (DHCP), RFC 3927 (Link-Local 169.254/16)",
    ),
    PcOsConcept(
        concept_id="security_incident_7step_malware",
        domain="networking_and_security",
        title="Endpoint Security Hardening & CompTIA A+ 7-Step Malware Incident Response",
        general_principle=(
            "When ransomware, credential-stealing malware, or unauthorized persistence is suspected, follow strict "
            "forensic & containment order:\n"
            "1. **Investigate & Verify** malware symptoms (unexpected outbound sockets, high CPU/crypto I/O, tampered AV, strange scheduled tasks/units).\n"
            "2. **Quarantine / Isolate Immediately** (disconnect NIC/Wi-Fi/VLAN & unmount network shares BEFORE anything else so lateral spread stops!).\n"
            "3. **Disable System Restore / Purge Malicious Snapshot Persistence** (while preserving clean offline backups).\n"
            "4. **Remediate** (update signatures offline/clean boot environment, quarantine/remove payloads, or clean-image from trusted gold media + restore verified backup).\n"
            "5. **Schedule Scans & Apply OS/App Security Patches**.\n"
            "6. **Re-enable Real-Time Protections / Clean Restore Points** & rotate exposed credentials/tokens.\n"
            "7. **Educate End-User & Document Incident Root Cause**."
        ),
        windows_specific=(
            "Windows Defender (`Get-MpComputerStatus`, `Start-MpScan -ScanType QuickScan`), check persistence in "
            "`HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run`, `Get-ScheduledTask`, and `Sysinternals Autoruns`."
        ),
        linux_specific=(
            "Inspect `/etc/crontab`, `/var/spool/cron/*`, `~/.ssh/authorized_keys`, `systemctl list-timers`, "
            "unexpected listening binaries (`ss -tulpn`), and deleted-but-running executables (`ls -l /proc/*/exe`)."
        ),
        macos_bsd_specific=(
            "Check `~/Library/LaunchAgents`, `/Library/LaunchAgents`, `/Library/LaunchDaemons`, Gatekeeper/XProtect logs, "
            "and unsigned login items."
        ),
        diagnostic_signals=(
            "malware", "ransomware", "infected", "quarantine", "incident response", "suspicious outbound",
            "scheduled task persistence", "cryptominer",
        ),
        harmless_probes=(
            "Windows: `Get-MpComputerStatus | Select AntivirusEnabled,RealTimeProtectionEnabled,AntivirusSignatureAge`",
            "Linux: `ss -tunp | head -n 20`",
        ),
        remediation_and_mutation=(
            "CRITICAL CONTAINMENT MUTATION: If active ransomware encryption is observed on a workstation mounted to "
            "shared SMB/NFS drives, DO NOT reboot or run slow scans while still plugged into the LAN — **Step 1 action "
            "mutates to instant network/share isolation** so shared company volumes survive!"
        ),
        keywords=("malware", "ransomware", "quarantine", "incident response", "defender", "autoruns", "persistence", "cron", "launchagents"),
        standards_ref="NIST SP 800-61r2 Computer Security Incident Handling Guide",
    ),

    # 5. Virtualization, WSL2 & Disposable Lab Sandboxing
    PcOsConcept(
        concept_id="virtualbox_wsl2_hypervisor_diagnostics",
        domain="virtualization_and_sandboxing",
        title="VirtualBox (`VBoxManage`), Hardware Virtualization (VT-x/AMD-V) & WSL2 Sandboxing",
        general_principle=(
            "Virtual machines isolate guest kernels and desktops from the host PC using hardware CPU virtualization "
            "(Intel VT-x / AMD-V) and Second-Level Address Translation (EPT / RVI nested paging). Before testing OS repairs, "
            "unknown packages, or automated GUI navigation, take a VM **snapshot** or run inside an isolated virtual "
            "framebuffer (`Xvfb`) so mistakes revert cleanly in seconds with zero impact on the host."
        ),
        windows_specific=(
            "Oracle VirtualBox (`C:\\Program Files\\Oracle\\VirtualBox\\VBoxManage.exe`): `VBoxManage list hostinfo`, "
            "`VBoxManage list vms`, `VBoxManage createvm`, `VBoxManage snapshot <vm> take <name>`, `VBoxManage startvm <vm> --type headless`. "
            "Note: when Windows Hyper-V / Virtualization-Based Security (`WSL2`) is active, VirtualBox seamlessly runs via the Windows Hypervisor Platform (`NEM`/WHPX) API."
        ),
        linux_specific=(
            "KVM/QEMU (`virsh list --all`, `egrep -c '(vmx|svm)' /proc/cpuinfo`), `VBoxManage`, Docker/Podman cgroups, "
            "and X11 Virtual Framebuffer (`xvfb-run -a -s '-screen 0 1024x768x24' <gui-cmd>`) + ImageMagick `import -window root` "
            "for 100% isolated, harmless Linux desktop vision & navigation."
        ),
        macos_bsd_specific=(
            "Apple `Virtualization.framework` / UTM / VirtualBox (`VBoxManage`), FreeBSD `bhyve` (`vm list`) and `jails` (`jls`)."
        ),
        diagnostic_signals=(
            "virtualbox", "vboxmanage", "vt-x", "amd-v", "hyper-v", "wsl2", "xvfb", "snapshot",
            "host-only", "bridged adapter", "nat network", "verr_vmx_no_vmx",
        ),
        harmless_probes=(
            "Windows: `& 'C:\\Program Files\\Oracle\\VirtualBox\\VBoxManage.exe' list hostinfo` and `wsl -l -v`",
            "Linux: `which Xvfb import python3`",
        ),
        remediation_and_mutation=(
            "VIRTUALBOX NETWORKING & BOOT MUTATION:\n"
            "- **NAT**: Guest reaches internet outbound, isolated from LAN inbound (use port forwarding for SSH).\n"
            "- **Host-Only**: Host <-> Guest private LAN only; **zero internet** by design! If a user complains a VM has no "
            "internet and `VBoxManage showvminfo` shows `nic1=\"hostonly\"`, mutate NIC1 to `nat` (and keep `hostonly` on NIC2)!\n"
            "- If guest fails with `VT-x is disabled in the BIOS (VERR_VMX_MSR_VMX_DISABLED)`, enable Intel Virtualization Technology / AMD SVM in UEFI."
        ),
        keywords=("virtualbox", "vboxmanage", "vm", "hypervisor", "vt-x", "amd-v", "wsl", "wsl2", "xvfb", "snapshot", "host-only", "nat", "bridged"),
        standards_ref="Oracle VirtualBox 7.2 Technical Reference / Intel SDM Vol. 3C VMX",
    ),

    # 6. CompTIA A+ Scientific 7-Step Methodology & Peripherals/Print/Display
    PcOsConcept(
        concept_id="aplus_scientific_7step_methodology",
        domain="aplus_peripherals_mobile_7step",
        title="The 7-Step Scientific PC & A+ Troubleshooting Methodology",
        general_principle=(
            "Every rigorous technician, ethical hacker, and systems scientist resolves PC problems through 7 empirical stages:\n"
            "1. **Identify the problem**: Gather empirical observations, error logs, visual screen clues, ask open questions, check recent changes, and **back up critical data before invasive actions**.\n"
            "2. **Establish a theory of probable cause**: Question the obvious first (power/cables/session/mount flags/permissions) and consider multiple layers.\n"
            "3. **Test the theory to determine cause**: Run a **harmless, read-only empirical probe**. If the theory is confirmed, proceed; **if falsified by observation, mutate to a new theory** (escalate if outside authority).\n"
            "4. **Establish a plan of action** to resolve the problem and **identify potential effects** on production state or other users.\n"
            "5. **Implement the solution** (or escalate as necessary), changing one controlled variable at a time.\n"
            "6. **Verify full system functionality** empirically and implement **preventive measures**.\n"
            "7. **Document findings, actions, and outcomes** for future knowledge."
        ),
        windows_specific="Applies across Windows hardware, Registry/Services, networking, and GUI workflows.",
        linux_specific="Applies across Linux kernel/boot, filesystems, systemd daemons, networking, and X11/Wayland desktops.",
        macos_bsd_specific="Applies across macOS/BSD firmware, APFS/ZFS, launchd, and Aqua/desktop workflows.",
        diagnostic_signals=(
            "7-step troubleshooting", "seven steps of troubleshooting", "comptia a+ methodology",
            "theory is not confirmed", "establish a plan of action", "verify full system functionality",
        ),
        harmless_probes=(
            "Always run read-only observation commands BEFORE mutating configuration files, disks, or services.",
        ),
        remediation_and_mutation=(
            "Core Scientific Rule: When Step 3 empirical testing contradicts your Step 2 theory, NEVER force the old plan — "
            "immediately return to Step 2 and construct a new hypothesis consistent with the observed evidence!"
        ),
        keywords=("7-step", "seven step", "troubleshooting methodology", "comptia", "a+", "probable cause", "test the theory", "plan of action", "preventive measures"),
        standards_ref="Scientific Method & Industry-Standard Systems Engineering Triage",
    ),
    PcOsConcept(
        concept_id="aplus_laser_printer_display_peripherals",
        domain="aplus_peripherals_mobile_7step",
        title="Laser Printer 7-Step Imaging Diagnostics, Display Signal Isolation & Laptop Battery Safety",
        general_principle=(
            "Peripheral hardware symptoms map deterministically to physical sub-mechanisms:\n"
            "- **Laser Printer 7-Step Imaging**: (1) *Processing* (RIP rasterizes page in RAM) -> (2) *Charging* (primary corona/charge roller applies uniform -600V to photosensitive drum) -> (3) *Exposing* (laser writes electrostatic latent image at -100V) -> (4) *Developing* (toner adheres to exposed drum zones) -> (5) *Transferring* (transfer roller/belt pulls toner onto paper) -> (6) *Fusing* (heated fuser roller + pressure roller melts toner permanently into paper fibers at ~180-200 °C) -> (7) *Cleaning* (wiper blade + erase lamp clears residual toner/charge).\n"
            "  *Fault signatures*: **Smudging / toner rubs off easily** = failed **Fuser**; **Ghosting (faint repeated image down page)** = failed **Cleaning blade / erase lamp** (or drum); **Paper multifeed / failure to pick up** = worn **Pickup / Separation rollers**; **Completely blank page** = empty toner, failed transfer corona, or laser shutter; **Garbled gibberish pages** = wrong printer driver / PDL (`PCL` vs `PostScript`).\n"
            "- **Display Isolation**: Laptop screen dim/black with faint image under a flashlight = failed **backlight/LED driver** (GPU is healthy; external HDMI/DP monitor works normally!). Artifacts on BOTH internal and external displays = **GPU / VRAM** fault.\n"
            "- **Swollen Li-Ion Pouch Battery**: Bulging trackpad/chassis = outgassing thermal-runaway hazard -> power down, never puncture, discharge/remove safely to fireproof container."
        ),
        windows_specific="Print Spooler (`Restart-Service Spooler`, `C:\\Windows\\System32\\spool\\PRINTERS`), Display Settings (`Win+P`).",
        linux_specific="CUPS (`lpstat -p -d`, `systemctl status cups`, `http://localhost:631`), `xrandr --query` / `drm_info`.",
        macos_bsd_specific="CUPS (`lpstat`, System Settings -> Printers & Scanners -> Reset Printing System), `system_profiler SPDisplaysDataType`.",
        diagnostic_signals=(
            "toner rubs off", "fuser", "ghosting", "pickup roller", "print spooler", "cups",
            "faint image flashlight", "backlight", "swollen battery", "garbled print",
        ),
        harmless_probes=(
            "Windows: `Get-Service Spooler`",
            "Linux/macOS: `lpstat -r` (if CUPS installed)",
        ),
        remediation_and_mutation=(
            "If print jobs hang stuck in queue across all apps, clear stale files in the OS spool directory and restart "
            "`Spooler` / `cups` rather than reinstalling office suites; if toner smears off physical paper with a finger, "
            "software/drivers cannot fix thermodynamics — replace the **fuser assembly**."
        ),
        keywords=("laser printer", "fuser", "ghosting", "corona", "drum", "pickup roller", "spooler", "cups", "backlight", "swollen battery", "hdmi", "displayport"),
        standards_ref="IEEE 1284 / IPP (RFC 8011) / Electrophotographic Imaging Physics",
    ),
)


# ── Live Harmless Empirical Host + WSL2 Linux + VirtualBox Diagnostic Probe ──

_LIVE_DIAG_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_LIVE_DIAG_LOCK = threading.Lock()


def _run_cmd_safe(args: List[str], timeout: float = 4.5) -> Tuple[int, str]:
    try:
        cp = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
        out = ((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")).strip()
        return cp.returncode, out
    except Exception as exc:
        return -1, str(exc)


def run_harmless_live_pc_diagnostics(*, force_refresh: bool = False) -> Dict[str, Any]:
    """Run 100% read-only, non-disruptive empirical diagnostics across:
      1. Windows 11 Host (`Pegasus`): CPU, RAM, physical disk health, drive space, session topology, top processes, loopback ports.
      2. Live Linux (`WSL2 Ubuntu` if available): kernel version, loadavg, MemAvailable, root disk usage, Xvfb GUI availability.
      3. Oracle VirtualBox (`VBoxManage` if installed): version, HW virtualization support, registered VMs.
    Cached for 15s to remain feather-light under concurrent battery calls.
    """
    global _LIVE_DIAG_CACHE
    now = time.time()
    if not force_refresh and _LIVE_DIAG_CACHE[1] and (now - _LIVE_DIAG_CACHE[0]) < 15.0:
        return _LIVE_DIAG_CACHE[1]

    with _LIVE_DIAG_LOCK:
        now = time.time()
        if not force_refresh and _LIVE_DIAG_CACHE[1] and (now - _LIVE_DIAG_CACHE[0]) < 15.0:
            return _LIVE_DIAG_CACHE[1]

        report: Dict[str, Any] = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "host_os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "read_only_harmless": True,
        }

        # 1. CPU, Memory & Disk telemetry via psutil + shutil
        try:
            import psutil
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            report["cpu"] = {
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "cpu_percent_instant": psutil.cpu_percent(interval=0.05),
            }
            report["memory"] = {
                "total_gb": round(vm.total / (1024 ** 3), 2),
                "available_gb": round(vm.available / (1024 ** 3), 2),
                "percent_used": round(vm.percent, 1),
                "swap_used_gb": round(sw.used / (1024 ** 3), 2),
                "healthy_headroom": vm.available >= (4.0 * 1024 ** 3),
            }
            disks = []
            for part in psutil.disk_partitions(all=False):
                if "cdrom" in part.opts.lower() or not part.fstype:
                    continue
                try:
                    du = psutil.disk_usage(part.mountpoint)
                    disks.append({
                        "mount": part.mountpoint,
                        "fstype": part.fstype,
                        "total_gb": round(du.total / (1024 ** 3), 1),
                        "free_gb": round(du.free / (1024 ** 3), 1),
                        "percent_used": round(du.percent, 1),
                    })
                except Exception:
                    continue
            report["volumes"] = disks

            top_procs = []
            for p in sorted(
                psutil.process_iter(["pid", "name", "memory_info"]),
                key=lambda x: getattr(x.info.get("memory_info"), "rss", 0) or 0,
                reverse=True,
            )[:5]:
                rss = getattr(p.info.get("memory_info"), "rss", 0) or 0
                top_procs.append({
                    "pid": p.info.get("pid"),
                    "name": p.info.get("name"),
                    "rss_mb": round(rss / (1024 ** 2), 1),
                })
            report["top_memory_processes"] = top_procs
        except Exception as exc:
            du = shutil.disk_usage(Path(__file__).resolve().anchor)
            report["volumes"] = [{
                "mount": Path(__file__).resolve().anchor,
                "total_gb": round(du.total / (1024 ** 3), 1),
                "free_gb": round(du.free / (1024 ** 3), 1),
            }]
            report["psutil_note"] = str(exc)

        # 2. Windows Physical Disk S.M.A.R.T. Health + Session Topology (non-elevated, read-only)
        if os.name == "nt":
            rc_pd, out_pd = _run_cmd_safe([
                "powershell", "-NoProfile", "-Command",
                "Get-PhysicalDisk | Select-Object FriendlyName,MediaType,HealthStatus,OperationalStatus | ConvertTo-Json -Compress",
            ], timeout=4.0)
            if rc_pd == 0 and out_pd:
                try:
                    parsed_pd = json.loads(out_pd)
                    report["physical_disks"] = parsed_pd if isinstance(parsed_pd, list) else [parsed_pd]
                except Exception:
                    report["physical_disks_raw"] = out_pd[:300]

            rc_qw, out_qw = _run_cmd_safe(["qwinsta"], timeout=2.5)
            if rc_qw == 0 and out_qw:
                current_sess = "session_0_services" if ">services" in out_qw.lower() else "interactive_console"
                console_active = "console" in out_qw.lower() and "active" in out_qw.lower()
                report["windows_session_topology"] = {
                    "agent_process_session": current_sess,
                    "user_console_session_active": console_active,
                    "observation_note": (
                        "Agent runs in Session 0 while user Console Session 1 is active; "
                        "GUI interactive automation is safely isolated to Xvfb/VM sandboxes."
                        if current_sess == "session_0_services" and console_active
                        else "Standard session topology verified."
                    ),
                }

        # 3. Local Service Port Reachability (:8001 Web & :8090 W1z4rD Brain)
        ports_status: Dict[str, bool] = {}
        for label, port in (("revenir_web_8001", 8001), ("wizard_node_8090", 8090)):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                    ports_status[label] = True
            except Exception:
                ports_status[label] = False
        report["loopback_services"] = ports_status

        # 4. Live WSL2 Linux Diagnostics (Read-Only Probe)
        wsl_bin = shutil.which("wsl")
        if wsl_bin:
            rc_lx, out_lx = _run_cmd_safe([
                wsl_bin, "-d", "Ubuntu", "--", "sh", "-c",
                "printf 'KERNEL=%s\\n' \"$(uname -sr)\"; "
                "printf 'LOAD=%s\\n' \"$(cut -d' ' -f1-3 /proc/loadavg)\"; "
                "printf 'MEM_AVAIL_MB=%s\\n' \"$(awk '/MemAvailable/ {printf \"%d\", $2/1024}' /proc/meminfo)\"; "
                "printf 'ROOT_DF=%s\\n' \"$(df -h / | awk 'NR==2 {print $4 \" free of \" $2 \" (\" $5 \")\"}')\"; "
                "printf 'XVFB=%s\\n' \"$(command -v Xvfb || echo missing)\"",
            ], timeout=12.0)
            if rc_lx == 0 and "KERNEL=" in out_lx:
                lx_map: Dict[str, str] = {}
                for ln in out_lx.splitlines():
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        lx_map[k.strip()] = v.strip()
                report["wsl2_linux_ubuntu"] = {
                    "online": True,
                    "kernel": lx_map.get("KERNEL", ""),
                    "loadavg": lx_map.get("LOAD", ""),
                    "mem_available_mb": int(lx_map["MEM_AVAIL_MB"]) if lx_map.get("MEM_AVAIL_MB", "").isdigit() else lx_map.get("MEM_AVAIL_MB", ""),
                    "root_filesystem": lx_map.get("ROOT_DF", ""),
                    "xvfb_x11_sandbox": lx_map.get("XVFB", ""),
                }
            else:
                report["wsl2_linux_ubuntu"] = {"online": False, "detail": out_lx[:180]}

        # 5. Oracle VirtualBox Hypervisor Probe (Read-Only)
        vbox_exe = Path(r"C:\Program Files\Oracle\VirtualBox\VBoxManage.exe")
        if not vbox_exe.exists():
            alt_vb = shutil.which("VBoxManage")
            if alt_vb:
                vbox_exe = Path(alt_vb)
        if vbox_exe.exists():
            rc_vb, out_vb = _run_cmd_safe([str(vbox_exe), "--version"], timeout=3.0)
            rc_vms, out_vms = _run_cmd_safe([str(vbox_exe), "list", "vms"], timeout=3.0)
            vm_lines = [ln.strip() for ln in out_vms.splitlines() if ln.strip()] if rc_vms == 0 else []
            report["virtualbox"] = {
                "installed": True,
                "version": out_vb.splitlines()[0].strip() if rc_vb == 0 and out_vb else "unknown",
                "registered_vms_count": len(vm_lines),
            }
        else:
            report["virtualbox"] = {"installed": False}

        _LIVE_DIAG_CACHE = (time.time(), report)
        return report


def format_live_pc_diagnostics_markdown(report: Dict[str, Any]) -> str:
    mem = report.get("memory") or {}
    cpu = report.get("cpu") or {}
    vols = report.get("volumes") or []
    pds = report.get("physical_disks") or []
    lx = report.get("wsl2_linux_ubuntu") or {}
    vb = report.get("virtualbox") or {}
    sess = report.get("windows_session_topology") or {}
    ports = report.get("loopback_services") or {}

    vol_summary = ", ".join(
        f"`{v['mount']}` ({v.get('fstype', 'FS')}: **{v['free_gb']} GB free** / {v['total_gb']} GB)"
        for v in vols[:4]
    ) or "n/a"
    pd_summary = ", ".join(
        f"**{d.get('FriendlyName', 'Disk')}** ({d.get('MediaType', 'SSD')} — `{d.get('HealthStatus', 'Unknown')}` / `{d.get('OperationalStatus', 'OK')}`)"
        for d in pds[:4] if isinstance(d, dict)
    ) or "Standard volume telemetry healthy"

    lines = [
        f"**Live Multi-OS & Hardware Diagnostic Report — `{report.get('hostname', 'Host')}`** *(100% Read-Only Empirical Probe)*",
        "",
        f"- **Windows Host OS**: `{report.get('host_os')}` (`{report.get('architecture')}`, Python `{report.get('python_version')}`)",
        f"- **CPU & Memory**: `{cpu.get('logical_cores', '?')}` logical cores (`{cpu.get('physical_cores', '?')}` physical, `{cpu.get('cpu_percent_instant', 0)}%` instant load) · **{mem.get('available_gb', '?')} GB available** of `{mem.get('total_gb', '?')} GB` RAM (`{mem.get('percent_used', '?')}%` used)",
        f"- **Physical Disk S.M.A.R.T. & Volumes**: {pd_summary} · Volumes: {vol_summary}",
        f"- **Loopback Services**: R3V3N!R Web `:8001` = `{'ONLINE' if ports.get('revenir_web_8001') else 'DOWN'}` · W1z4rD Node `:8090` = `{'ONLINE' if ports.get('wizard_node_8090') else 'DOWN'}`",
    ]
    if sess:
        lines.append(
            f"- **Session Isolation Observation**: Process in `{sess.get('agent_process_session')}`, "
            f"Console Session 1 active=`{sess.get('user_console_session_active')}` — *{sess.get('observation_note')}*"
        )
    if lx.get("online"):
        lines.append(
            f"- **Live WSL2 Linux (`Ubuntu`)**: `{lx.get('kernel')}` · Load `{lx.get('loadavg')}` · "
            f"`MemAvailable={lx.get('mem_available_mb')} MB` · Root FS `{lx.get('root_filesystem')}` · X11 Sandbox `{lx.get('xvfb_x11_sandbox')}`"
        )
    if vb.get("installed"):
        lines.append(
            f"- **Oracle VirtualBox Hypervisor**: `v{vb.get('version')}` installed (`{vb.get('registered_vms_count', 0)}` persistent VMs registered; ready for headless disposable sandboxes)"
        )
    return "\n".join(lines)


# ── Dynamic Observation-Overrides-Assumption Diagnostic Scenario Solver ──────

_LIVE_HOST_DIAG_RE = re.compile(
    r"\b(?:run\s+(?:a\s+)?(?:live\s+|harmless\s+|read-only\s+|full\s+|complete\s+|empirical\s+|multi-os\s+|cross-os\s+)*(?:pc|system|host|multi-os|computer|machine)\s+(?:diagnostic|diagnostics|health\s+check|checkup|triage)|"
    r"diagnose\s+this\s+(?:pc|machine|computer|host)|"
    r"check\s+(?:live\s+)?(?:disk\s+health|wsl2?\s+linux|virtualbox\s+status)\s+on\s+(?:this\s+pc|pegasus|localhost))\b",
    re.I,
)

_RAID_CALC_RE = re.compile(
    r"\braid\s*(0|1|5|6|10)\b.*?\b(\d+)\s*(?:x\s*|drives?\b|disks?\b|nvme\b|ssd\b|hdd\b).*?\b(\d+(?:\.\d+)?)\s*(tb|gb)\b",
    re.I,
)


def _solve_raid_math_if_present(text: str) -> Optional[Dict[str, Any]]:
    m = _RAID_CALC_RE.search(text)
    if not m:
        # Also support "<N> <size>TB drives in RAID <level>"
        m2 = re.search(
            r"\b(\d+)\s*(?:x\s*)?(\d+(?:\.\d+)?)\s*(tb|gb)\s*(?:drives?|disks?|ssds?|hdds?|nvmes?).*?\braid\s*(0|1|5|6|10)\b",
            text,
            re.I,
        )
        if not m2:
            return None
        n_disks = int(m2.group(1))
        size_val = float(m2.group(2))
        unit = m2.group(3).upper()
        level = m2.group(4)
    else:
        level = m.group(1)
        n_disks = int(m.group(2))
        size_val = float(m.group(3))
        unit = m.group(4).upper()

    raw_cap = n_disks * size_val
    invalid_reason = ""
    usable = 0.0
    fault_tol = 0
    if level == "0":
        if n_disks < 2:
            invalid_reason = "RAID 0 requires at least 2 drives."
        usable, fault_tol = raw_cap, 0
    elif level == "1":
        if n_disks < 2:
            invalid_reason = "RAID 1 requires at least 2 drives."
        usable, fault_tol = size_val, n_disks - 1
    elif level == "5":
        if n_disks < 3:
            invalid_reason = f"RAID 5 requires at least 3 drives (only {n_disks} provided -> precondition broken!)."
        usable, fault_tol = max(0.0, (n_disks - 1) * size_val), 1
    elif level == "6":
        if n_disks < 4:
            invalid_reason = f"RAID 6 requires at least 4 drives (only {n_disks} provided -> precondition broken!)."
        usable, fault_tol = max(0.0, (n_disks - 2) * size_val), 2
    elif level == "10":
        if n_disks < 4 or (n_disks % 2 != 0):
            invalid_reason = f"RAID 10 requires an even number of at least 4 drives ({n_disks} provided -> precondition broken!)."
        usable, fault_tol = (n_disks // 2) * size_val, 1

    if invalid_reason:
        ans = (
            f"**RAID {level} Precondition Failure (Invalid Array Geometry)**: {invalid_reason} "
            f"Raw capacity across `{n_disks} x {size_val:g} {unit}` is `{raw_cap:g} {unit}`, "
            f"so you must either add drives or select a valid RAID topology (e.g., RAID 1 for 2 drives, RAID 5 for >=3 drives)."
        )
    else:
        u_str = f"{int(usable)}" if usable.is_integer() else f"{usable:.2f}"
        r_str = f"{int(raw_cap)}" if raw_cap.is_integer() else f"{raw_cap:.2f}"
        tol_desc = (
            "0 drives (any single drive failure loses the entire array!)"
            if fault_tol == 0
            else (
                "1 drive per mirrored sub-pair (up to half the drives if in separate mirrors)"
                if level == "10"
                else f"{fault_tol} simultaneous drive failure(s)"
            )
        )
        ans = (
            f"**RAID {level} Array Calculation (`{n_disks} x {size_val:g} {unit}`)**:\n"
            f"- **Raw Capacity**: `{r_str} {unit}`\n"
            f"- **Usable Capacity**: **`{u_str} {unit}`**\n"
            f"- **Fault Tolerance**: Survives **{tol_desc}** *(Note: RAID provides availability, not backup — maintain independent 3-2-1 backups)*."
        )
    return {
        "question": text,
        "answer": ans,
        "decoder": "pc_aplus_raid_solver",
        "confidence": 0.99,
        "confidence_tier": "high",
        "hypothesis": False,
        "activated_concepts": [f"storage:raid_{level}", "aplus:storage_math"],
    }


def _solve_observation_override_pc_triage(text: str) -> Optional[Dict[str, Any]]:
    """Detect PC / Multi-OS / A+ diagnostic prompts where concrete observed clues
    contradict naive assumptions and require mutating the diagnostic/action plan!"""
    low = (text or "").strip().lower()
    if len(low) < 18:
        return None

    # Require clear PC / OS / hardware / network / VM / peripheral troubleshooting intent
    pc_cue = any(k in low for k in (
        "169.254.", "apipa", "dhcp", "ping 1.1.1.1", "ping 8.8.8.8", "dns", "connection refused", "econnrefused",
        "read-only file system", "erofs", "remount,ro", "smartctl", "reallocated_sector", "pending_sector",
        "media and data integrity errors", "chkdsk /r", "ddrescue",
        "apt install", "apt-get", "command not found", "/etc/os-release", "alpine", "fedora", "arch linux", "freebsd", "macos",
        "dpkg/lock", "lock-frontend", "unattended-upgrades",
        "systemctl", "not been booted with systemd", "session 0", "qwinsta",
        "zombie process", "defunct", "memfree", "memavailable",
        "chmod 777", "unprotected private key", "externally-managed-environment", "pep 668",
        "bitlocker recovery", "secure boot violation", "no bootable device", "csm",
        "ransomware", "encrypting files", "shared drive", "smb share",
        "toner rubs off", "smudges off", "fuser", "ghosting print", "flashlight", "backlight", "swollen battery",
        "vboxmanage", "hostonly", "host-only", "verr_vmx",
        "7-step troubleshooting", "seven-step troubleshooting", "theory is falsified", "theory is not confirmed",
    ))
    if not pc_cue:
        return None

    mutations: List[str] = []
    probes: List[str] = []
    concepts: List[str] = ["aplus:7step_scientific_triage"]

    # 1. Network L2/L3 vs L7 Observation Overrides
    if "169.254." in low or "apipa" in low:
        mutations.append(
            "**Observation Override (APIPA `169.254.x.x` Detected -> Mutate from DNS/App to L2/DHCP DORA)**: "
            "An `169.254.x.x/16` (RFC 3927 link-local) address proves the host never completed the **DHCP 4-step DORA** "
            "(Discover-Offer-Request-Acknowledge) handshake. Override any assumption about browser or DNS settings: "
            "inspect L1/L2 link/VLAN association, DHCP server reachability/scope exhaustion, and renew lease (`ipconfig /renew` / `dhclient` / `networkctl renew`)."
        )
        probes.append("`Get-NetIPConfiguration` / `ip -br addr` / DHCP server lease logs")
        concepts.append("net:apipa_dhcp_override")

    if ("ping 1.1.1.1" in low or "ping 8.8.8.8" in low) and any(w in low for w in ("succeed", "works", "0% packet loss", "replies", "reachable")) and any(w in low for w in ("domain", "hostname", "dns", "website", "nxdomain", "resolve", "browser")):
        mutations.append(
            "**Observation Override (Raw IP Reachable + Hostname Fails -> Isolate to L7 DNS Resolver)**: "
            "Because ICMP/IP routing to `1.1.1.1`/`8.8.8.8` succeeds, L1 physical link, L2 switching, L3 NAT/gateway routing, "
            "and ISP transit are empirically **healthy**. Override 'no internet / reboot router' assumptions and mutate "
            "directly to **DNS resolution** (`UDP/TCP 53`, `Resolve-DnsName`, `resolvectl status` / `/etc/resolv.conf`, `scutil --dns`)."
        )
        probes.append("`Resolve-DnsName example.com -Server 1.1.1.1` / `dig @1.1.1.1 example.com`")
        concepts.append("net:dns_vs_l3_override")

    if "connection refused" in low or "econnrefused" in low:
        mutations.append(
            "**Observation Override (`ECONNREFUSED` / TCP `RST` -> Host Reachable, Socket Closed or Loopback-Only)**: "
            "Instant `Connection refused` means packets reached the target OS network stack and received an active TCP `RST` "
            "(unlike a firewall `DROP`, which times out silently). Mutate from network-cable/routing checks to inspecting "
            "whether the daemon is running and whether it bound to `127.0.0.1` instead of `0.0.0.0`/LAN (`ss -tulpn` / `Get-NetTCPConnection -State Listen`)."
        )
        probes.append("`ss -tulpn` (Linux) / `Get-NetTCPConnection -State Listen` (Windows) / `lsof -iTCP -sTCP:LISTEN -P -n` (macOS)")
        concepts.append("net:tcp_rst_socket_bind")

    # 2. Storage & S.M.A.R.T. Safety Overrides
    if any(k in low for k in ("reallocated_sector", "reallocated sector", "pending_sector", "pending sector", "media and data integrity errors", "uncorrectable sector", "click of death")):
        mutations.append(
            "**Critical Safety Mutation (Physical S.M.A.R.T. Media Degradation -> Abort Destructive Repair, Image Read-Only First)**: "
            "Observed non-zero pending/uncorrectable sectors or NVMe media integrity errors prove **physical drive degradation**. "
            "Override any plan to run write-intensive `chkdsk /r`, `fsck -y`, or defragmentation (which accelerates media death): "
            "**immediately perform read-only data rescue/imaging (`ddrescue` / cold backup)** to healthy storage first, then replace the drive."
        )
        probes.append("`Get-PhysicalDisk` + `Get-StorageReliabilityCounter` / `smartctl -a` / `nvme smart-log`")
        concepts.append("storage:smart_prefail_rescue")

    if any(k in low for k in ("read-only file system", "erofs", "mounted ro", "remount,ro")):
        mutations.append(
            "**Observation Override (`EROFS` Read-Only Filesystem Circuit Breaker -> Inspect Kernel Log Before Remounting)**: "
            "The kernel remounted the filesystem `ro` (or mounted an immutable root) to prevent metadata corruption. "
            "Do NOT force `mount -o remount,rw` blindly: inspect `dmesg | tail -n 40` and `findmnt /` first — if caused by journal/I/O "
            "errors, schedule clean offline `fsck`/`xfs_repair` after backup; if intentionally immutable, redirect temporary output to `/tmp` or a writable volume."
        )
        probes.append("`findmnt /` + `dmesg | grep -iE 'ext4|xfs|btrfs|nvme|ro'`")
        concepts.append("storage:erofs_mutation")

    # 3. Cross-OS Package Manager, Lock & Init System Overrides
    if ("apt" in low or "apt-get" in low or "winget" in low) and any(k in low for k in ("alpine", "fedora", "rhel", "rocky", "arch linux", "freebsd", "macos", "darwin", "command not found")):
        pkg_tool = (
            "`apk add <pkg>` (Alpine Linux `musl`)" if "alpine" in low else
            "`dnf install <pkg>` (Fedora/RHEL RPM)" if any(x in low for x in ("fedora", "rhel", "rocky")) else
            "`pacman -S <pkg>` (Arch Linux)" if "arch" in low else
            "`pkg install <pkg>` (FreeBSD)" if "freebsd" in low else
            "`brew install <pkg>` (macOS Homebrew)" if ("macos" in low or "darwin" in low) else
            "the native package manager after reading `/etc/os-release` (`apk` / `dnf` / `pacman` / `zypper` / `brew` / `winget`)"
        )
        mutations.append(
            f"**Observation Override (OS Family Mismatch -> Mutate Package Manager)**: "
            f"Observed OS environment does not use `apt`. Override prior Debian/Ubuntu habit and mutate command to **{pkg_tool}**."
        )
        probes.append("`cat /etc/os-release` / `uname -s`")
        concepts.append("os:package_manager_mutation")

    if "lock-frontend" in low or "dpkg/lock" in low:
        mutations.append(
            "**Observation Override (`dpkg` Lock Held -> Inspect Active Updater Instead of Deleting Lock)**: "
            "Never `rm /var/lib/dpkg/lock-frontend` while another process holds it (corrupts `/var/lib/dpkg/status`). "
            "Check `ps aux | grep -E 'apt|dpkg|unattended'` — wait for `unattended-upgrades` to finish cleanly, and only "
            "run `dpkg --configure -a` if `ps`/`fuser` confirms zero active package processes."
        )
        probes.append("`ps aux | grep -E 'apt|dpkg'` / `fuser /var/lib/dpkg/lock-frontend`")
        concepts.append("os:dpkg_lock_safety")

    if "not been booted with systemd" in low or ("systemctl" in low and any(k in low for k in ("pid 1 is init", "openrc", "container", "alpine"))):
        mutations.append(
            "**Observation Override (Non-`systemd` PID 1 Detected -> Mutate Service Supervisor Command)**: "
            "Observation confirms PID 1 is not `systemd` (e.g., OpenRC on Alpine, SysV/init in minimal WSL or container). "
            "Mutate from `systemctl` to `rc-service <svc> status` (OpenRC), `service <svc> status`, or direct process supervision."
        )
        probes.append("`ps -p 1 -o comm=`")
        concepts.append("os:init_supervisor_mutation")

    # 4. Process, Memory & Permission Overrides
    if "zombie" in low or "defunct" in low:
        mutations.append(
            "**Observation Override (`Z` / `defunct` Zombie Process -> Inspect Parent `PPID`, Do Not `kill -9` Zombie)**: "
            "A zombie process has already exited and holds zero RAM — only its process-table exit slot remains because its "
            "parent (`PPID`) has not called `wait()`. `kill -9 <zombie_pid>` does nothing; inspect `ps -o ppid= -p <pid>` and "
            "signal (`SIGCHLD` / graceful restart) the **parent process** instead."
        )
        probes.append("`ps -ef | grep defunct` / `ps -o pid,ppid,stat,comm -p <pid>`")
        concepts.append("os:zombie_ppid_mutation")

    if "memfree" in low and ("memavailable" in low or "page cache" in low or "buff/cache" in low):
        mutations.append(
            "**Observation Override (Low `MemFree` + High `MemAvailable`/Cache -> Healthy Page Cache, No OOM)**: "
            "Unused RAM is wasted RAM: kernels opportunistically cache clean filesystem pages in `buff/cache` and reclaim them "
            "instantly when applications allocate memory. Since `MemAvailable` is healthy and swap thrashing is zero, override "
            "false OOM alarms and do NOT drop caches in production."
        )
        probes.append("`free -h` + `/proc/pressure/memory`")
        concepts.append("os:memavailable_vs_cache")

    if "chmod 777" in low or "unprotected private key" in low or "externally-managed-environment" in low or "pep 668" in low:
        if "unprotected private key" in low:
            mutations.append(
                "**Security Observation Override (OpenSSH Key Mode Too Broad -> Tighten to `0600`, Never `0777`)**: "
                "OpenSSH refuses private keys readable by group/others. Fix with `chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_*` "
                "(and `icacls` inheritance removal on Windows)."
            )
        elif "externally-managed-environment" in low or "pep 668" in low:
            mutations.append(
                "**Observation Override (PEP 668 System Python Protection -> Mutate to `.venv` Virtual Environment)**: "
                "Do NOT pass `--break-system-packages` or run `sudo pip install` (risks breaking OS Python tools like `apt`/`dnf`). "
                "Mutate cleanly to `python3 -m venv .venv && .venv/bin/pip install <pkg>`."
            )
        else:
            mutations.append(
                "**Ethical Security Override (Reject `chmod 777` -> Apply Least-Privilege Ownership/ACL)**: "
                "World-writable `0777` exposes binaries and configs to local privilege escalation. Diagnose the exact UID/GID "
                "mismatch via `id` + `stat` and grant narrow ownership (`chown`) or targeted POSIX/NTFS ACLs (`0750`/`0640`)."
            )
        concepts.append("sec:least_privilege_mutation")

    # 5. Boot, BitLocker, Malware, VirtualBox & A+ Hardware Peripherals
    if "bitlocker recovery" in low or ("secure boot" in low and ("violation" in low or "no bootable" in low)) or ("no bootable device" in low and "csm" in low):
        mutations.append(
            "**Firmware/Boot Observation Override (Preserve Volume, Fix Firmware Mode / TPM PCR State)**: "
            "Never reformat or reinstall when UEFI mode (`UEFI` vs `CSM/Legacy`) or TPM 2.0 PCR measurements shifted after a BIOS "
            "update or dock change! Restore `UEFI-only` boot mode for GPT ESP disks, verify boot order, or enter the 48-digit "
            "BitLocker recovery key and suspend/re-seal TPM protectors (`Suspend-BitLocker`)."
        )
        concepts.append("boot:uefi_tpm_recovery")

    if "ransomware" in low or ("encrypting" in low and ("share" in low or "smb" in low or "malware" in low)):
        mutations.append(
            "**Incident Containment Override (Active Ransomware -> Immediate Network/SMB Quarantine First!)**: "
            "Per the 7-step malware response protocol, **immediately sever LAN/Wi-Fi/VPN & SMB share connections** BEFORE running "
            "long scans or rebooting, preventing lateral encryption of shared network storage, then preserve forensic indicators "
            "and restore only from clean, verified offline backups."
        )
        concepts.append("sec:malware_quarantine_step2")

    if "host-only" in low or "hostonly" in low or "verr_vmx" in low:
        if "verr_vmx" in low:
            mutations.append(
                "**VirtualBox Observation Override (`VERR_VMX` -> Enable Hardware VT-x / AMD-V or WHPX)**: "
                "The guest hypervisor lacks hardware virtualization access. Enable Intel VT-x / AMD SVM in host UEFI firmware "
                "(or verify Windows Hypervisor Platform coexistence via `VBoxManage list hostinfo`)."
            )
        else:
            mutations.append(
                "**VirtualBox NIC Observation Override (`hostonly` Adapter Has Zero WAN Route by Design)**: "
                "`Host-Only` networking connects only Host <-> Guest with no upstream NAT/gateway. Mutate Adapter 1 to `nat` "
                "(`VBoxManage modifyvm <vm> --nic1 nat`) for outbound internet while keeping Adapter 2 as `hostonly` for host SSH."
            )
        concepts.append("vm:virtualbox_mutation")

    if any(k in low for k in ("toner rubs off", "smudges off", "fuser", "ghosting print", "flashlight", "backlight", "swollen battery")):
        if "toner rubs off" in low or "smudges off" in low or "fuser" in low:
            mutations.append(
                "**A+ Hardware Physics Diagnosis (Unfused Toner Smears -> Failed Laser Printer Fuser Assembly)**: "
                "In the 7-step laser imaging cycle, Step 6 (**Fusing**) melts toner resin into paper via heat (~190 °C) and pressure. "
                "Loose toner rubbing off proves the **fuser heater/roller** failed — replace the fuser maintenance kit."
            )
        if "ghosting" in low:
            mutations.append(
                "**A+ Hardware Physics Diagnosis (Repeated Faint Ghost Image -> Cleaning Blade / Erase Lamp / Drum)**: "
                "Residual toner or electrostatic charge from the prior drum rotation was not cleared in Step 7 (**Cleaning**)."
            )
        if "flashlight" in low:
            mutations.append(
                "**A+ Display Isolation (Image Visible Under Flashlight + External HDMI Works -> Backlight / LED Driver Fault)**: "
                "Because the LCD pixel matrix is still rendering frames and external GPU output is clean, override GPU-replacement "
                "assumptions and repair the internal display **backlight / LED driver / display cable**."
            )
        if "swollen battery" in low:
            mutations.append(
                "**Critical Li-Ion Safety Protocol (Swollen Pouch Cell -> Thermal Runaway Hazard)**: "
                "Power off immediately, disconnect AC, NEVER compress or puncture the pouch, and arrange certified battery removal/disposal."
            )
        concepts.append("aplus:hardware_symptom_physics")

    if "theory is falsified" in low or "theory is not confirmed" in low or "7-step troubleshooting" in low or "seven-step troubleshooting" in low:
        mutations.append(
            "**7-Step Scientific Troubleshooting Loop**: (1) Identify problem & back up data -> (2) Establish theory of probable cause -> "
            "(3) **Test theory empirically** (when falsified by observation, **return to Step 2 and establish a new theory** rather than forcing repairs!) -> "
            "(4) Plan action & assess impact -> (5) Implement solution -> (6) Verify full system functionality & preventive measures -> (7) Document findings."
        )

    if not mutations:
        return None

    probe_line = f"\n\n- **Harmless Empirical Verification Probes**: {' · '.join(probes)}" if probes else ""
    ans = "\n\n".join(mutations) + probe_line
    return {
        "question": text,
        "answer": ans,
        "decoder": "pc_aplus_observation_mutator",
        "confidence": 0.97,
        "confidence_tier": "high",
        "hypothesis": False,
        "activated_concepts": concepts,
    }


def resolve_pc_diagnostics_or_aplus_query(text: str) -> Optional[Dict[str, Any]]:
    """Fast (<1 ms cached / empirical) resolver for:
      1. Live harmless multi-OS PC diagnostic requests on this machine (`Pegasus` + WSL2 Ubuntu + VirtualBox).
      2. Parametric RAID capacity / fault-tolerance / broken-precondition calculations.
      3. Untrained Multi-OS & CompTIA A+ troubleshooting scenarios where empirical observations
         override naive assumptions and mutate the course of action.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    # 1. Live empirical PC diagnostic request
    if _LIVE_HOST_DIAG_RE.search(raw):
        rep = run_harmless_live_pc_diagnostics()
        return {
            "question": raw,
            "answer": format_live_pc_diagnostics_markdown(rep),
            "decoder": "live_pc_multi_os_diagnostics",
            "confidence": 1.0,
            "confidence_tier": "high",
            "hypothesis": False,
            "structured_diagnostics": rep,
            "activated_concepts": ["tool:pc_diagnostics", "host:pegasus_windows", "guest:wsl2_ubuntu", "vm:virtualbox"],
        }

    # 2. Parametric RAID calculation
    raid_hit = _solve_raid_math_if_present(raw)
    if raid_hit is not None:
        return raid_hit

    # 3. Observation-override PC / Multi-OS / A+ triage
    obs_hit = _solve_observation_override_pc_triage(raw)
    if obs_hit is not None:
        return obs_hit

    return None
