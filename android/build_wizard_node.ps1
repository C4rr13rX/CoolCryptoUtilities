# ---------------------------------------------------------------------------
# Cross-compile the W1z4rDV1510n node for Android arm64.
#
# The node uses rustls rather than OpenSSL, which is what makes this a straight
# cross-compile. Note that this required a fix in the W1z4rD repo itself:
# crates/core and crates/node asked for reqwest's "rustls-tls" feature WITHOUT
# default-features = false. Cargo features are additive, so reqwest's default
# native-tls came along too and dragged in openssl-sys, which needs a C OpenSSL
# build and fails any cross-compile. Both Cargo.toml files now disable defaults.
#
# The output is copied to jniLibs as `libw1z4rd_node.so` even though it is an
# executable, not a library. That naming is required, not cosmetic: Android
# only extracts and grants execute permission to files matching `lib*.so` in
# the native library directory. A binary placed in assets/ lands on a noexec
# mount and fails with "Permission denied" at exec time.
#
# Usage:
#   ./build_wizard_node.ps1                 # release build
#   ./build_wizard_node.ps1 -Debug          # faster, larger, unoptimised
# ---------------------------------------------------------------------------
param(
    [switch]$Debug,
    [string]$WizardRepo = "D:\Projects\W1z4rDV1510n",
    [string]$NdkVersion = "27.1.12297006",
    # 26 matches minSdk in app/build.gradle. A higher API here would produce a
    # binary that refuses to load on devices the app otherwise supports.
    [int]$ApiLevel = 26
)

$ErrorActionPreference = "Stop"
$Target = "aarch64-linux-android"
$AndroidHome = if ($env:ANDROID_HOME) { $env:ANDROID_HOME }
               else { "$env:LOCALAPPDATA\Android\Sdk" }
$Ndk = Join-Path $AndroidHome "ndk\$NdkVersion"

Write-Host "=== Wizard node -> Android arm64 ===" -ForegroundColor Cyan
Write-Host "  repo : $WizardRepo"
Write-Host "  ndk  : $Ndk"

if (-not (Test-Path $WizardRepo)) { throw "Wizard repo not found: $WizardRepo" }
if (-not (Test-Path $Ndk))        { throw "NDK not found: $Ndk. Install it via the SDK Manager." }

# --- 1. Rust target -------------------------------------------------------
$installed = & rustup target list --installed
if ($installed -notcontains $Target) {
    Write-Host "[1/4] adding rust target $Target"
    & rustup target add $Target
} else {
    Write-Host "[1/4] rust target present"
}

# --- 2. Point cargo at the NDK linker ------------------------------------
# Cargo cannot infer the Android linker; without these it invokes the host
# linker and fails with a wall of unresolved symbols.
$ToolchainBin = Join-Path $Ndk "toolchains\llvm\prebuilt\windows-x86_64\bin"
$Clang = Join-Path $ToolchainBin "$Target$ApiLevel-clang.cmd"
if (-not (Test-Path $Clang)) { throw "NDK clang wrapper not found: $Clang" }

$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = $Clang
$env:CC_aarch64_linux_android  = $Clang
$env:AR_aarch64_linux_android  = Join-Path $ToolchainBin "llvm-ar.exe"
# ring (rustls's crypto backend) compiles C and assembly in its build script,
# so a linker alone is not enough -- without CFLAGS naming the target it fails
# with "failed to run custom build command for `ring`". ANDROID_NDK_ROOT is
# read by several -sys crates that probe for the NDK themselves.
$env:CFLAGS_aarch64_linux_android = "--target=aarch64-linux-android$ApiLevel"
$env:ANDROID_NDK_ROOT = $Ndk
Write-Host "[2/4] linker: $Clang"

# --- 3. Build -------------------------------------------------------------
$Profile = if ($Debug) { "debug" } else { "release" }
Write-Host "[3/4] cargo build ($Profile)…"
Push-Location $WizardRepo
try {
    # --bin: the node crate also produces w1z4rd_brain_server, and building
    # every target doubles the time for a binary we do not ship.
    $buildArgs = @("build", "-p", "w1z4rdv1510n-node",
                   "--bin", "w1z4rdv1510n-node", "--target", $Target)
    if (-not $Debug) { $buildArgs += "--release" }
    & cargo @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

# --- 4. Stage into jniLibs -----------------------------------------------
$Built = Join-Path $WizardRepo "target\$Target\$Profile\w1z4rdv1510n-node"
if (-not (Test-Path $Built)) { throw "built binary not found: $Built" }

$JniDir = Join-Path $PSScriptRoot "app\src\main\jniLibs\arm64-v8a"
New-Item -ItemType Directory -Force -Path $JniDir | Out-Null
$Dest = Join-Path $JniDir "libw1z4rd_node.so"
Copy-Item $Built $Dest -Force

$SizeMb = [math]::Round((Get-Item $Dest).Length / 1MB, 1)
Write-Host "[4/4] staged $Dest ($SizeMb MB)" -ForegroundColor Green
Write-Host ""
Write-Host "Now build the APK:  ./gradlew assembleDebug" -ForegroundColor Cyan
