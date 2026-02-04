import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = join(__dirname, '..', '..');
const viteBinaryName = process.platform === 'win32' ? 'vite.cmd' : 'vite';
const viteBinaryPath = join(repoRoot, 'apps', 'web', 'node_modules', '.bin', viteBinaryName);

if (!existsSync(viteBinaryPath)) {
  console.log('Installing dependencies for @state-of-loci/web...');
  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const installResult = spawnSync(npmCommand, ['install', '--include=dev'], {
    cwd: repoRoot,
    stdio: 'inherit'
  });

  if (installResult.status !== 0) {
    console.error('\nFailed to install workspace dependencies. Please fix the errors above and rerun `npm install`.');
    process.exit(installResult.status ?? 1);
  }

  if (!existsSync(viteBinaryPath)) {
    console.error('\nVite binary is still missing after installation. Please run `npm install` manually and verify there were no errors.');
    process.exit(1);
  }
}
