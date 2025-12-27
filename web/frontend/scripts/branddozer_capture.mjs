import fs from 'fs';
import path from 'path';
let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (err) {
  const message = err instanceof Error ? err.message : String(err);
  console.error(`playwright not installed: ${message}`);
  process.exit(1);
}

const baseUrl = (process.env.BRANDDOZER_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const outputDir = process.env.BRANDDOZER_SCREENSHOT_DIR || path.resolve(process.cwd(), 'playwright-artifacts');
const routesEnv = process.env.BRANDDOZER_ROUTES || '';

let routes = [
  { name: 'dashboard', path: '/' },
  { name: 'pipeline', path: '/pipeline' },
  { name: 'streams', path: '/streams' },
  { name: 'telemetry', path: '/telemetry' },
  { name: 'organism', path: '/organism' },
  { name: 'wallet', path: '/wallet' },
  { name: 'advisories', path: '/advisories' },
  { name: 'model-lab', path: '/lab' },
  { name: 'data-lab', path: '/datalab' },
  { name: 'guardian', path: '/guardian' },
  { name: 'code-graph', path: '/codegraph' },
  { name: 'integrations', path: '/integrations' },
  { name: 'settings', path: '/settings' },
  { name: 'branddozer', path: '/branddozer' },
  { name: 'branddozer-solo', path: '/branddozer/solo' },
  { name: 'u53rxr080t', path: '/u53rxr080t' },
];
if (routesEnv) {
  try {
    const parsed = JSON.parse(routesEnv);
    if (Array.isArray(parsed) && parsed.length) {
      routes = parsed;
    }
  } catch (err) {
    console.warn('Invalid BRANDDOZER_ROUTES; using defaults.');
  }
}

const viewports = [
  { name: 'mobile', width: 390, height: 844, isMobile: true, hasTouch: true },
  { name: 'desktop', width: 1440, height: 900 },
];

fs.mkdirSync(outputDir, { recursive: true });

const authUser = process.env.BRANDDOZER_AUTH_USER || '';
const authPass = process.env.BRANDDOZER_AUTH_PASS || '';
const authPath = process.env.BRANDDOZER_AUTH_LOGIN_PATH || '/';
const results = [];
const authStatus = { status: 'skipped', detail: '' };
const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

async function safeGoto(page, url) {
  try {
    await page.goto(url, { waitUntil: 'networkidle', timeout: 45000 });
  } catch (err) {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  }
  await page.waitForTimeout(800);
}

async function ensureLogin(page) {
  await safeGoto(page, `${baseUrl}${authPath}`);
  const loginForm = await page.$('form.login-card');
  if (!loginForm) {
    authStatus.status = 'not-required';
    return;
  }
  if (!authUser || !authPass) {
    authStatus.status = 'failed';
    authStatus.detail = 'Login required but BRANDDOZER_AUTH_USER/BRANDDOZER_AUTH_PASS not set.';
    throw new Error(authStatus.detail);
  }
  const usernameSelector = '#id_username, input[name="username"], input[type="text"]';
  const passwordSelector = '#id_password, input[name="password"], input[type="password"]';
  await page.fill(usernameSelector, authUser);
  await page.fill(passwordSelector, authPass);
  await Promise.allSettled([
    page.waitForNavigation({ waitUntil: 'domcontentloaded', timeout: 15000 }),
    page.click('button[type="submit"]'),
  ]);
  try {
    await page.waitForSelector('#app', { timeout: 15000 });
    authStatus.status = 'ok';
  } catch (err) {
    authStatus.status = 'failed';
    authStatus.detail = 'Login failed; app did not load after submit.';
    throw new Error(authStatus.detail);
  }
}

let exitCode = 0;
try {
  for (const viewport of viewports) {
    const context = await browser.newContext({
      viewport: { width: viewport.width, height: viewport.height },
      isMobile: Boolean(viewport.isMobile),
      hasTouch: Boolean(viewport.hasTouch),
      deviceScaleFactor: viewport.isMobile ? 2 : 1,
    });
    const page = await context.newPage();
    await page.emulateMedia({ colorScheme: 'dark' });
    await ensureLogin(page);
    for (const route of routes) {
      const url = `${baseUrl}${route.path || ''}`;
      await safeGoto(page, url);
      const filename = `${route.name || 'route'}-${viewport.name}.png`;
      const shotPath = path.join(outputDir, filename);
      await page.screenshot({ path: shotPath, fullPage: true });
      results.push({ route: route.name, viewport: viewport.name, path: shotPath });
    }
    await context.close();
  }
} catch (err) {
  exitCode = 1;
  const message = err instanceof Error ? err.message : String(err);
  if (!authStatus.detail) {
    authStatus.detail = message;
  }
} finally {
  await browser.close();
}

process.stdout.write(JSON.stringify({ baseUrl, outputDir, routes, shots: results, auth: authStatus }));
process.exit(exitCode);
