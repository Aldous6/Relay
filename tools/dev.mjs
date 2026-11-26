import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const root = fileURLToPath(new URL('../', import.meta.url));
const web = path.join(root, 'web');
const windows = process.platform === 'win32';

function findPython() {
  const venv = path.join(root, '.venv-relay', windows ? 'Scripts/python.exe' : 'bin/python');
  const candidates = [];
  if (existsSync(venv)) candidates.push([venv]);
  if (windows) candidates.push(['py', '-3.12'], ['python'], ['python3']);
  else candidates.push(['python3'], ['python']);
  for (const [command, ...args] of candidates) {
    const check = spawnSync(command, [...args, '-c', 'import sys; sys.exit(sys.version_info < (3, 12))'],
      { stdio: 'ignore', windowsHide: true });
    if (check.status === 0) return [command, args];
  }
  throw new Error('Se necesita Python 3.12 o posterior para ejecutar la API de Relay.');
}

function start() {
  const { values } = parseArgs({ options: {
    port: { type: 'string', short: 'p' },
    help: { type: 'boolean', short: 'h' }
  } });
  if (values.help) {
    console.log('npm run dev [-- --port PUERTO]  (predeterminado: PORT o 3000)');
    return;
  }
  const port = Number(values.port ?? process.env.PORT ?? 3000);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error('El puerto del frontend debe ser un entero entre 1024 y 65535.');
  }

  let next;
  try {
    next = createRequire(path.join(web, 'package.json')).resolve('next/dist/bin/next');
  } catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND') throw error;
    throw new Error('Falta Next.js. Instala las dependencias del proyecto antes de iniciar Relay.');
  }
  const [python, pythonArgs] = findPython();
  const children = [];
  let stopping = false;

  function stop(code, message) {
    if (stopping) return;
    stopping = true;
    process.exitCode = code;
    if (message) console.error(message);
    const groups = [];
    for (const child of children) {
      if (!child.pid) continue;
      if (windows) {
        if (child.exitCode !== null || child.signalCode !== null) continue;
        const taskkill = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'taskkill.exe');
        const result = spawnSync(taskkill, ['/pid', String(child.pid), '/t', '/f'], { windowsHide: true, stdio: 'ignore' });
        if (result.status !== 0) child.kill();
      } else {
        // Detached POSIX children lead their own groups, including the Next worker.
        try {
          process.kill(-child.pid, 'SIGTERM');
          groups.push(child.pid);
        } catch (error) {
          if (error.code !== 'ESRCH') console.error(error.message);
        }
      }
    }
    if (groups.length) {
      setTimeout(() => {
        for (const group of groups) {
          try { process.kill(-group, 'SIGKILL'); } catch (error) {
            if (error.code !== 'ESRCH') console.error(error.message);
          }
        }
      }, 1500);
    }
  }

  function launch(command, args, options, name) {
    const child = spawn(command, args, { ...options, detached: true, windowsHide: true });
    children.push(child);
    child.once('error', error => stop(1, `No se pudo iniciar ${name}: ${error.message}`));
    child.once('exit', (code, signal) => {
      const expected = name === 'Next.js' && code === 0;
      stop(expected ? 0 : code || 1, expected ? undefined : `${name} terminó (${signal || code}).`);
    });
    return child;
  }

  process.once('SIGINT', () => stop(130));
  process.once('SIGTERM', () => stop(143));
  process.once('SIGHUP', () => stop(129));
  if (windows) process.once('SIGBREAK', () => stop(130));

  const pythonPath = [path.join(root, 'src'), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
  const api = launch(python, [...pythonArgs, '-u', '-B', '-m', 'relay_router.web_server',
    '--dev-api', '--frontend-port', String(port)], {
    cwd: root, stdio: ['ignore', 'pipe', 'inherit'], env: { ...process.env, PYTHONPATH: pythonPath }
  }, 'la API de Relay');
  const lines = readline.createInterface({ input: api.stdout, crlfDelay: Infinity });
  lines.once('line', first => {
    if (stopping) return;
    let apiPort;
    try { apiPort = JSON.parse(first).relay_api_port; } catch { /* Invalid readiness message. */ }
    if (!Number.isInteger(apiPort) || apiPort < 1 || apiPort > 65535) {
      stop(1, 'La API local de Relay no indicó su puerto de desarrollo.');
      return;
    }
    lines.on('line', line => console.log(line));
    console.log(`Relay en desarrollo: http://127.0.0.1:${port}`);
    launch(process.execPath, [next, 'dev', '--webpack', '--hostname', '127.0.0.1', '--port', String(port)], {
      cwd: web, stdio: 'inherit',
      env: { ...process.env, RELAY_DEV_API_PORT: String(apiPort), NEXT_TELEMETRY_DISABLED: '1' }
    }, 'Next.js');
  });
}

try {
  start();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
