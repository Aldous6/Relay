import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const execute = promisify(execFile);
const entry = fileURLToPath(new URL('../tools/dev.mjs', import.meta.url));

test('validates port before starting either process', async () => {
  await assert.rejects(execute(process.execPath, [entry, '--port', '65536']), error => {
    assert.equal(error.code, 1);
    assert.match(error.stderr, /entre 1024 y 65535/);
    return true;
  });
});

test('starts Next after readiness and stops the Python process when Next exits', async t => {
  const temp = path.resolve(os.tmpdir());
  const root = await mkdtemp(path.join(temp, 'relay-dev-test-'));
  t.after(async () => {
    assert.equal(path.dirname(root), temp);
    await rm(root, { recursive: true, force: true });
  });
  const runner = path.join(root, 'tools', 'dev.mjs');
  const cli = path.join(root, 'web', 'node_modules', 'next', 'dist', 'bin', 'next.js');
  const pythonModule = path.join(root, 'src', 'relay_router', 'web_server.py');
  for (const file of [runner, cli, pythonModule]) await mkdir(path.dirname(file), { recursive: true });
  await copyFile(entry, runner);
  await writeFile(pythonModule, [
    'from pathlib import Path',
    'import json, os, sys, time',
    'assert sys.argv[1:] == ["--dev-api", "--frontend-port", "3401"]',
    'Path("backend.pid").write_text(str(os.getpid()))',
    'print(json.dumps({"relay_api_port": 41567}), flush=True)',
    'time.sleep(12)',
  ].join('\n'));
  await writeFile(cli, `
    const assert = require('node:assert/strict');
    assert.deepEqual(process.argv.slice(2), ['dev', '--webpack', '--hostname', '127.0.0.1', '--port', '3401']);
    assert.equal(process.env.RELAY_DEV_API_PORT, '41567');
    assert.equal(process.env.NEXT_TELEMETRY_DISABLED, '1');
    assert.equal(require('node:path').basename(process.cwd()), 'web');
    console.log('next-fixture-ok');
  `);
  const { stdout } = await execute(process.execPath, [runner, '-p', '3401'], {
    env: { ...process.env, PORT: '3007' }, timeout: 10000
  });
  assert.match(stdout, /next-fixture-ok/);
  const pid = Number(await readFile(path.join(root, 'backend.pid'), 'utf8'));
  assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' });
});
