import constants from 'next/constants.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export default function nextConfig(phase) {
  const common = { poweredByHeader: false, reactStrictMode: true, images: { unoptimized: true } };
  const project = fileURLToPath(new URL('.', import.meta.url));
  const client = path.dirname(fileURLToPath(import.meta.resolve('next/dist/client/next.js')));
  if (process.platform === 'win32' && path.isAbsolute(path.relative(project, client))) {
    // Next's Pages entry prefixes "./" even when a junction points to a different drive.
    common.webpack = config => {
      // Webpack cannot persist a dependency snapshot across these two volumes.
      config.cache = { type: 'memory' };
      for (const entry of ['next.js', 'next-dev.js', 'app-next-dev.js']) {
        const absolute = path.join(client, entry).replaceAll('\\', '/');
        config.resolve.alias['./' + absolute + '$'] = absolute;
      }
      return config;
    };
  }
  if (phase !== constants.PHASE_DEVELOPMENT_SERVER) {
    return { ...common, output: 'export' };
  }

  const port = Number(process.env.RELAY_DEV_API_PORT);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('Inicia Relay con npm run dev para conectar con el servicio Python.');
  }
  const backend = `http://127.0.0.1:${port}`;
  return {
    ...common,
    devIndicators: false,
    experimental: { proxyTimeout: 1800000, proxyClientMaxBodySize: '32mb' },
    async rewrites() {
      return ['api', 'estudio'].map(path => ({
        source: `/${path}/:rest*`,
        destination: `${backend}/${path}/:rest*`
      }));
    }
  };
}
