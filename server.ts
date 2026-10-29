import 'dotenv/config';
import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { apiRouter } from './server/apiRouter.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;
  const HOST = '0.0.0.0';
  const isProduction = process.env.NODE_ENV === 'production';

  // Enable JSON parser with sufficient limit for computer vision image payloads
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  // Mount API router and ensure unmatched /api/* paths return JSON 404
  app.use('/api', apiRouter);
  app.use('/api', (_req, res) => {
    res.status(404).json({ error: 'API endpoint not found' });
  });

  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    const indexHtmlPath = path.join(distPath, 'index.html');

    if (!fs.existsSync(indexHtmlPath)) {
      console.error(
        `[ModelGuard Production Warning] Built frontend not found at ${indexHtmlPath}. Run "npm run build" before starting in production.`
      );
    }

    app.use(express.static(distPath));

    app.get('*', (req, res, next) => {
      if (req.path.startsWith('/api')) {
        return next();
      }
      if (!fs.existsSync(indexHtmlPath)) {
        res.status(503).json({
          error: 'Frontend bundle missing. Build the frontend with "npm run build" first.',
        });
        return;
      }
      res.sendFile(indexHtmlPath);
    });
  }

  app.listen(PORT, HOST, () => {
    console.log(`=======================================================`);
    console.log(`ModelGuard MLOps Platform Server Active (${isProduction ? 'production' : 'development'})`);
    console.log(`Listening on: http://${HOST}:${PORT}`);
    console.log(`=======================================================`);
  });
}

startServer();
