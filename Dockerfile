# syntax=docker/dockerfile:1

# Supply Bun binary strictly for deterministic frozen-lockfile installs from existing bun.lock
FROM oven/bun:1-slim AS bun_bin

# Shared Node.js 22 Debian Bookworm Slim base with root CA certificates for HTTPS registry & API calls
FROM node:22-bookworm-slim AS node_base
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Stage 1: Install full locked dependencies & build Vite production SPA
FROM node_base AS builder

COPY --from=bun_bin /usr/local/bin/bun /usr/local/bin/bun
RUN ln -sf /usr/local/bin/bun /usr/local/bin/bunx && bun --version
WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY index.html tsconfig.json vite.config.ts metadata.json ./
COPY public ./public
COPY src ./src
COPY server ./server
COPY server.ts ./server.ts

RUN npm run build

# Stage 2: Install locked production-only runtime dependencies
FROM node_base AS prod-deps

COPY --from=bun_bin /usr/local/bin/bun /usr/local/bin/bun
RUN ln -sf /usr/local/bin/bun /usr/local/bin/bunx
WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production

# Stage 3: Minimal Node.js 22 Production Runtime Image (no Vite dev server)
FROM node_base AS runner

WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV MODELGUARD_DATA_DIR=/app/data

COPY --chown=node:node package.json ./
COPY --chown=node:node --from=prod-deps /app/node_modules ./node_modules
COPY --chown=node:node --from=builder /app/dist ./dist
COPY --chown=node:node public ./public
COPY --chown=node:node src/types ./src/types
COPY --chown=node:node server ./server
COPY --chown=node:node server.ts ./server.ts

RUN mkdir -p /app/data/uploads /app/data/models && chown -R node:node /app/data

USER node

EXPOSE 3000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:' + (process.env.PORT || 3000) + '/api/health').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["node", "--experimental-strip-types", "server.ts"]
