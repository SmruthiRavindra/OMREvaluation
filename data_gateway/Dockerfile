# =============================================================================
# Dockerfile — Vision OMR Data Gateway (Node.js / Express)
# =============================================================================

FROM node:20-alpine

WORKDIR /app

# Install dependencies first (cached layer)
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy source
COPY . .

EXPOSE 3000

CMD ["node", "index.js"]
