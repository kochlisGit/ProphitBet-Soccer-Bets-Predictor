FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Verify production compile (dev server does not require this at runtime)
RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]
