# syntax=docker/dockerfile:1

# Build the static app with Node, then serve it with nginx, which also proxies
# /api to the backend so the browser sees a single origin.
FROM node:24-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
