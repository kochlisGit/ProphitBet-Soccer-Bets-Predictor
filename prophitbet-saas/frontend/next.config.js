/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001",
  },
  async rewrites() {
    const backendUrl = process.env.INTERNAL_API_URL || "http://backend:8000";
    return [
      {
        source: "/api/proxy/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
  async headers() {
    const isDev = process.env.NODE_ENV !== "production";
    const scriptSrc = isDev ? "'self' 'unsafe-inline' 'unsafe-eval'" : "'self' 'unsafe-inline'";
    return [{
      source: "/:path*",
      headers: [
        { key: "Content-Security-Policy", value: `default-src 'self'; script-src ${scriptSrc}; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; font-src 'self' data:; connect-src 'self' http://localhost:* http://127.0.0.1:* https: ws: wss:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'` },
        { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "X-Frame-Options", value: "DENY" },
      ],
    }];
  },
};

module.exports = nextConfig;
