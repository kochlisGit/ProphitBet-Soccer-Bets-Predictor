import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jev Code Quality & Security Engine",
  description: "Continuous real-time code evaluation and automated security remediation with TypeSafe AI Jev via Vercel AI Gateway",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
