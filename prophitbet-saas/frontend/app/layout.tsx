import type { Metadata } from "next";
import { DM_Sans } from "next/font/google";
import { AuthProvider } from "@/lib/auth";
import BackNavigation from "@/components/BackNavigation";
import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "ProphitBet — Soccer predictions & analysis",
  description:
    "Match predictions across 36+ leagues, statistical analysis, and custom models—built for people who read the numbers.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${dmSans.variable}`}>
      <body className="min-h-screen antialiased font-sans text-zinc-50 bg-zinc-900">
        <AuthProvider><BackNavigation />{children}</AuthProvider>
      </body>
    </html>
  );
}
