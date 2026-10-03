import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Sign In — ProphitBet",
  description: "Sign in to your ProphitBet account to access AI soccer predictions and analysis tools.",
};

export default function LoginLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
