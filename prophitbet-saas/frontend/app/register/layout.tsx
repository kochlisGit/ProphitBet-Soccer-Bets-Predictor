import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Create Account — ProphitBet",
  description: "Create a free ProphitBet account and start getting AI-powered soccer predictions in minutes.",
};

export default function RegisterLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
