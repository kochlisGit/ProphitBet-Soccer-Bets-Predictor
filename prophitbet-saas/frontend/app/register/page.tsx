"use client";

import { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import Navbar from "@/components/Navbar";
import MotionPageRoot from "@/components/MotionPageRoot";

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    setLoading(true);
    try {
      await register(email, password, name);
      router.push("/dashboard");
    } catch (err: any) {
      setError(err.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-900">
      <Navbar />
      <MotionPageRoot
        reviveKey={pathname}
        className="flex items-center justify-center pt-20 pb-12 px-4 w-full"
      >
        <div className="anim-reveal-section w-full max-w-md">
          <div className="anim-reveal-inner card w-full">
            <h1 className="text-2xl font-bold text-zinc-50 mb-2">Create your account</h1>
            <p className="text-zinc-400 mb-8">Start getting AI predictions in minutes</p>

            {error && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-1">Name</label>
                <input
                  type="text"
                  className="input"
                  placeholder="John Doe"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-1">Email</label>
                <input
                  type="email"
                  className="input"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-1">Password</label>
                <input
                  type="password"
                  className="input"
                  placeholder="At least 8 characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={8}
                />
              </div>
              <button type="submit" disabled={loading} className="btn-primary w-full disabled:opacity-50">
                {loading ? "Creating account..." : "Create account"}
              </button>
            </form>

            <p className="mt-6 text-center text-sm text-zinc-400">
              Already have an account?{" "}
              <Link href="/login" className="text-brand-400 hover:text-brand-300 font-medium">
                Sign in
              </Link>
            </p>
          </div>
        </div>
      </MotionPageRoot>
    </div>
  );
}
