"use client";

import { createContext, useContext, useEffect, useState, useCallback, ReactNode } from "react";
import { authApi, setAccessTokenRefreshHandler } from "./api";

interface UserPreferences {
  notify_daily_digest: boolean;
  notify_training: boolean;
  notify_weekly_report: boolean;
}

interface User {
  id: string;
  email: string;
  name: string;
  avatar_url: string | null;
  provider: string;
  plan: string;
  is_admin: boolean;
  preferences: UserPreferences;
}

interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthState>({
  user: null,
  token: null,
  loading: true,
  login: async () => {},
  register: async () => {},
  logout: async () => {},
  refreshUser: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadUser = useCallback(async (accessToken: string) => {
    try {
      const userData = await authApi.me(accessToken);
      setUser(userData);
      setToken(accessToken);
    } catch {
      setUser(null);
      setToken(null);
    }
  }, []);

  useEffect(() => {
    authApi.refresh()
      .then((res) => loadUser(res.access_token))
      .catch(() => { setUser(null); setToken(null); })
      .finally(() => setLoading(false));
  }, [loadUser]);

  useEffect(() => {
    setAccessTokenRefreshHandler((accessToken) => {
      setToken(accessToken);
      authApi.me(accessToken).then(setUser).catch(() => {});
    });
    return () => setAccessTokenRefreshHandler(null);
  }, []);

  const REFRESH_MS = 25 * 60 * 1000;
  useEffect(() => {
    if (!token) return;
    const id = window.setInterval(async () => {
      try {
        const res = await authApi.refresh();
        setToken(res.access_token);
        const userData = await authApi.me(res.access_token);
        setUser(userData);
      } catch {
        /* network / expired refresh — next API call will 401 and clear if needed */
      }
    }, REFRESH_MS);
    return () => window.clearInterval(id);
  }, [token]);

  const login = async (email: string, password: string) => {
    const res = await authApi.login(email, password);
    await loadUser(res.access_token);
  };

  const register = async (email: string, password: string, name: string) => {
    const res = await authApi.register(email, password, name);
    await loadUser(res.access_token);
  };

  const logout = async () => {
    try {
      await authApi.logout();
    } finally {
    setUser(null);
    setToken(null);
    }
  };

  const refreshUser = useCallback(async () => {
    if (token) await loadUser(token);
  }, [loadUser, token]);

  return (
    <AuthContext.Provider
      value={{ user, token, loading, login, register, logout, refreshUser }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
