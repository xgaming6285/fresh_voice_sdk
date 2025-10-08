import React, { createContext, useState, useContext, useEffect } from "react";
import api from "../services/api";

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(localStorage.getItem("token"));

  useEffect(() => {
    // Set token in API service
    if (token) {
      api.setAuthToken(token);
      // Verify token and load user data
      loadUser();
    } else {
      setLoading(false);
    }
  }, [token]);

  const loadUser = async () => {
    try {
      const response = await api.get("/api/auth/me");
      setUser(response.data);
    } catch (error) {
      console.error("Failed to load user:", error);
      // Token invalid, clear it
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (username, password) => {
    try {
      const response = await api.post("/api/auth/login", {
        username,
        password,
      });
      const { access_token, user: userData } = response.data;

      localStorage.setItem("token", access_token);
      setToken(access_token);
      setUser(userData);
      api.setAuthToken(access_token);

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || "Login failed",
      };
    }
  };

  const register = async (userData) => {
    try {
      const response = await api.post("/api/auth/register", userData);
      const { access_token, user: newUser } = response.data;

      localStorage.setItem("token", access_token);
      setToken(access_token);
      setUser(newUser);
      api.setAuthToken(access_token);

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || "Registration failed",
      };
    }
  };

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUser(null);
    api.setAuthToken(null);
  };

  const isAdmin = () => {
    return user?.role === "admin";
  };

  const isAgent = () => {
    return user?.role === "agent";
  };

  const isSuperAdmin = () => {
    return user?.role === "superadmin";
  };

  const value = {
    user,
    loading,
    token,
    login,
    register,
    logout,
    isAdmin,
    isAgent,
    isSuperAdmin,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;
