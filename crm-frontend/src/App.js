import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDateFns } from "@mui/x-date-pickers/AdapterDateFns";

import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Leads from "./pages/Leads";
import Campaigns from "./pages/Campaigns";
import CampaignDetail from "./pages/CampaignDetail";
import Sessions from "./pages/Sessions";
import SessionDetail from "./pages/SessionDetail";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#C85C3C", // Warm cinnamon
      light: "#E07B5F",
      dark: "#A0462A",
      contrastText: "#FFFFFF",
    },
    secondary: {
      main: "#8B5E3C", // Deep brown
      light: "#A67C52",
      dark: "#6B4423",
      contrastText: "#FFFFFF",
    },
    success: {
      main: "#6B9A5A", // Sage green
      light: "#8BB876",
      dark: "#517542",
    },
    info: {
      main: "#5C8AA6", // Dusty blue
      light: "#7BA5BF",
      dark: "#426D87",
    },
    warning: {
      main: "#D99C5E", // Amber
      light: "#E7B680",
      dark: "#B57E42",
    },
    error: {
      main: "#C75450", // Muted red
      light: "#D97874",
      dark: "#A43D3A",
    },
    background: {
      default: "#F8F3EF", // Cream background
      paper: "rgba(255, 255, 255, 0.85)",
    },
    text: {
      primary: "#3A2E2A",
      secondary: "#6B5B54",
    },
  },
  typography: {
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif',
    h1: {
      fontWeight: 800,
      letterSpacing: "-0.03em",
      fontSize: "3rem",
    },
    h2: {
      fontWeight: 700,
      letterSpacing: "-0.025em",
      fontSize: "2.5rem",
    },
    h3: {
      fontWeight: 700,
      letterSpacing: "-0.025em",
      fontSize: "2rem",
    },
    h4: {
      fontWeight: 700,
      letterSpacing: "-0.02em",
      fontSize: "1.75rem",
    },
    h5: {
      fontWeight: 600,
      letterSpacing: "-0.015em",
      fontSize: "1.5rem",
    },
    h6: {
      fontWeight: 600,
      letterSpacing: "-0.01em",
      fontSize: "1.25rem",
    },
    body1: {
      fontSize: "1rem",
      lineHeight: 1.6,
      letterSpacing: "-0.005em",
    },
    body2: {
      fontSize: "0.875rem",
      lineHeight: 1.5,
      letterSpacing: "-0.003em",
    },
    button: {
      textTransform: "none",
      fontWeight: 600,
      letterSpacing: "-0.01em",
    },
  },
  shape: {
    borderRadius: 16,
  },
  shadows: [
    "none",
    "0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.08)",
    "0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04)",
    "0 4px 16px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.04)",
    "0 8px 24px rgba(0, 0, 0, 0.08), 0 3px 8px rgba(0, 0, 0, 0.04)",
    "0 12px 32px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.04)",
    "0 16px 40px rgba(0, 0, 0, 0.08), 0 6px 16px rgba(0, 0, 0, 0.04)",
    "0 20px 48px rgba(0, 0, 0, 0.08), 0 8px 20px rgba(0, 0, 0, 0.04)",
    "0 24px 56px rgba(0, 0, 0, 0.08), 0 10px 24px rgba(0, 0, 0, 0.04)",
    "0 32px 72px rgba(0, 0, 0, 0.08), 0 12px 28px rgba(0, 0, 0, 0.04)",
    ...Array(15).fill("0 0 0 rgba(0,0,0,0)"),
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
          borderRadius: 12,
          padding: "12px 24px",
          transition: "all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)",
          backdropFilter: "blur(10px)",
          "&:hover": {
            transform: "translateY(-2px) scale(1.02)",
            boxShadow: "0 10px 30px rgba(0, 0, 0, 0.12)",
          },
          "&:active": {
            transform: "scale(0.98)",
          },
        },
        contained: {
          backgroundImage:
            "linear-gradient(135deg, var(--mui-palette-primary-main) 0%, var(--mui-palette-primary-dark) 100%)",
          boxShadow: "0 4px 16px rgba(0, 0, 0, 0.08)",
          "&:hover": {
            boxShadow: "0 8px 24px rgba(0, 0, 0, 0.12)",
          },
        },
        outlined: {
          borderWidth: 1.5,
          "&:hover": {
            borderWidth: 1.5,
            backgroundColor: "rgba(200, 92, 60, 0.04)",
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          background: "rgba(255, 255, 255, 0.72)",
          backdropFilter: "blur(20px) saturate(180%)",
          WebkitBackdropFilter: "blur(20px) saturate(180%)",
          border: "1px solid rgba(255, 255, 255, 0.3)",
          boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.07)",
          transition: "all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
          "&:hover": {
            transform: "translateY(-8px) scale(1.01)",
            boxShadow: "0 20px 40px rgba(0, 0, 0, 0.08)",
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          borderRadius: 20,
          background: "rgba(255, 255, 255, 0.72)",
          backdropFilter: "blur(20px) saturate(180%)",
          WebkitBackdropFilter: "blur(20px) saturate(180%)",
          border: "1px solid rgba(255, 255, 255, 0.3)",
          boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.07)",
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          letterSpacing: "-0.01em",
          borderRadius: 10,
          backdropFilter: "blur(10px)",
          border: "1px solid rgba(255, 255, 255, 0.3)",
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": {
            borderRadius: 12,
            backgroundColor: "rgba(255, 255, 255, 0.5)",
            backdropFilter: "blur(10px)",
            transition: "all 0.2s ease",
            "&:hover": {
              backgroundColor: "rgba(255, 255, 255, 0.7)",
            },
            "&.Mui-focused": {
              backgroundColor: "rgba(255, 255, 255, 0.9)",
              boxShadow: "0 0 0 3px rgba(200, 92, 60, 0.1)",
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 24,
          background: "rgba(255, 255, 255, 0.95)",
          backdropFilter: "blur(20px) saturate(180%)",
          WebkitBackdropFilter: "blur(20px) saturate(180%)",
          boxShadow: "0 20px 60px rgba(0, 0, 0, 0.15)",
        },
      },
    },
    MuiDataGrid: {
      styleOverrides: {
        root: {
          border: "none",
          borderRadius: 16,
          "& .MuiDataGrid-cell": {
            borderBottom: "1px solid rgba(224, 224, 224, 0.3)",
          },
          "& .MuiDataGrid-columnHeaders": {
            backgroundColor: "rgba(248, 243, 239, 0.5)",
            backdropFilter: "blur(10px)",
            borderBottom: "2px solid rgba(200, 92, 60, 0.1)",
          },
          "& .MuiDataGrid-footerContainer": {
            borderTop: "2px solid rgba(200, 92, 60, 0.1)",
            backgroundColor: "rgba(248, 243, 239, 0.5)",
            backdropFilter: "blur(10px)",
          },
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/leads" element={<Leads />} />
              <Route path="/campaigns" element={<Campaigns />} />
              <Route path="/campaigns/:id" element={<CampaignDetail />} />
              <Route path="/sessions" element={<Sessions />} />
              <Route path="/sessions/:id" element={<SessionDetail />} />
            </Routes>
          </Layout>
        </Router>
      </LocalizationProvider>
    </ThemeProvider>
  );
}

export default App;
