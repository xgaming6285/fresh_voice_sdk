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
      paper: "#FFFFFF",
    },
    text: {
      primary: "#3A2E2A",
      secondary: "#6B5B54",
    },
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: "-0.02em",
    },
    h6: {
      fontWeight: 600,
      letterSpacing: "-0.01em",
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    "none",
    "0 2px 8px rgba(184, 92, 56, 0.08)",
    "0 4px 16px rgba(184, 92, 56, 0.12)",
    "0 8px 24px rgba(184, 92, 56, 0.16)",
    "0 12px 32px rgba(184, 92, 56, 0.20)",
    "0 16px 40px rgba(184, 92, 56, 0.24)",
    "0 20px 48px rgba(184, 92, 56, 0.28)",
    "0 24px 56px rgba(184, 92, 56, 0.32)",
    "0 28px 64px rgba(184, 92, 56, 0.36)",
    "0 32px 72px rgba(184, 92, 56, 0.40)",
    ...Array(15).fill("0 0 0 rgba(0,0,0,0)"),
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
          borderRadius: 10,
          padding: "10px 24px",
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": {
            transform: "translateY(-2px)",
            boxShadow: "0 8px 24px rgba(184, 92, 56, 0.3)",
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": {
            transform: "translateY(-4px)",
            boxShadow: "0 12px 32px rgba(184, 92, 56, 0.2)",
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          letterSpacing: "0.5px",
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
