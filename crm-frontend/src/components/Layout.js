import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  AppBar,
  Box,
  CssBaseline,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Badge,
  Menu,
  MenuItem,
  Avatar,
  Chip,
} from "@mui/material";
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Campaign as CampaignIcon,
  Phone as PhoneIcon,
  Assessment as AssessmentIcon,
  Group as GroupIcon,
  AccountCircle,
  Logout as LogoutIcon,
  AdminPanelSettings as AdminPanelSettingsIcon,
} from "@mui/icons-material";
import { voiceAgentAPI } from "../services/api";
import { useAuth } from "../contexts/AuthContext";

const drawerWidth = 240;

function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout, isAdmin, isSuperAdmin } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeCalls, setActiveCalls] = useState(0);
  const [voiceAgentOnline, setVoiceAgentOnline] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);

  const menuItems = [
    // Superadmin only menu item
    ...(isSuperAdmin()
      ? [
          {
            text: "Super Admin",
            icon: <AdminPanelSettingsIcon />,
            path: "/superadmin",
            emoji: "🔐",
          },
        ]
      : []),
    // Regular menu items (not for superadmin)
    ...(!isSuperAdmin()
      ? [
          {
            text: "Dashboard",
            icon: <DashboardIcon />,
            path: "/dashboard",
            emoji: "📊",
          },
          { text: "Leads", icon: <PeopleIcon />, path: "/leads", emoji: "👥" },
          {
            text: "Campaigns",
            icon: <CampaignIcon />,
            path: "/campaigns",
            emoji: "🎯",
          },
          {
            text: "Call Sessions",
            icon: <PhoneIcon />,
            path: "/sessions",
            emoji: "📞",
          },
        ]
      : []),
    // Admin only menu item
    ...(isAdmin()
      ? [
          {
            text: "Agents",
            icon: <GroupIcon />,
            path: "/agents",
            emoji: "👨‍💼",
          },
        ]
      : []),
  ];

  useEffect(() => {
    // Check voice agent status
    const checkVoiceAgentStatus = async () => {
      try {
        await voiceAgentAPI.status();
        setVoiceAgentOnline(true);
      } catch (error) {
        setVoiceAgentOnline(false);
      }
    };

    // Check status immediately
    checkVoiceAgentStatus();

    // Check status every 10 seconds
    const interval = setInterval(checkVoiceAgentStatus, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleUserMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleUserMenuClose();
    logout();
    navigate("/login");
  };

  const drawer = (
    <div className="h-full">
      <Toolbar
        sx={{
          background: "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
          color: "white",
          backdropFilter: "blur(10px)",
          boxShadow: "0 4px 16px rgba(200, 92, 60, 0.2)",
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography variant="h6">🎙️</Typography>
          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{
              fontWeight: 700,
              letterSpacing: "-0.02em",
              textShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
            }}
          >
            Voice Agent CRM
          </Typography>
        </Box>
      </Toolbar>
      <Divider sx={{ borderColor: "rgba(200, 92, 60, 0.1)", opacity: 0.5 }} />
      <List sx={{ px: 1, py: 2 }}>
        {menuItems.map((item, index) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
              className="ios-button"
              sx={{
                borderRadius: 2,
                transition: "all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
                backdropFilter: "blur(10px)",
                "&.Mui-selected": {
                  background:
                    "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                  color: "white",
                  boxShadow: "0 4px 16px rgba(200, 92, 60, 0.3)",
                  "& .MuiListItemIcon-root": {
                    color: "white",
                  },
                  "&:hover": {
                    background:
                      "linear-gradient(135deg, #E07B5F 0%, #C85C3C 100%)",
                    boxShadow: "0 6px 20px rgba(200, 92, 60, 0.4)",
                  },
                },
                "&:hover": {
                  backgroundColor: "rgba(200, 92, 60, 0.08)",
                  transform: "translateX(6px) scale(1.02)",
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color:
                    location.pathname === item.path
                      ? "inherit"
                      : "primary.main",
                  minWidth: 40,
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                primaryTypographyProps={{ fontWeight: 500 }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        className="glass-effect"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          background:
            "linear-gradient(135deg, rgba(200, 92, 60, 0.95) 0%, rgba(160, 70, 42, 0.95) 100%)",
          backdropFilter: "blur(20px) saturate(180%)",
          WebkitBackdropFilter: "blur(20px) saturate(180%)",
          boxShadow: "0 8px 32px rgba(200, 92, 60, 0.25)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{
              mr: 2,
              display: { sm: "none" },
              "&:hover": {
                backgroundColor: "rgba(255, 255, 255, 0.15)",
                transform: "scale(1.1)",
              },
              transition: "all 0.3s ease",
            }}
          >
            <MenuIcon />
          </IconButton>
          <Box
            sx={{ display: "flex", alignItems: "center", gap: 1, flexGrow: 1 }}
          >
            <Typography variant="h4">
              {menuItems.find((item) => item.path === location.pathname)
                ?.emoji || "🎙️"}
            </Typography>
            <Typography
              variant="h4"
              noWrap
              component="div"
              sx={{
                fontWeight: 700,
                letterSpacing: "-0.02em",
              }}
            >
              {menuItems.find((item) => item.path === location.pathname)
                ?.text || "Voice Agent CRM"}
            </Typography>
          </Box>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 2,
            }}
          >
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                backgroundColor: voiceAgentOnline
                  ? "success.main"
                  : "error.main",
                animation: voiceAgentOnline
                  ? "pulse 2s ease-in-out infinite"
                  : "none",
                boxShadow: voiceAgentOnline
                  ? "0 0 12px rgba(107, 154, 90, 0.8)"
                  : "0 0 12px rgba(211, 47, 47, 0.8)",
              }}
            />

            {/* User Info */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Chip
                label={user?.role || "User"}
                size="small"
                sx={{
                  bgcolor: "rgba(255, 255, 255, 0.2)",
                  color: "white",
                  fontWeight: 600,
                  textTransform: "capitalize",
                }}
              />
              <IconButton
                onClick={handleUserMenuOpen}
                sx={{
                  color: "white",
                  "&:hover": {
                    backgroundColor: "rgba(255, 255, 255, 0.15)",
                  },
                }}
              >
                <AccountCircle />
              </IconButton>
            </Box>
          </Box>
        </Toolbar>
      </AppBar>

      {/* User Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleUserMenuClose}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="subtitle2" color="text.secondary">
            Signed in as
          </Typography>
          <Typography variant="body1" fontWeight={600}>
            {user?.full_name || user?.username}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {user?.email}
          </Typography>
        </Box>
        <Divider />
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <LogoutIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Logout</ListItemText>
        </MenuItem>
      </Menu>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: "block", sm: "none" },
            "& .MuiDrawer-paper": {
              boxSizing: "border-box",
              width: drawerWidth,
              background: "rgba(255, 255, 255, 0.85)",
              backdropFilter: "blur(20px) saturate(180%)",
              WebkitBackdropFilter: "blur(20px) saturate(180%)",
              borderRight: "1px solid rgba(200, 92, 60, 0.1)",
            },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: "none", sm: "block" },
            "& .MuiDrawer-paper": {
              boxSizing: "border-box",
              width: drawerWidth,
              background: "rgba(255, 255, 255, 0.85)",
              backdropFilter: "blur(20px) saturate(180%)",
              WebkitBackdropFilter: "blur(20px) saturate(180%)",
              borderRight: "1px solid rgba(200, 92, 60, 0.1)",
              boxShadow: "4px 0 24px rgba(200, 92, 60, 0.08)",
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

export default Layout;
