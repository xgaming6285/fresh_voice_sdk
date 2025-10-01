import React, { useState } from "react";
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
} from "@mui/material";
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Campaign as CampaignIcon,
  Phone as PhoneIcon,
  Assessment as AssessmentIcon,
} from "@mui/icons-material";

const drawerWidth = 240;

const menuItems = [
  { text: "Dashboard", icon: <DashboardIcon />, path: "/dashboard" },
  { text: "Leads", icon: <PeopleIcon />, path: "/leads" },
  { text: "Campaigns", icon: <CampaignIcon />, path: "/campaigns" },
  { text: "Call Sessions", icon: <PhoneIcon />, path: "/sessions" },
];

function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeCalls, setActiveCalls] = useState(0);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
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
          🎙️ Voice Agent CRM
        </Typography>
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
      <Divider sx={{ borderColor: "rgba(200, 92, 60, 0.1)" }} />
      <Box
        className="glass-effect-colored"
        sx={{
          p: 2,
          m: 2,
          borderRadius: 3,
          animation: "springIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
          animationDelay: "0.5s",
          animationFillMode: "both",
        }}
      >
        <Typography variant="body2" color="text.secondary" fontWeight={600}>
          Voice Agent Status
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", mt: 1 }}>
          <Box
            sx={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              backgroundColor: "success.main",
              mr: 1,
              animation: "pulse 2s ease-in-out infinite",
              boxShadow: "0 0 10px rgba(107, 154, 90, 0.6)",
            }}
          />
          <Typography variant="body2" fontWeight={500}>
            Online
          </Typography>
        </Box>
        {activeCalls > 0 && (
          <Box
            sx={{
              mt: 1,
              p: 1,
              borderRadius: 1.5,
              backgroundColor: "error.light",
              color: "white",
              animation: "pulse 1.5s ease-in-out infinite",
            }}
          >
            <Typography variant="body2" fontWeight={600}>
              📞 Active Calls: {activeCalls}
            </Typography>
          </Box>
        )}
      </Box>
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
          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{
              flexGrow: 1,
              fontWeight: 700,
              letterSpacing: "-0.02em",
            }}
          >
            {menuItems.find((item) => item.path === location.pathname)?.text ||
              "Voice Agent CRM"}
          </Typography>
          <IconButton
            color="inherit"
            sx={{
              "&:hover": {
                backgroundColor: "rgba(255, 255, 255, 0.15)",
                transform: "scale(1.1) rotate(10deg)",
              },
              transition: "all 0.3s ease",
            }}
          >
            <Badge
              badgeContent={activeCalls}
              color="error"
              sx={{
                "& .MuiBadge-badge": {
                  animation:
                    activeCalls > 0
                      ? "pulse 1.5s ease-in-out infinite"
                      : "none",
                },
              }}
            >
              <PhoneIcon />
            </Badge>
          </IconButton>
        </Toolbar>
      </AppBar>
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
