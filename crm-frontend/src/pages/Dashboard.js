import React, { useState, useEffect } from "react";
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  IconButton,
  Button,
  Container,
} from "@mui/material";
import SubscriptionBanner from "../components/SubscriptionBanner";
import { DashboardSkeleton } from "../components/LoadingSkeleton";
import {
  People as PeopleIcon,
  Campaign as CampaignIcon,
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
} from "@mui/icons-material";
import { Line, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from "chart.js";
import {
  leadAPI,
  campaignAPI,
  sessionAPI,
  voiceAgentAPI,
} from "../services/api";
import { formatDateTime } from "../utils/dateUtils";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalLeads: 0,
    activeCampaigns: 0,
    totalCalls: 0,
    answeredCalls: 0,
    rejectedCalls: 0,
  });
  const [recentSessions, setRecentSessions] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // Load stats
      const [leadsRes, campaignsRes, sessionsRes, healthRes] =
        await Promise.all([
          leadAPI.getAll({ per_page: 1 }),
          campaignAPI.getAll(),
          sessionAPI.getAll({ limit: 10 }),
          voiceAgentAPI.health(),
        ]);

      // Calculate stats
      const campaigns = campaignsRes.data;
      const sessions = sessionsRes.data;

      setStats({
        totalLeads: leadsRes.data.total || 0,
        activeCampaigns: campaigns.filter((c) => c.status === "running").length,
        totalCalls: sessions.length,
        answeredCalls: sessions.filter((s) => s.status === "answered").length,
        rejectedCalls: sessions.filter((s) => s.status === "rejected").length,
      });

      setRecentSessions(sessions);
      setSystemHealth(healthRes.data);
    } catch (error) {
      console.error("Error loading dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  const callSuccessData = {
    labels: ["Answered", "Rejected", "No Answer", "Failed"],
    datasets: [
      {
        data: [
          stats.answeredCalls,
          stats.rejectedCalls,
          stats.totalCalls - stats.answeredCalls - stats.rejectedCalls,
          0,
        ],
        backgroundColor: ["#4caf50", "#f44336", "#ff9800", "#9e9e9e"],
        borderWidth: 0,
      },
    ],
  };

  const callTrendData = {
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    datasets: [
      {
        label: "Calls Made",
        data: [12, 19, 15, 25, 22, 30, 18],
        borderColor: "rgb(75, 192, 192)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        tension: 0.4,
      },
      {
        label: "Successful Calls",
        data: [8, 12, 10, 18, 15, 22, 12],
        borderColor: "rgb(54, 162, 235)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        tension: 0.4,
      },
    ],
  };

  const StatCard = ({ title, value, icon, color, index }) => (
    <Card
      className="glass-effect ios-card hover-lift"
      sx={{
        animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
        animationDelay: `${index * 0.1}s`,
        animationFillMode: "both",
        position: "relative",
        overflow: "hidden",
        "&::before": {
          content: '""',
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `linear-gradient(135deg, ${
            color === "primary"
              ? "rgba(200, 92, 60, 0.05)"
              : color === "secondary"
              ? "rgba(139, 94, 60, 0.05)"
              : color === "info"
              ? "rgba(92, 138, 166, 0.05)"
              : color === "success"
              ? "rgba(107, 154, 90, 0.05)"
              : "rgba(200, 92, 60, 0.05)"
          }, transparent)`,
          transform: "translateX(-100%)",
          transition: "transform 0.5s ease",
        },
        "&:hover": {
          "& .stat-icon": {
            transform: "rotate(10deg) scale(1.15)",
            animation: "heartbeat 1s ease-in-out",
          },
          "&::before": {
            transform: "translateX(100%)",
          },
          "& .stat-value": {
            transform: "scale(1.1)",
          },
        },
      }}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box sx={{ zIndex: 1 }}>
            <Typography
              color="textSecondary"
              gutterBottom
              fontWeight={600}
              fontSize="0.875rem"
              sx={{
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              {title}
            </Typography>
            <Typography
              className="stat-value"
              variant="h4"
              fontWeight={700}
              sx={{
                background: `linear-gradient(135deg, ${
                  color === "primary"
                    ? "#C85C3C"
                    : color === "secondary"
                    ? "#8B5E3C"
                    : color === "info"
                    ? "#5C8AA6"
                    : color === "success"
                    ? "#6B9A5A"
                    : "#C85C3C"
                } 0%, ${
                  color === "primary"
                    ? "#A0462A"
                    : color === "secondary"
                    ? "#6B4423"
                    : color === "info"
                    ? "#426D87"
                    : color === "success"
                    ? "#517542"
                    : "#A0462A"
                } 100%)`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
                transition: "transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
              }}
            >
              {value}
            </Typography>
          </Box>
          <Box
            className="stat-icon glass-effect-colored"
            sx={{
              borderRadius: "50%",
              p: 2,
              display: "flex",
              background: `linear-gradient(135deg, ${
                color === "primary"
                  ? "rgba(200, 92, 60, 0.15)"
                  : color === "secondary"
                  ? "rgba(139, 94, 60, 0.15)"
                  : color === "info"
                  ? "rgba(92, 138, 166, 0.15)"
                  : color === "success"
                  ? "rgba(107, 154, 90, 0.15)"
                  : "rgba(200, 92, 60, 0.15)"
              } 0%, transparent 100%)`,
              boxShadow: `0 8px 24px ${
                color === "primary"
                  ? "rgba(200, 92, 60, 0.25)"
                  : color === "secondary"
                  ? "rgba(139, 94, 60, 0.25)"
                  : color === "info"
                  ? "rgba(92, 138, 166, 0.25)"
                  : color === "success"
                  ? "rgba(107, 154, 90, 0.25)"
                  : "rgba(200, 92, 60, 0.25)"
              }`,
              transition: "all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
              zIndex: 1,
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <DashboardSkeleton />
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <SubscriptionBanner />
      <Box className="fade-in">
        {/* System Status */}
        <Paper
          className="glass-effect-colored ios-blur-container hover-glow gradient-border"
          sx={{
            p: 3,
            mb: 4,
            animation: "slideUp 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)",
            position: "relative",
            overflow: "hidden",
            "&::after": {
              content: '""',
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background:
                "linear-gradient(135deg, rgba(200, 92, 60, 0.05) 0%, transparent 50%, rgba(92, 138, 166, 0.05) 100%)",
              pointerEvents: "none",
            },
          }}
        >
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            mb={2}
            sx={{ position: "relative", zIndex: 1 }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Typography variant="h5" className="wave">
                ⚡
              </Typography>
              <Typography variant="h6" fontWeight={700}>
                System Status
              </Typography>
            </Box>
            <IconButton
              onClick={loadDashboardData}
              className="ripple-container hover-scale"
              size="small"
              sx={{
                background: "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                color: "white",
                boxShadow: "0 4px 16px rgba(200, 92, 60, 0.3)",
                "&:hover": {
                  background:
                    "linear-gradient(135deg, #E07B5F 0%, #C85C3C 100%)",
                  transform: "rotate(360deg) scale(1.1)",
                  boxShadow: "0 8px 24px rgba(200, 92, 60, 0.5)",
                },
                transition: "all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
              }}
            >
              <RefreshIcon />
            </IconButton>
          </Box>
          <Grid
            container
            spacing={2}
            alignItems="center"
            sx={{ position: "relative", zIndex: 1 }}
          >
            <Grid item xs={12} md={4}>
              <Box
                className="scale-in"
                sx={{
                  display: "flex",
                  alignItems: "center",
                  p: 1.5,
                  borderRadius: 2,
                  background: "rgba(255, 255, 255, 0.5)",
                  backdropFilter: "blur(10px)",
                }}
              >
                <Box
                  sx={{
                    width: 16,
                    height: 16,
                    borderRadius: "50%",
                    backgroundColor:
                      systemHealth?.status === "healthy"
                        ? "success.main"
                        : "error.main",
                    mr: 1.5,
                    animation: "pulse 2s ease-in-out infinite",
                    boxShadow:
                      systemHealth?.status === "healthy"
                        ? "0 0 20px rgba(107, 154, 90, 0.8)"
                        : "0 0 20px rgba(199, 84, 80, 0.8)",
                  }}
                />
                <Typography fontWeight={600}>
                  Voice Agent:{" "}
                  <Box
                    component="span"
                    sx={{
                      color:
                        systemHealth?.status === "healthy"
                          ? "success.main"
                          : "error.main",
                      fontWeight: 700,
                    }}
                  >
                    {systemHealth?.status === "healthy" ? "Online" : "Offline"}
                  </Box>
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box
                className="scale-in stagger-1"
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  p: 1.5,
                  borderRadius: 2,
                  background: "rgba(255, 255, 255, 0.5)",
                  backdropFilter: "blur(10px)",
                }}
              >
                <Typography variant="h6">📞</Typography>
                <Typography fontWeight={600}>
                  Active Sessions:{" "}
                  <Box
                    component="span"
                    sx={{ color: "info.main", fontWeight: 700 }}
                  >
                    {systemHealth?.active_sessions || 0}
                  </Box>
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box
                className="scale-in stagger-2"
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  p: 1.5,
                  borderRadius: 2,
                  background: "rgba(255, 255, 255, 0.5)",
                  backdropFilter: "blur(10px)",
                }}
              >
                <Typography variant="h6">📱</Typography>
                <Typography fontWeight={600}>
                  Phone:{" "}
                  <Box
                    component="span"
                    sx={{ color: "primary.main", fontWeight: 700 }}
                  >
                    {systemHealth?.phone_number || "Not configured"}
                  </Box>
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Stats Cards */}
        <Grid container spacing={3} mb={3}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Total Leads"
              value={stats.totalLeads}
              icon={<PeopleIcon color="primary" sx={{ fontSize: 40 }} />}
              color="primary"
              index={0}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Active Campaigns"
              value={stats.activeCampaigns}
              icon={<CampaignIcon color="secondary" sx={{ fontSize: 40 }} />}
              color="secondary"
              index={1}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Total Calls"
              value={stats.totalCalls}
              icon={<PhoneIcon color="info" sx={{ fontSize: 40 }} />}
              color="info"
              index={2}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Success Rate"
              value={
                stats.totalCalls > 0
                  ? `${Math.round(
                      (stats.answeredCalls / stats.totalCalls) * 100
                    )}%`
                  : "0%"
              }
              icon={<CheckCircleIcon color="success" sx={{ fontSize: 40 }} />}
              color="success"
              index={3}
            />
          </Grid>
        </Grid>

        {/* Charts */}
        <Grid container spacing={3} mb={4}>
          <Grid item xs={12} md={8}>
            <Paper
              className="glass-effect ios-blur-container hover-lift"
              sx={{
                p: 3,
                animation: "springIn 0.7s cubic-bezier(0.34, 1.56, 0.64, 1)",
                animationDelay: "0.4s",
                animationFillMode: "both",
                position: "relative",
                overflow: "hidden",
                "&::before": {
                  content: '""',
                  position: "absolute",
                  top: -50,
                  right: -50,
                  width: 100,
                  height: 100,
                  background:
                    "radial-gradient(circle, rgba(200, 92, 60, 0.1) 0%, transparent 70%)",
                  borderRadius: "50%",
                  pointerEvents: "none",
                },
              }}
            >
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}
              >
                <TrendingUpIcon sx={{ color: "primary.main", fontSize: 28 }} />
                <Typography variant="h6" fontWeight={700}>
                  Call Trends (Last 7 Days)
                </Typography>
              </Box>
              <Box height={300}>
                <Line
                  data={callTrendData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: "top",
                      },
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper
              className="glass-effect ios-blur-container hover-lift"
              sx={{
                p: 3,
                animation: "springIn 0.7s cubic-bezier(0.34, 1.56, 0.64, 1)",
                animationDelay: "0.5s",
                animationFillMode: "both",
                position: "relative",
                overflow: "hidden",
                "&::before": {
                  content: '""',
                  position: "absolute",
                  top: -50,
                  left: -50,
                  width: 100,
                  height: 100,
                  background:
                    "radial-gradient(circle, rgba(92, 138, 166, 0.1) 0%, transparent 70%)",
                  borderRadius: "50%",
                  pointerEvents: "none",
                },
              }}
            >
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}
              >
                <Typography variant="h5" className="rotate-in">
                  🎯
                </Typography>
                <Typography variant="h6" fontWeight={700}>
                  Call Outcomes
                </Typography>
              </Box>
              <Box
                height={300}
                display="flex"
                justifyContent="center"
                alignItems="center"
              >
                <Doughnut
                  data={callSuccessData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: "bottom",
                      },
                    },
                  }}
                />
              </Box>
            </Paper>
          </Grid>
        </Grid>

        {/* Recent Sessions */}
        <Paper
          className="glass-effect ios-blur-container hover-lift"
          sx={{
            p: 3,
            animation: "slideUp 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
            animationDelay: "0.6s",
            animationFillMode: "both",
            position: "relative",
            overflow: "hidden",
            "&::before": {
              content: '""',
              position: "absolute",
              top: -50,
              left: "50%",
              width: 100,
              height: 100,
              background:
                "radial-gradient(circle, rgba(107, 154, 90, 0.1) 0%, transparent 70%)",
              borderRadius: "50%",
              transform: "translateX(-50%)",
              pointerEvents: "none",
            },
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 3 }}>
            <PhoneIcon sx={{ color: "info.main", fontSize: 28 }} />
            <Typography variant="h6" fontWeight={700}>
              Recent Call Sessions
            </Typography>
          </Box>
          {recentSessions.length > 0 ? (
            <Box>
              {recentSessions.map((session, index) => (
                <Box
                  key={session.id}
                  className="fade-in"
                  sx={{
                    p: 2.5,
                    borderBottom: "1px solid",
                    borderColor: "divider",
                    borderRadius: 2,
                    mb: 1,
                    transition: "all 0.3s ease",
                    animation: "slideInLeft 0.5s ease-out",
                    animationDelay: `${index * 0.05}s`,
                    animationFillMode: "both",
                    "&:last-child": { borderBottom: 0, mb: 0 },
                    "&:hover": {
                      backgroundColor: "rgba(200, 92, 60, 0.03)",
                      transform: "translateX(8px)",
                      boxShadow: "-4px 0 0 0 rgba(200, 92, 60, 0.3)",
                    },
                  }}
                >
                  <Grid container alignItems="center" spacing={2}>
                    <Grid item xs={12} md={3}>
                      <Typography
                        variant="body2"
                        color="textSecondary"
                        fontWeight={500}
                      >
                        {formatDateTime(session.started_at)}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Typography fontWeight={600}>
                        {session.called_number}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <Box display="flex" alignItems="center">
                        {session.status === "answered" ? (
                          <CheckCircleIcon
                            color="success"
                            fontSize="small"
                            className="heartbeat"
                            sx={{ mr: 1 }}
                          />
                        ) : (
                          <CancelIcon
                            color="error"
                            fontSize="small"
                            sx={{ mr: 1 }}
                          />
                        )}
                        <Typography variant="body2" fontWeight={500}>
                          {session.status.charAt(0).toUpperCase() +
                            session.status.slice(1)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <Typography variant="body2" fontWeight={500}>
                        Duration:{" "}
                        {session.duration ? `${session.duration}s` : "-"}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <Button
                        size="small"
                        variant="outlined"
                        href={`/sessions/${session.session_id}`}
                        className="ripple-container hover-scale"
                        sx={{
                          borderRadius: 10,
                          fontWeight: 600,
                          borderWidth: 1.5,
                          "&:hover": {
                            backgroundColor: "rgba(200, 92, 60, 0.06)",
                            borderWidth: 1.5,
                          },
                        }}
                      >
                        View Details
                      </Button>
                    </Grid>
                  </Grid>
                </Box>
              ))}
            </Box>
          ) : (
            <Box
              sx={{
                textAlign: "center",
                py: 4,
                animation: "fadeIn 0.5s ease-out",
              }}
            >
              <Typography variant="h3" sx={{ mb: 1, opacity: 0.3 }}>
                📞
              </Typography>
              <Typography color="textSecondary">No recent sessions</Typography>
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

export default Dashboard;
