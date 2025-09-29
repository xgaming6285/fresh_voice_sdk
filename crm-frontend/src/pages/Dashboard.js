import React, { useState, useEffect } from "react";
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  IconButton,
  Button,
} from "@mui/material";
import {
  People as PeopleIcon,
  Campaign as CampaignIcon,
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
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

  const StatCard = ({ title, value, icon, color }) => (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography color="textSecondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4">{value}</Typography>
          </Box>
          <Box
            sx={{
              backgroundColor: `${color}.light`,
              borderRadius: "50%",
              p: 1,
              display: "flex",
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
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="400px"
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={3}
      >
        <Typography variant="h4">Dashboard</Typography>
        <IconButton onClick={loadDashboardData}>
          <RefreshIcon />
        </IconButton>
      </Box>

      {/* System Status */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          System Status
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <Box display="flex" alignItems="center">
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  backgroundColor:
                    systemHealth?.status === "healthy"
                      ? "success.main"
                      : "error.main",
                  mr: 1,
                }}
              />
              <Typography>
                Voice Agent:{" "}
                {systemHealth?.status === "healthy" ? "Online" : "Offline"}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              Active Sessions: {systemHealth?.active_sessions || 0}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              Phone: {systemHealth?.phone_number || "Not configured"}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Stats Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Leads"
            value={stats.totalLeads}
            icon={<PeopleIcon color="primary" />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Active Campaigns"
            value={stats.activeCampaigns}
            icon={<CampaignIcon color="secondary" />}
            color="secondary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Calls"
            value={stats.totalCalls}
            icon={<PhoneIcon color="info" />}
            color="info"
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
            icon={<CheckCircleIcon color="success" />}
            color="success"
          />
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Call Trends (Last 7 Days)
            </Typography>
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
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Call Outcomes
            </Typography>
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
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Recent Call Sessions
        </Typography>
        {recentSessions.length > 0 ? (
          <Box>
            {recentSessions.map((session) => (
              <Box
                key={session.id}
                sx={{
                  p: 2,
                  borderBottom: "1px solid",
                  borderColor: "divider",
                  "&:last-child": { borderBottom: 0 },
                }}
              >
                <Grid container alignItems="center" spacing={2}>
                  <Grid item xs={12} md={3}>
                    <Typography variant="body2" color="textSecondary">
                      {new Date(session.started_at).toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Typography>{session.called_number}</Typography>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Box display="flex" alignItems="center">
                      {session.status === "answered" ? (
                        <CheckCircleIcon
                          color="success"
                          fontSize="small"
                          sx={{ mr: 1 }}
                        />
                      ) : (
                        <CancelIcon
                          color="error"
                          fontSize="small"
                          sx={{ mr: 1 }}
                        />
                      )}
                      <Typography variant="body2">
                        {session.status.charAt(0).toUpperCase() +
                          session.status.slice(1)}
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Typography variant="body2">
                      Duration:{" "}
                      {session.duration ? `${session.duration}s` : "-"}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Button
                      size="small"
                      variant="outlined"
                      href={`/sessions/${session.session_id}`}
                    >
                      View Details
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            ))}
          </Box>
        ) : (
          <Typography color="textSecondary">No recent sessions</Typography>
        )}
      </Paper>
    </Box>
  );
}

export default Dashboard;
