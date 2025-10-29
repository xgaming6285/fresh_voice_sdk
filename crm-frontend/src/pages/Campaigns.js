import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Typography,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
  Snackbar,
  Alert,
} from "@mui/material";
import {
  Add as AddIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Edit as EditIcon,
  People as PeopleIcon,
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Person as PersonIcon,
} from "@mui/icons-material";
import { campaignAPI } from "../services/api";
import { useAuth } from "../contexts/AuthContext";
import SubscriptionBanner from "../components/SubscriptionBanner";
import CampaignCallConfigDialog from "../components/CampaignCallConfigDialog";
import { Tooltip } from "@mui/material";

function Campaigns() {
  const navigate = useNavigate();
  const { isAdmin, hasActiveSubscription } = useAuth();
  const [campaigns, setCampaigns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingCampaign, setEditingCampaign] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: "",
    severity: "success",
  });
  const [callConfigDialog, setCallConfigDialog] = useState(false);
  const [selectedCampaign, setSelectedCampaign] = useState(null);

  const [formData, setFormData] = useState({
    name: "",
    description: "",
    bot_config: {
      greeting_delay: 1,
      max_call_duration: 300,
      voice_speed: 1.0,
    },
    dialing_config: {
      concurrent_calls: 1,
      retry_attempts: 2,
    },
  });

  useEffect(() => {
    loadCampaigns();
  }, []);

  const loadCampaigns = async () => {
    setLoading(true);
    try {
      const response = await campaignAPI.getAll();
      setCampaigns(response.data);
    } catch (error) {
      console.error("Error loading campaigns:", error);
      showSnackbar("Error loading campaigns", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleOpenCreateDialog = () => {
    setEditingCampaign(null);
    setFormData({
      name: "",
      description: "",
      bot_config: {
        greeting_delay: 1,
        max_call_duration: 300,
        voice_speed: 1.0,
      },
      dialing_config: {
        concurrent_calls: 1,
        retry_attempts: 2,
      },
    });
    setOpenDialog(true);
  };

  const handleOpenEditDialog = (campaign) => {
    setEditingCampaign(campaign);
    setFormData({
      name: campaign.name,
      description: campaign.description || "",
      bot_config: campaign.bot_config,
      dialing_config: campaign.dialing_config,
    });
    setOpenDialog(true);
  };

  const handleSaveCampaign = async () => {
    try {
      if (editingCampaign) {
        await campaignAPI.update(editingCampaign.id, formData);
        showSnackbar("Campaign updated successfully", "success");
        setOpenDialog(false);
        loadCampaigns();
      } else {
        const response = await campaignAPI.create(formData);
        showSnackbar("Campaign created successfully", "success");
        setOpenDialog(false);
        navigate(`/campaigns/${response.data.id}`);
      }
    } catch (error) {
      console.error("Error saving campaign:", error);
      showSnackbar(
        `Error ${editingCampaign ? "updating" : "creating"} campaign`,
        "error"
      );
    }
  };

  const handleStartCampaign = async (id) => {
    try {
      // Find campaign first to show dialog
      const campaign = campaigns.find((c) => c.id === id);
      if (!campaign) {
        showSnackbar("Campaign not found", "error");
        return;
      }
      setSelectedCampaign(campaign);
      setCallConfigDialog(true);
    } catch (error) {
      console.error("Error preparing campaign start:", error);
      showSnackbar("Error preparing campaign", "error");
    }
  };

  const handleStartCampaignWithConfig = async (callConfig) => {
    try {
      // Start campaign with call configuration
      await campaignAPI.start(selectedCampaign.id, callConfig);
      showSnackbar("Campaign started", "success");
      setCallConfigDialog(false);
      setSelectedCampaign(null);
      loadCampaigns();
    } catch (error) {
      console.error("Error starting campaign:", error);
      showSnackbar("Error starting campaign", "error");
    }
  };

  const handlePauseCampaign = async (id) => {
    try {
      await campaignAPI.pause(id);
      showSnackbar("Campaign paused", "success");
      loadCampaigns();
    } catch (error) {
      console.error("Error pausing campaign:", error);
      showSnackbar("Error pausing campaign", "error");
    }
  };

  const showSnackbar = (message, severity) => {
    setSnackbar({ open: true, message, severity });
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "draft":
        return "default";
      case "ready":
        return "info";
      case "running":
        return "success";
      case "paused":
        return "warning";
      default:
        return "default";
    }
  };

  return (
    <Box className="fade-in">
      <SubscriptionBanner />
      {loading ? (
        <Box display="flex" justifyContent="center" p={4}>
          <LinearProgress sx={{ width: "50%" }} />
        </Box>
      ) : campaigns.length === 0 ? (
        /* Empty State - First Time */
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "60vh",
            py: 8,
          }}
        >
          <Box
            className="scale-in"
            sx={{
              textAlign: "center",
              maxWidth: 600,
              px: 3,
            }}
          >
            {/* Animated Icon */}
            <Box
              className="bounce-in float"
              sx={{
                mb: 4,
                position: "relative",
                display: "inline-block",
              }}
            >
              <Box
                sx={{
                  width: 140,
                  height: 140,
                  borderRadius: "50%",
                  background:
                    "linear-gradient(135deg, rgba(200, 92, 60, 0.15) 0%, rgba(139, 94, 60, 0.15) 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  position: "relative",
                  animation: "borderGlow 2s ease-in-out infinite",
                  "&::before": {
                    content: '""',
                    position: "absolute",
                    top: -10,
                    left: -10,
                    right: -10,
                    bottom: -10,
                    borderRadius: "50%",
                    border: "2px dashed rgba(200, 92, 60, 0.3)",
                    animation: "spin 20s linear infinite",
                  },
                }}
              >
                <Box
                  sx={{
                    width: 100,
                    height: 100,
                    borderRadius: "50%",
                    background:
                      "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: "0 8px 32px rgba(200, 92, 60, 0.4)",
                  }}
                >
                  <AddIcon sx={{ fontSize: 50, color: "white" }} />
                </Box>
              </Box>
            </Box>

            {/* Heading */}
            <Typography
              variant="h3"
              className="text-shimmer"
              sx={{
                fontWeight: 800,
                mb: 2,
                animation: "slideUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
                animationDelay: "0.2s",
                animationFillMode: "both",
              }}
            >
              Create Your First Campaign
            </Typography>

            {/* Description */}
            <Typography
              variant="body1"
              sx={{
                color: "text.secondary",
                mb: 4,
                lineHeight: 1.8,
                fontSize: "1.1rem",
                animation: "slideUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
                animationDelay: "0.3s",
                animationFillMode: "both",
              }}
            >
              Start reaching your leads with automated voice campaigns.
              <br />
              Set up your first campaign and watch your outreach come to life!
              🚀
            </Typography>

            {/* Features List */}
            <Box
              sx={{
                display: "flex",
                justifyContent: "center",
                gap: 3,
                mb: 5,
                flexWrap: "wrap",
                animation: "slideUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
                animationDelay: "0.4s",
                animationFillMode: "both",
              }}
            >
              {[
                { icon: "📞", text: "Automated Calls" },
                { icon: "🎯", text: "Smart Targeting" },
                { icon: "📊", text: "Real-time Analytics" },
              ].map((feature, index) => (
                <Box
                  key={index}
                  className={`scale-in stagger-${index + 1}`}
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    px: 2,
                    py: 1,
                    borderRadius: 3,
                    background: "rgba(200, 92, 60, 0.05)",
                    backdropFilter: "blur(10px)",
                    border: "1px solid rgba(200, 92, 60, 0.1)",
                  }}
                >
                  <Typography variant="h6">{feature.icon}</Typography>
                  <Typography variant="body2" fontWeight={600}>
                    {feature.text}
                  </Typography>
                </Box>
              ))}
            </Box>

            {/* CTA Button */}
            <Tooltip
              title={
                !hasActiveSubscription()
                  ? "Subscription required to create campaigns"
                  : ""
              }
            >
              <Box
                sx={{
                  animation: "slideUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
                  animationDelay: "0.5s",
                  animationFillMode: "both",
                }}
              >
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<AddIcon />}
                  onClick={handleOpenCreateDialog}
                  disabled={!hasActiveSubscription()}
                  className="ripple-container hover-scale"
                  sx={{
                    py: 2,
                    px: 5,
                    fontSize: "1.1rem",
                    fontWeight: 700,
                    borderRadius: 4,
                    background:
                      "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                    boxShadow: "0 8px 32px rgba(200, 92, 60, 0.4)",
                    position: "relative",
                    overflow: "hidden",
                    "&::before": {
                      content: '""',
                      position: "absolute",
                      top: 0,
                      left: "-100%",
                      width: "100%",
                      height: "100%",
                      background:
                        "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)",
                      transition: "left 0.5s",
                    },
                    "&:hover": {
                      boxShadow: "0 12px 40px rgba(200, 92, 60, 0.6)",
                      transform: "translateY(-4px) scale(1.05)",
                      "&::before": {
                        left: "100%",
                      },
                    },
                    "&:disabled": {
                      background: "rgba(0, 0, 0, 0.12)",
                      boxShadow: "none",
                    },
                  }}
                >
                  Create Your First Campaign
                </Button>
              </Box>
            </Tooltip>

            {!hasActiveSubscription() && (
              <Typography
                variant="caption"
                sx={{
                  display: "block",
                  mt: 2,
                  color: "warning.main",
                  fontWeight: 600,
                  animation: "fadeIn 0.8s ease-out",
                  animationDelay: "0.6s",
                  animationFillMode: "both",
                }}
              >
                💡 Subscribe to start creating campaigns
              </Typography>
            )}
          </Box>
        </Box>
      ) : (
        /* Campaigns Grid with Template Card */
        <Grid container spacing={3}>
          {/* Add Campaign Template Card */}
          <Grid item xs={4} md={3} lg={2}>
            <Tooltip
              title={
                !hasActiveSubscription()
                  ? "Subscription required to create campaigns"
                  : "Create new campaign"
              }
            >
              <Card
                onClick={
                  hasActiveSubscription() ? handleOpenCreateDialog : undefined
                }
                className="glass-effect ios-card hover-lift"
                sx={{
                  height: "100%",
                  minHeight: 120,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: hasActiveSubscription() ? "pointer" : "not-allowed",
                  border: "2px dashed rgba(200, 92, 60, 0.3)",
                  background: "rgba(200, 92, 60, 0.02)",
                  transition: "all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
                  opacity: hasActiveSubscription() ? 1 : 0.5,
                  animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
                  "&:hover": hasActiveSubscription()
                    ? {
                        border: "2px dashed rgba(200, 92, 60, 0.6)",
                        background: "rgba(200, 92, 60, 0.08)",
                        transform: "translateY(-4px) scale(1.02)",
                        boxShadow: "0 8px 24px rgba(200, 92, 60, 0.2)",
                      }
                    : {},
                }}
              >
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 1,
                  }}
                >
                  <Box
                    className="hover-rotate"
                    sx={{
                      width: 35,
                      height: 35,
                      borderRadius: "50%",
                      background:
                        "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      transition: "all 0.3s ease",
                    }}
                  >
                    <AddIcon sx={{ fontSize: 20, color: "white" }} />
                  </Box>
                  <Typography
                    variant="body2"
                    fontWeight={600}
                    sx={{
                      background:
                        "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      backgroundClip: "text",
                      fontSize: "0.75rem",
                    }}
                  >
                    New Campaign
                  </Typography>
                </Box>
              </Card>
            </Tooltip>
          </Grid>

          {/* Existing Campaigns */}
          {campaigns.map((campaign, index) => (
            <Grid item xs={4} md={3} lg={2} key={campaign.id}>
              <Card
                className="glass-effect ios-card"
                sx={{
                  height: "100%",
                  minHeight: 120,
                  animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
                  animationDelay: `${index * 0.1}s`,
                  animationFillMode: "both",
                  "&:hover": {
                    "& .campaign-status": {
                      transform: "scale(1.1)",
                    },
                  },
                }}
              >
                <CardContent sx={{ p: 1.5 }}>
                  <Box
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    mb={1}
                  >
                    <Typography
                      variant="body2"
                      component="div"
                      fontWeight={700}
                      noWrap
                      sx={{
                        background:
                          "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                        backgroundClip: "text",
                        fontSize: "0.875rem",
                        maxWidth: "70%",
                      }}
                    >
                      {campaign.name}
                    </Typography>
                    <Chip
                      label={campaign.status.toUpperCase()}
                      color={getStatusColor(campaign.status)}
                      size="small"
                      className="campaign-status"
                      sx={{
                        fontWeight: 700,
                        backdropFilter: "blur(10px)",
                        transition:
                          "all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
                        height: 18,
                        fontSize: "0.65rem",
                      }}
                    />
                  </Box>

                  {/* Show owner name for admins */}
                  {isAdmin() && campaign.owner_name && (
                    <Box mb={1}>
                      <Chip
                        icon={<PersonIcon sx={{ fontSize: 12 }} />}
                        label={campaign.owner_name}
                        size="small"
                        sx={{
                          height: 20,
                          fontSize: "0.65rem",
                          fontWeight: 600,
                          bgcolor: "rgba(200, 92, 60, 0.08)",
                          color: "primary.main",
                          "& .MuiChip-icon": {
                            color: "primary.main",
                          },
                        }}
                      />
                    </Box>
                  )}

                  <Box mb={1}>
                    <Box
                      display="flex"
                      alignItems="center"
                      justifyContent="space-between"
                      sx={{
                        p: 0.75,
                        borderRadius: 1,
                        background: "rgba(248, 243, 239, 0.5)",
                        backdropFilter: "blur(10px)",
                      }}
                    >
                      <PeopleIcon
                        sx={{ fontSize: 14, mr: 0.5, color: "primary.main" }}
                      />
                      <Typography variant="caption" fontWeight={500}>
                        {campaign.total_leads}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>

                <CardActions sx={{ p: 1, pt: 0, justifyContent: "center" }}>
                  {campaign.status === "draft" ||
                  campaign.status === "ready" ? (
                    <IconButton
                      size="small"
                      onClick={() => handleStartCampaign(campaign.id)}
                      disabled={campaign.total_leads === 0}
                      sx={{
                        color: "success.main",
                        "&:hover": {
                          backgroundColor: "rgba(107, 154, 90, 0.08)",
                        },
                      }}
                    >
                      <PlayIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  ) : campaign.status === "running" ? (
                    <IconButton
                      size="small"
                      onClick={() => handlePauseCampaign(campaign.id)}
                      sx={{
                        color: "warning.main",
                        "&:hover": {
                          backgroundColor: "rgba(217, 156, 94, 0.08)",
                        },
                      }}
                    >
                      <PauseIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  ) : campaign.status === "paused" ? (
                    <IconButton
                      size="small"
                      onClick={() => handleStartCampaign(campaign.id)}
                      sx={{
                        color: "success.main",
                        "&:hover": {
                          backgroundColor: "rgba(107, 154, 90, 0.08)",
                        },
                      }}
                    >
                      <PlayIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  ) : null}

                  <IconButton
                    size="small"
                    onClick={() => handleOpenEditDialog(campaign)}
                    sx={{
                      color: "primary.main",
                      "&:hover": {
                        backgroundColor: "rgba(139, 94, 60, 0.08)",
                      },
                    }}
                  >
                    <EditIcon sx={{ fontSize: 20 }} />
                  </IconButton>

                  <IconButton
                    size="small"
                    onClick={() => navigate(`/campaigns/${campaign.id}`)}
                    sx={{
                      color: "primary.main",
                      "&:hover": {
                        backgroundColor: "rgba(200, 92, 60, 0.08)",
                      },
                    }}
                  >
                    <PeopleIcon sx={{ fontSize: 20 }} />
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Create/Edit Campaign Dialog */}
      <Dialog
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingCampaign ? "Edit Campaign" : "Create New Campaign"}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              margin="normal"
              label="Campaign Name"
              value={formData.name}
              onChange={(e) =>
                setFormData({ ...formData, name: e.target.value })
              }
              required
            />
            <TextField
              fullWidth
              margin="normal"
              label="Description"
              multiline
              rows={3}
              value={formData.description}
              onChange={(e) =>
                setFormData({ ...formData, description: e.target.value })
              }
            />

            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              Bot Configuration
            </Typography>
            <TextField
              fullWidth
              margin="normal"
              label="Greeting Delay (seconds)"
              type="number"
              value={formData.bot_config.greeting_delay}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  bot_config: {
                    ...formData.bot_config,
                    greeting_delay: parseInt(e.target.value),
                  },
                })
              }
              inputProps={{ min: 0, max: 10 }}
            />
            <TextField
              fullWidth
              margin="normal"
              label="Max Call Duration (seconds)"
              type="number"
              value={formData.bot_config.max_call_duration}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  bot_config: {
                    ...formData.bot_config,
                    max_call_duration: parseInt(e.target.value),
                  },
                })
              }
              inputProps={{ min: 30, max: 3600 }}
            />

            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              Dialing Configuration
            </Typography>
            <TextField
              fullWidth
              margin="normal"
              label="Concurrent Calls"
              type="number"
              value={formData.dialing_config.concurrent_calls}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  dialing_config: {
                    ...formData.dialing_config,
                    concurrent_calls: parseInt(e.target.value),
                  },
                })
              }
              inputProps={{ min: 1, max: 10 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button
            onClick={handleSaveCampaign}
            variant="contained"
            disabled={!formData.name}
          >
            {editingCampaign ? "Save Changes" : "Create Campaign"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* Campaign Call Config Dialog */}
      <CampaignCallConfigDialog
        open={callConfigDialog}
        onClose={() => {
          setCallConfigDialog(false);
          setSelectedCampaign(null);
        }}
        campaign={selectedCampaign}
        onStartCampaign={handleStartCampaignWithConfig}
      />
    </Box>
  );
}

export default Campaigns;
