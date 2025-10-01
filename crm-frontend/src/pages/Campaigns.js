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
} from "@mui/icons-material";
import { campaignAPI } from "../services/api";

function Campaigns() {
  const navigate = useNavigate();
  const [campaigns, setCampaigns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingCampaign, setEditingCampaign] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: "",
    severity: "success",
  });

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
      await campaignAPI.start(id);
      showSnackbar("Campaign started", "success");
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
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={4}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography variant="h4">🎯</Typography>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 700,
              background: "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Campaigns
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleOpenCreateDialog}
          sx={{
            background: "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
            fontWeight: 600,
            px: 3,
            py: 1.5,
            borderRadius: 2.5,
            boxShadow: "0 4px 16px rgba(200, 92, 60, 0.3)",
            "&:hover": {
              background: "linear-gradient(135deg, #E07B5F 0%, #C85C3C 100%)",
              boxShadow: "0 8px 24px rgba(200, 92, 60, 0.4)",
              transform: "translateY(-2px)",
            },
          }}
        >
          Create Campaign
        </Button>
      </Box>

      {loading ? (
        <Box display="flex" justifyContent="center" p={4}>
          <LinearProgress sx={{ width: "50%" }} />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {campaigns.map((campaign, index) => (
            <Grid item xs={12} md={6} lg={4} key={campaign.id}>
              <Card
                className="glass-effect ios-card"
                sx={{
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
                <CardContent>
                  <Box
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    mb={2}
                  >
                    <Typography
                      variant="h6"
                      component="div"
                      fontWeight={700}
                      sx={{
                        background:
                          "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                        backgroundClip: "text",
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
                      }}
                    />
                  </Box>

                  {campaign.description && (
                    <Typography variant="body2" color="text.secondary" mb={2}>
                      {campaign.description}
                    </Typography>
                  )}

                  <Box mb={2}>
                    <Box
                      display="flex"
                      alignItems="center"
                      mb={1}
                      sx={{
                        p: 1.5,
                        borderRadius: 2,
                        background: "rgba(248, 243, 239, 0.5)",
                        backdropFilter: "blur(10px)",
                      }}
                    >
                      <PeopleIcon
                        fontSize="small"
                        sx={{ mr: 1, color: "primary.main" }}
                      />
                      <Typography variant="body2" fontWeight={500}>
                        {campaign.total_leads} leads
                      </Typography>
                    </Box>
                    <Box
                      display="flex"
                      alignItems="center"
                      mb={1}
                      sx={{
                        p: 1.5,
                        borderRadius: 2,
                        background: "rgba(248, 243, 239, 0.5)",
                        backdropFilter: "blur(10px)",
                      }}
                    >
                      <PhoneIcon
                        fontSize="small"
                        sx={{ mr: 1, color: "info.main" }}
                      />
                      <Typography variant="body2" fontWeight={500}>
                        {campaign.leads_called} called
                      </Typography>
                    </Box>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Box
                        display="flex"
                        alignItems="center"
                        sx={{
                          p: 1,
                          px: 1.5,
                          borderRadius: 2,
                          background: "rgba(107, 154, 90, 0.1)",
                          backdropFilter: "blur(10px)",
                        }}
                      >
                        <CheckCircleIcon
                          fontSize="small"
                          color="success"
                          sx={{ mr: 0.5 }}
                        />
                        <Typography variant="body2" fontWeight={500}>
                          {campaign.leads_answered}
                        </Typography>
                      </Box>
                      <Box
                        display="flex"
                        alignItems="center"
                        sx={{
                          p: 1,
                          px: 1.5,
                          borderRadius: 2,
                          background: "rgba(199, 84, 80, 0.1)",
                          backdropFilter: "blur(10px)",
                        }}
                      >
                        <CancelIcon
                          fontSize="small"
                          color="error"
                          sx={{ mr: 0.5 }}
                        />
                        <Typography variant="body2" fontWeight={500}>
                          {campaign.leads_rejected}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>

                  <Typography variant="caption" color="text.secondary">
                    Created:{" "}
                    {new Date(campaign.created_at).toLocaleDateString()}
                  </Typography>
                </CardContent>

                <CardActions sx={{ p: 2, pt: 0 }}>
                  {campaign.status === "draft" ||
                  campaign.status === "ready" ? (
                    <Button
                      size="small"
                      startIcon={<PlayIcon />}
                      onClick={() => handleStartCampaign(campaign.id)}
                      disabled={campaign.total_leads === 0}
                      className="ios-button"
                      sx={{
                        fontWeight: 600,
                        color: "success.main",
                        "&:hover": {
                          backgroundColor: "rgba(107, 154, 90, 0.08)",
                        },
                      }}
                    >
                      Start
                    </Button>
                  ) : campaign.status === "running" ? (
                    <Button
                      size="small"
                      startIcon={<PauseIcon />}
                      onClick={() => handlePauseCampaign(campaign.id)}
                      className="ios-button"
                      sx={{
                        fontWeight: 600,
                        color: "warning.main",
                        "&:hover": {
                          backgroundColor: "rgba(217, 156, 94, 0.08)",
                        },
                      }}
                    >
                      Pause
                    </Button>
                  ) : campaign.status === "paused" ? (
                    <Button
                      size="small"
                      startIcon={<PlayIcon />}
                      onClick={() => handleStartCampaign(campaign.id)}
                      className="ios-button"
                      sx={{
                        fontWeight: 600,
                        color: "success.main",
                        "&:hover": {
                          backgroundColor: "rgba(107, 154, 90, 0.08)",
                        },
                      }}
                    >
                      Resume
                    </Button>
                  ) : null}

                  <Button
                    size="small"
                    startIcon={<EditIcon />}
                    onClick={() => handleOpenEditDialog(campaign)}
                    className="ios-button"
                    sx={{
                      fontWeight: 600,
                      "&:hover": {
                        backgroundColor: "rgba(139, 94, 60, 0.08)",
                      },
                    }}
                  >
                    Edit
                  </Button>

                  <Button
                    size="small"
                    onClick={() => navigate(`/campaigns/${campaign.id}`)}
                    variant="text"
                    className="ios-button"
                    sx={{
                      fontWeight: 600,
                      color: "primary.main",
                      "&:hover": {
                        backgroundColor: "rgba(200, 92, 60, 0.08)",
                      },
                    }}
                  >
                    Details →
                  </Button>
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
    </Box>
  );
}

export default Campaigns;
