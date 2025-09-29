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
  Stop as StopIcon,
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
      wait_between_calls: 5,
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

  const handleCreateCampaign = async () => {
    try {
      const response = await campaignAPI.create(formData);
      showSnackbar("Campaign created successfully", "success");
      setOpenDialog(false);
      navigate(`/campaigns/${response.data.id}`);
    } catch (error) {
      console.error("Error creating campaign:", error);
      showSnackbar("Error creating campaign", "error");
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

  const handleStopCampaign = async (id) => {
    if (window.confirm("Are you sure you want to stop this campaign?")) {
      try {
        await campaignAPI.stop(id);
        showSnackbar("Campaign stopped", "success");
        loadCampaigns();
      } catch (error) {
        console.error("Error stopping campaign:", error);
        showSnackbar("Error stopping campaign", "error");
      }
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
      case "completed":
        return "primary";
      case "cancelled":
        return "error";
      default:
        return "default";
    }
  };

  const getProgressPercentage = (campaign) => {
    if (campaign.total_leads === 0) return 0;
    return Math.round((campaign.leads_called / campaign.total_leads) * 100);
  };

  return (
    <Box>
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={3}
      >
        <Typography variant="h4">Campaigns</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenDialog(true)}
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
          {campaigns.map((campaign) => (
            <Grid item xs={12} md={6} lg={4} key={campaign.id}>
              <Card>
                <CardContent>
                  <Box
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    mb={2}
                  >
                    <Typography variant="h6" component="div">
                      {campaign.name}
                    </Typography>
                    <Chip
                      label={campaign.status.toUpperCase()}
                      color={getStatusColor(campaign.status)}
                      size="small"
                    />
                  </Box>

                  {campaign.description && (
                    <Typography variant="body2" color="text.secondary" mb={2}>
                      {campaign.description}
                    </Typography>
                  )}

                  <Box mb={2}>
                    <Box display="flex" alignItems="center" mb={1}>
                      <PeopleIcon fontSize="small" sx={{ mr: 1 }} />
                      <Typography variant="body2">
                        {campaign.total_leads} leads
                      </Typography>
                    </Box>
                    <Box display="flex" alignItems="center" mb={1}>
                      <PhoneIcon fontSize="small" sx={{ mr: 1 }} />
                      <Typography variant="body2">
                        {campaign.leads_called} called
                      </Typography>
                    </Box>
                    <Box display="flex" alignItems="center" gap={2}>
                      <Box display="flex" alignItems="center">
                        <CheckCircleIcon
                          fontSize="small"
                          color="success"
                          sx={{ mr: 0.5 }}
                        />
                        <Typography variant="body2">
                          {campaign.leads_answered} answered
                        </Typography>
                      </Box>
                      <Box display="flex" alignItems="center">
                        <CancelIcon
                          fontSize="small"
                          color="error"
                          sx={{ mr: 0.5 }}
                        />
                        <Typography variant="body2">
                          {campaign.leads_rejected} rejected
                        </Typography>
                      </Box>
                    </Box>
                  </Box>

                  {campaign.status === "running" && (
                    <Box mb={2}>
                      <Box display="flex" justifyContent="space-between" mb={1}>
                        <Typography variant="body2">Progress</Typography>
                        <Typography variant="body2">
                          {getProgressPercentage(campaign)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={getProgressPercentage(campaign)}
                      />
                    </Box>
                  )}

                  <Typography variant="caption" color="text.secondary">
                    Created:{" "}
                    {new Date(campaign.created_at).toLocaleDateString()}
                  </Typography>
                </CardContent>

                <CardActions>
                  {campaign.status === "draft" ||
                  campaign.status === "ready" ? (
                    <Button
                      size="small"
                      startIcon={<PlayIcon />}
                      onClick={() => handleStartCampaign(campaign.id)}
                      disabled={campaign.total_leads === 0}
                    >
                      Start
                    </Button>
                  ) : campaign.status === "running" ? (
                    <>
                      <Button
                        size="small"
                        startIcon={<PauseIcon />}
                        onClick={() => handlePauseCampaign(campaign.id)}
                      >
                        Pause
                      </Button>
                      <Button
                        size="small"
                        startIcon={<StopIcon />}
                        onClick={() => handleStopCampaign(campaign.id)}
                        color="error"
                      >
                        Stop
                      </Button>
                    </>
                  ) : campaign.status === "paused" ? (
                    <>
                      <Button
                        size="small"
                        startIcon={<PlayIcon />}
                        onClick={() => handleStartCampaign(campaign.id)}
                      >
                        Resume
                      </Button>
                      <Button
                        size="small"
                        startIcon={<StopIcon />}
                        onClick={() => handleStopCampaign(campaign.id)}
                        color="error"
                      >
                        Stop
                      </Button>
                    </>
                  ) : null}

                  <Button
                    size="small"
                    onClick={() => navigate(`/campaigns/${campaign.id}`)}
                  >
                    View Details
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Create Campaign Dialog */}
      <Dialog
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create New Campaign</DialogTitle>
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
            <TextField
              fullWidth
              margin="normal"
              label="Wait Between Calls (seconds)"
              type="number"
              value={formData.dialing_config.wait_between_calls}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  dialing_config: {
                    ...formData.dialing_config,
                    wait_between_calls: parseInt(e.target.value),
                  },
                })
              }
              inputProps={{ min: 1, max: 60 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button
            onClick={handleCreateCampaign}
            variant="contained"
            disabled={!formData.name}
          >
            Create Campaign
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
