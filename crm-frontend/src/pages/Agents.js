import React, { useState, useEffect } from "react";
import { useAuth } from "../contexts/AuthContext";
import { userAPI } from "../services/api";
import {
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Typography,
  IconButton,
  Grid,
  Chip,
  Alert,
  CircularProgress,
} from "@mui/material";
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
} from "@mui/icons-material";

const Agents = () => {
  const { user, isAdmin } = useAuth();
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [openDialog, setOpenDialog] = useState(false);
  const [editingAgent, setEditingAgent] = useState(null);
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    first_name: "",
    last_name: "",
  });

  useEffect(() => {
    if (isAdmin()) {
      loadAgents();
    }
  }, []);

  const loadAgents = async () => {
    try {
      setLoading(true);
      const response = await userAPI.getAgents();
      setAgents(response.data);
      setError("");
    } catch (err) {
      setError("Failed to load agents");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDialog = (agent = null) => {
    if (agent) {
      setEditingAgent(agent);
      setFormData({
        username: agent.username,
        email: agent.email,
        password: "",
        first_name: agent.first_name || "",
        last_name: agent.last_name || "",
      });
    } else {
      setEditingAgent(null);
      setFormData({
        username: "",
        email: "",
        password: "",
        first_name: "",
        last_name: "",
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingAgent(null);
    setFormData({
      username: "",
      email: "",
      password: "",
      first_name: "",
      last_name: "",
    });
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async () => {
    try {
      if (editingAgent) {
        // Update agent
        const updateData = { ...formData };
        // Remove password if empty
        if (!updateData.password) {
          delete updateData.password;
        }
        await userAPI.updateAgent(editingAgent.id, updateData);
      } else {
        // Create new agent
        await userAPI.createAgent(formData);
      }
      handleCloseDialog();
      loadAgents();
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to save agent");
    }
  };

  const handleDelete = async (agentId) => {
    if (
      window.confirm(
        "Are you sure you want to delete this agent? This will also delete all their leads and campaigns."
      )
    ) {
      try {
        await userAPI.deleteAgent(agentId);
        loadAgents();
      } catch (err) {
        setError(err.response?.data?.detail || "Failed to delete agent");
      }
    }
  };

  if (!isAdmin()) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          You must be an admin to access this page.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 3 }}>
        <Typography variant="h4">Manage Agents</Typography>
        <Box>
          <Button
            startIcon={<RefreshIcon />}
            onClick={loadAgents}
            sx={{ mr: 1 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => handleOpenDialog()}
          >
            Add Agent
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>
          {error}
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {agents.map((agent) => (
            <Grid item xs={12} sm={6} md={4} key={agent.id}>
              <Card>
                <CardContent>
                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 2,
                    }}
                  >
                    <Typography variant="h6">{agent.full_name}</Typography>
                    <Box>
                      <IconButton
                        size="small"
                        onClick={() => handleOpenDialog(agent)}
                      >
                        <EditIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDelete(agent.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </Box>

                  <Typography
                    variant="body2"
                    color="text.secondary"
                    gutterBottom
                  >
                    @{agent.username}
                  </Typography>

                  <Typography
                    variant="body2"
                    color="text.secondary"
                    gutterBottom
                  >
                    {agent.email}
                  </Typography>

                  <Box sx={{ mt: 2 }}>
                    <Chip
                      label={agent.is_active ? "Active" : "Inactive"}
                      color={agent.is_active ? "success" : "default"}
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    <Chip label="Agent" color="primary" size="small" />
                  </Box>

                  {agent.last_login && (
                    <Typography
                      variant="caption"
                      display="block"
                      sx={{ mt: 2 }}
                    >
                      Last login:{" "}
                      {new Date(agent.last_login).toLocaleDateString()}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Add/Edit Agent Dialog */}
      <Dialog
        open={openDialog}
        onClose={handleCloseDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingAgent ? "Edit Agent" : "Add New Agent"}
        </DialogTitle>
        <DialogContent>
          <TextField
            label="Username"
            name="username"
            value={formData.username}
            onChange={handleChange}
            fullWidth
            required
            margin="normal"
            disabled={!!editingAgent}
          />

          <TextField
            label="Email"
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            fullWidth
            required
            margin="normal"
          />

          <TextField
            label="First Name"
            name="first_name"
            value={formData.first_name}
            onChange={handleChange}
            fullWidth
            margin="normal"
          />

          <TextField
            label="Last Name"
            name="last_name"
            value={formData.last_name}
            onChange={handleChange}
            fullWidth
            margin="normal"
          />

          <TextField
            label="Password"
            name="password"
            type="password"
            value={formData.password}
            onChange={handleChange}
            fullWidth
            required={!editingAgent}
            margin="normal"
            helperText={
              editingAgent
                ? "Leave blank to keep current password"
                : "Minimum 6 characters"
            }
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={
              !formData.username ||
              !formData.email ||
              (!editingAgent && !formData.password)
            }
          >
            {editingAgent ? "Update" : "Create"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Agents;
