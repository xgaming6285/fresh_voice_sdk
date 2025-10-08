import React, { useState, useEffect } from "react";
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Tooltip,
} from "@mui/material";
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Business as BusinessIcon,
  People as PeopleIcon,
  Person as PersonIcon,
  Campaign as CampaignIcon,
  ContactPage as ContactPageIcon,
  VpnKey as VpnKeyIcon,
} from "@mui/icons-material";
import api from "../services/api";

function SuperAdmin() {
  const [activeTab, setActiveTab] = useState(0);
  const [stats, setStats] = useState(null);
  const [admins, setAdmins] = useState([]);
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Dialog state
  const [openDialog, setOpenDialog] = useState(false);
  const [editingAdmin, setEditingAdmin] = useState(null);
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    organization: "",
    first_name: "",
    last_name: "",
  });

  // Password reset dialog state
  const [openPasswordDialog, setOpenPasswordDialog] = useState(false);
  const [resetPasswordTarget, setResetPasswordTarget] = useState(null); // { id, username, type: 'admin' | 'agent' }
  const [newPassword, setNewPassword] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError("");

      const [statsRes, adminsRes, agentsRes] = await Promise.all([
        api.superadminAPI.getStats(),
        api.superadminAPI.getAdmins(),
        api.superadminAPI.getAgents(),
      ]);

      setStats(statsRes.data);
      setAdmins(adminsRes.data);
      setAgents(agentsRes.data);
    } catch (err) {
      console.error("Error loading data:", err);
      setError(err.response?.data?.detail || "Failed to load data");
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDialog = (admin = null) => {
    if (admin) {
      setEditingAdmin(admin);
      setFormData({
        username: admin.username,
        email: admin.email,
        password: "",
        organization: admin.organization || "",
        first_name: admin.first_name || "",
        last_name: admin.last_name || "",
      });
    } else {
      setEditingAdmin(null);
      setFormData({
        username: "",
        email: "",
        password: "",
        organization: "",
        first_name: "",
        last_name: "",
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingAdmin(null);
    setFormData({
      username: "",
      email: "",
      password: "",
      organization: "",
      first_name: "",
      last_name: "",
    });
  };

  const handleSubmit = async () => {
    try {
      setError("");

      if (editingAdmin) {
        // Update admin (no password)
        const updateData = {
          email: formData.email,
          organization: formData.organization,
          first_name: formData.first_name,
          last_name: formData.last_name,
        };
        await api.superadminAPI.updateAdmin(editingAdmin.id, updateData);
        setSuccess("Admin updated successfully");
      } else {
        // Create new admin
        await api.superadminAPI.createAdmin(formData);
        setSuccess("Admin created successfully");
      }

      handleCloseDialog();
      loadData();
    } catch (err) {
      console.error("Error saving admin:", err);
      setError(err.response?.data?.detail || "Failed to save admin");
    }
  };

  const handleDeleteAdmin = async (adminId) => {
    if (
      !window.confirm(
        "Are you sure? This will also delete all agents and their data created by this admin."
      )
    ) {
      return;
    }

    try {
      setError("");
      await api.superadminAPI.deleteAdmin(adminId);
      setSuccess("Admin deleted successfully");
      loadData();
    } catch (err) {
      console.error("Error deleting admin:", err);
      setError(err.response?.data?.detail || "Failed to delete admin");
    }
  };

  const handleOpenPasswordDialog = (user, type) => {
    setResetPasswordTarget({ id: user.id, username: user.username, type });
    setNewPassword("");
    setOpenPasswordDialog(true);
  };

  const handleClosePasswordDialog = () => {
    setOpenPasswordDialog(false);
    setResetPasswordTarget(null);
    setNewPassword("");
  };

  const handleResetPassword = async () => {
    if (!newPassword || newPassword.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }

    try {
      setError("");

      if (resetPasswordTarget.type === "admin") {
        await api.superadminAPI.resetAdminPassword(
          resetPasswordTarget.id,
          newPassword
        );
        setSuccess(
          `Password for admin "${resetPasswordTarget.username}" reset successfully`
        );
      } else {
        await api.superadminAPI.resetAgentPassword(
          resetPasswordTarget.id,
          newPassword
        );
        setSuccess(
          `Password for agent "${resetPasswordTarget.username}" reset successfully`
        );
      }

      handleClosePasswordDialog();
    } catch (err) {
      console.error("Error resetting password:", err);
      setError(err.response?.data?.detail || "Failed to reset password");
    }
  };

  if (loading && !stats) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="80vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box mb={4}>
        <Typography variant="h4" gutterBottom>
          🔐 Super Admin Panel
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage all client organizations and their admins
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError("")} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" onClose={() => setSuccess("")} sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      {/* Statistics Cards */}
      {stats && (
        <Grid container spacing={3} mb={4}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <BusinessIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">{stats.total_admins}</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Total Admins
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <PeopleIcon color="secondary" sx={{ mr: 1 }} />
                  <Typography variant="h6">{stats.total_agents}</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Total Agents
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <ContactPageIcon color="success" sx={{ mr: 1 }} />
                  <Typography variant="h6">{stats.total_leads}</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Total Leads
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <CampaignIcon color="info" sx={{ mr: 1 }} />
                  <Typography variant="h6">{stats.total_campaigns}</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Total Campaigns
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Tabs */}
      <Paper sx={{ mb: 2 }}>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
        >
          <Tab label="Admins" />
          <Tab label="All Agents" />
        </Tabs>
      </Paper>

      {/* Admins Tab */}
      {activeTab === 0 && (
        <Paper>
          <Box
            p={2}
            display="flex"
            justifyContent="space-between"
            alignItems="center"
          >
            <Typography variant="h6">Client Admins</Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => handleOpenDialog()}
            >
              Create Admin
            </Button>
          </Box>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Username</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Organization</TableCell>
                  <TableCell>Full Name</TableCell>
                  <TableCell>Agents</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {admins.map((admin) => (
                  <TableRow key={admin.id}>
                    <TableCell>{admin.username}</TableCell>
                    <TableCell>{admin.email}</TableCell>
                    <TableCell>
                      {admin.organization && (
                        <Chip
                          icon={<BusinessIcon />}
                          label={admin.organization}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      )}
                    </TableCell>
                    <TableCell>{admin.full_name || "-"}</TableCell>
                    <TableCell>
                      <Chip
                        label={admin.agent_count}
                        size="small"
                        color="secondary"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={admin.is_active ? "Active" : "Inactive"}
                        size="small"
                        color={admin.is_active ? "success" : "default"}
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(admin.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title="Edit">
                        <IconButton
                          size="small"
                          onClick={() => handleOpenDialog(admin)}
                        >
                          <EditIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Reset Password">
                        <IconButton
                          size="small"
                          color="warning"
                          onClick={() =>
                            handleOpenPasswordDialog(admin, "admin")
                          }
                        >
                          <VpnKeyIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteAdmin(admin.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* All Agents Tab */}
      {activeTab === 1 && (
        <Paper>
          <Box p={2}>
            <Typography variant="h6">All Agents</Typography>
          </Box>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Username</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Full Name</TableCell>
                  <TableCell>Admin</TableCell>
                  <TableCell>Organization</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {agents.map((agent) => (
                  <TableRow key={agent.id}>
                    <TableCell>{agent.username}</TableCell>
                    <TableCell>{agent.email}</TableCell>
                    <TableCell>{agent.full_name || "-"}</TableCell>
                    <TableCell>
                      <Chip
                        icon={<PersonIcon />}
                        label={agent.admin_username}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      {agent.admin_organization && (
                        <Chip
                          icon={<BusinessIcon />}
                          label={agent.admin_organization}
                          size="small"
                          color="secondary"
                          variant="outlined"
                        />
                      )}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={agent.is_active ? "Active" : "Inactive"}
                        size="small"
                        color={agent.is_active ? "success" : "default"}
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(agent.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title="Reset Password">
                        <IconButton
                          size="small"
                          color="warning"
                          onClick={() =>
                            handleOpenPasswordDialog(agent, "agent")
                          }
                        >
                          <VpnKeyIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* Create/Edit Admin Dialog */}
      <Dialog
        open={openDialog}
        onClose={handleCloseDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingAdmin ? "Edit Admin" : "Create New Admin"}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 2 }}>
            <TextField
              label="Username"
              value={formData.username}
              onChange={(e) =>
                setFormData({ ...formData, username: e.target.value })
              }
              fullWidth
              required
              disabled={editingAdmin}
              helperText={editingAdmin && "Username cannot be changed"}
            />

            <TextField
              label="Email"
              type="email"
              value={formData.email}
              onChange={(e) =>
                setFormData({ ...formData, email: e.target.value })
              }
              fullWidth
              required
            />

            {!editingAdmin && (
              <TextField
                label="Password"
                type="password"
                value={formData.password}
                onChange={(e) =>
                  setFormData({ ...formData, password: e.target.value })
                }
                fullWidth
                required
                helperText="Minimum 6 characters"
              />
            )}

            <TextField
              label="Organization"
              value={formData.organization}
              onChange={(e) =>
                setFormData({ ...formData, organization: e.target.value })
              }
              fullWidth
              required
            />

            <TextField
              label="First Name"
              value={formData.first_name}
              onChange={(e) =>
                setFormData({ ...formData, first_name: e.target.value })
              }
              fullWidth
            />

            <TextField
              label="Last Name"
              value={formData.last_name}
              onChange={(e) =>
                setFormData({ ...formData, last_name: e.target.value })
              }
              fullWidth
            />

            {editingAdmin && (
              <Alert severity="info">
                Admins cannot change their passwords. Only superadmin can reset
                them.
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={
              !formData.username ||
              !formData.email ||
              !formData.organization ||
              (!editingAdmin && !formData.password)
            }
          >
            {editingAdmin ? "Update" : "Create"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Reset Password Dialog */}
      <Dialog
        open={openPasswordDialog}
        onClose={handleClosePasswordDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Reset Password for {resetPasswordTarget?.username}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 2 }}>
            <Alert severity="warning">
              You are about to reset the password for{" "}
              <strong>{resetPasswordTarget?.type}</strong>{" "}
              <strong>{resetPasswordTarget?.username}</strong>. Make sure to
              communicate the new password to them securely.
            </Alert>

            <TextField
              label="New Password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              fullWidth
              required
              helperText="Minimum 6 characters"
              autoFocus
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClosePasswordDialog}>Cancel</Button>
          <Button
            onClick={handleResetPassword}
            variant="contained"
            color="warning"
            disabled={!newPassword || newPassword.length < 6}
          >
            Reset Password
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default SuperAdmin;
