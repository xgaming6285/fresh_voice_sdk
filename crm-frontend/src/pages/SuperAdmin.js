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
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  AddCircle as AddCircleIcon,
  RemoveCircle as RemoveCircleIcon,
} from "@mui/icons-material";
import api from "../services/api";

function SuperAdmin() {
  const [activeTab, setActiveTab] = useState(0);
  const [stats, setStats] = useState(null);
  const [admins, setAdmins] = useState([]);
  const [agents, setAgents] = useState([]);
  const [paymentRequests, setPaymentRequests] = useState([]);
  const [walletAddress, setWalletAddress] = useState("");
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

  // Payment dialog state
  const [openPaymentDialog, setOpenPaymentDialog] = useState(false);
  const [selectedPaymentRequest, setSelectedPaymentRequest] = useState(null);
  const [paymentAction, setPaymentAction] = useState(""); // 'approve' or 'reject'
  const [paymentNotes, setPaymentNotes] = useState("");

  // Wallet dialog state
  const [openWalletDialog, setOpenWalletDialog] = useState(false);
  const [newWalletAddress, setNewWalletAddress] = useState("");

  // Slot adjustment dialog state
  const [openSlotDialog, setOpenSlotDialog] = useState(false);
  const [selectedAdmin, setSelectedAdmin] = useState(null);
  const [slotsChange, setSlotsChange] = useState("");
  const [slotAdjustmentReason, setSlotAdjustmentReason] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError("");

      const [statsRes, adminsRes, agentsRes, paymentsRes, walletRes] =
        await Promise.all([
          api.superadminAPI.getStats(),
          api.superadminAPI.getAdmins(),
          api.superadminAPI.getAgents(),
          api.superadminAPI.getPaymentRequests(),
          api.superadminAPI.getPaymentWallet(),
        ]);

      setStats(statsRes.data);
      setAdmins(adminsRes.data);
      setAgents(agentsRes.data);
      setPaymentRequests(paymentsRes.data);
      setWalletAddress(walletRes.data.wallet_address || "");
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

  const handleOpenPaymentDialog = (request, action) => {
    setSelectedPaymentRequest(request);
    setPaymentAction(action);
    setPaymentNotes("");
    setOpenPaymentDialog(true);
  };

  const handleClosePaymentDialog = () => {
    setOpenPaymentDialog(false);
    setSelectedPaymentRequest(null);
    setPaymentAction("");
    setPaymentNotes("");
  };

  const handlePaymentAction = async () => {
    try {
      setError("");

      if (paymentAction === "approve") {
        await api.superadminAPI.approvePaymentRequest(
          selectedPaymentRequest.id,
          paymentNotes
        );
        setSuccess("Payment request approved!");
      } else {
        await api.superadminAPI.rejectPaymentRequest(
          selectedPaymentRequest.id,
          paymentNotes
        );
        setSuccess("Payment request rejected");
      }

      handleClosePaymentDialog();
      loadData();
    } catch (err) {
      console.error("Error processing payment:", err);
      setError(err.response?.data?.detail || "Failed to process payment");
    }
  };

  const handleOpenWalletDialog = () => {
    setNewWalletAddress(walletAddress);
    setOpenWalletDialog(true);
  };

  const handleCloseWalletDialog = () => {
    setOpenWalletDialog(false);
  };

  const handleSaveWallet = async () => {
    try {
      setError("");

      await api.superadminAPI.setPaymentWallet(newWalletAddress);
      setSuccess("Payment wallet address updated");
      setWalletAddress(newWalletAddress);
      handleCloseWalletDialog();
    } catch (err) {
      console.error("Error updating wallet:", err);
      setError(err.response?.data?.detail || "Failed to update wallet");
    }
  };

  const handleOpenSlotDialog = (admin) => {
    setSelectedAdmin(admin);
    setSlotsChange("");
    setSlotAdjustmentReason("");
    setOpenSlotDialog(true);
  };

  const handleCloseSlotDialog = () => {
    setOpenSlotDialog(false);
    setSelectedAdmin(null);
    setSlotsChange("");
    setSlotAdjustmentReason("");
  };

  const handleAdjustSlots = async () => {
    try {
      setError("");

      const slotsNum = parseInt(slotsChange);
      if (isNaN(slotsNum) || slotsNum === 0) {
        setError(
          "Please enter a valid number (positive to add, negative to remove)"
        );
        return;
      }

      if (!slotAdjustmentReason || slotAdjustmentReason.trim().length < 3) {
        setError("Please provide a reason (at least 3 characters)");
        return;
      }

      await api.superadminAPI.adjustAdminSlots(
        selectedAdmin.id,
        slotsNum,
        slotAdjustmentReason
      );

      const action = slotsNum > 0 ? "added" : "removed";
      setSuccess(
        `Successfully ${action} ${Math.abs(slotsNum)} agent slot(s) for ${
          selectedAdmin.username
        }`
      );
      handleCloseSlotDialog();
      loadData();
    } catch (err) {
      console.error("Error adjusting slots:", err);
      setError(err.response?.data?.detail || "Failed to adjust slots");
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
          <Tab label="Payments" />
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
                      <Tooltip title="Adjust Agent Slots">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleOpenSlotDialog(admin)}
                        >
                          <AddCircleIcon />
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

      {/* Payments Tab */}
      {activeTab === 2 && (
        <>
          {/* Wallet Address Card */}
          <Card sx={{ mb: 3, bgcolor: "primary.light" }}>
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={8}>
                  <Typography variant="h6" gutterBottom>
                    💰 Payment Wallet Address
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      wordBreak: "break-all",
                      fontFamily: "monospace",
                      bgcolor: "rgba(255,255,255,0.2)",
                      p: 1,
                      borderRadius: 1,
                    }}
                  >
                    {walletAddress || "Not set"}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4} textAlign="right">
                  <Button
                    variant="contained"
                    onClick={handleOpenWalletDialog}
                    sx={{
                      bgcolor: "white",
                      color: "primary.main",
                      "&:hover": { bgcolor: "grey.100" },
                    }}
                  >
                    {walletAddress ? "Update Wallet" : "Set Wallet"}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Payment Requests Table */}
          <Paper>
            <Box p={2}>
              <Typography variant="h6">Payment Requests</Typography>
              <Typography variant="caption" color="text.secondary">
                ${300}/month per agent
              </Typography>
            </Box>

            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>Admin</TableCell>
                    <TableCell>Organization</TableCell>
                    <TableCell>Current/Max</TableCell>
                    <TableCell>Requested</TableCell>
                    <TableCell>Amount</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paymentRequests.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={9} align="center">
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          py={3}
                        >
                          No payment requests
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    paymentRequests.map((request) => (
                      <TableRow key={request.id}>
                        <TableCell>#{request.id}</TableCell>
                        <TableCell>{request.admin_username}</TableCell>
                        <TableCell>
                          {request.admin_organization && (
                            <Chip
                              icon={<BusinessIcon />}
                              label={request.admin_organization}
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </TableCell>
                        <TableCell>
                          {request.current_agents} / {request.max_agents}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={`+${request.num_agents}`}
                            size="small"
                            color="info"
                          />
                        </TableCell>
                        <TableCell fontWeight={600}>
                          ${request.total_amount}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={request.status.toUpperCase()}
                            size="small"
                            color={
                              request.status === "approved"
                                ? "success"
                                : request.status === "pending"
                                ? "warning"
                                : "error"
                            }
                          />
                        </TableCell>
                        <TableCell>
                          {new Date(request.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell align="right">
                          {request.status === "pending" && (
                            <>
                              <Tooltip title="Approve">
                                <IconButton
                                  size="small"
                                  color="success"
                                  onClick={() =>
                                    handleOpenPaymentDialog(request, "approve")
                                  }
                                >
                                  <CheckIcon />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Reject">
                                <IconButton
                                  size="small"
                                  color="error"
                                  onClick={() =>
                                    handleOpenPaymentDialog(request, "reject")
                                  }
                                >
                                  <CancelIcon />
                                </IconButton>
                              </Tooltip>
                            </>
                          )}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </>
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

      {/* Payment Action Dialog */}
      <Dialog
        open={openPaymentDialog}
        onClose={handleClosePaymentDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {paymentAction === "approve" ? "Approve" : "Reject"} Payment Request #
          {selectedPaymentRequest?.id}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 2 }}>
            {selectedPaymentRequest && (
              <Alert
                severity={paymentAction === "approve" ? "success" : "warning"}
              >
                <Typography variant="body2" gutterBottom>
                  <strong>Admin:</strong>{" "}
                  {selectedPaymentRequest.admin_username} (
                  {selectedPaymentRequest.admin_organization})
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>Agent Slots:</strong> +
                  {selectedPaymentRequest.num_agents}
                </Typography>
                <Typography variant="body2">
                  <strong>Amount:</strong> $
                  {selectedPaymentRequest.total_amount}
                </Typography>
              </Alert>
            )}

            {selectedPaymentRequest?.payment_notes && (
              <Alert severity="info">
                <Typography variant="caption" fontWeight={600}>
                  Admin Notes:
                </Typography>
                <Typography variant="body2">
                  {selectedPaymentRequest.payment_notes}
                </Typography>
              </Alert>
            )}

            <TextField
              label="Notes (Optional)"
              multiline
              rows={3}
              value={paymentNotes}
              onChange={(e) => setPaymentNotes(e.target.value)}
              fullWidth
              helperText="Add any notes about this decision"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClosePaymentDialog}>Cancel</Button>
          <Button
            onClick={handlePaymentAction}
            variant="contained"
            color={paymentAction === "approve" ? "success" : "error"}
          >
            {paymentAction === "approve" ? "Approve Payment" : "Reject Payment"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Wallet Address Dialog */}
      <Dialog
        open={openWalletDialog}
        onClose={handleCloseWalletDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Set Payment Wallet Address</DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 2 }}>
            <Alert severity="info">
              This wallet address will be shown to admins when they request
              agent slots. They'll send payments to this address.
            </Alert>

            <TextField
              label="Wallet Address/URL"
              value={newWalletAddress}
              onChange={(e) => setNewWalletAddress(e.target.value)}
              fullWidth
              required
              helperText="Enter cryptocurrency wallet address or payment link"
              multiline
              rows={2}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseWalletDialog}>Cancel</Button>
          <Button onClick={handleSaveWallet} variant="contained">
            Save Wallet Address
          </Button>
        </DialogActions>
      </Dialog>

      {/* Slot Adjustment Dialog */}
      <Dialog
        open={openSlotDialog}
        onClose={handleCloseSlotDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Adjust Agent Slots for {selectedAdmin?.username}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 2 }}>
            <Alert severity="info">
              <Typography variant="body2" gutterBottom>
                <strong>Current slots:</strong> {selectedAdmin?.max_agents || 0}
              </Typography>
              <Typography variant="caption">
                Enter a positive number to add slots or a negative number to
                remove slots.
              </Typography>
            </Alert>

            <TextField
              label="Slots Change"
              type="number"
              value={slotsChange}
              onChange={(e) => setSlotsChange(e.target.value)}
              fullWidth
              required
              helperText="Example: +5 to add 5 slots, -3 to remove 3 slots"
              placeholder="e.g., 5 or -3"
            />

            <TextField
              label="Reason for Adjustment"
              multiline
              rows={3}
              value={slotAdjustmentReason}
              onChange={(e) => setSlotAdjustmentReason(e.target.value)}
              fullWidth
              required
              helperText="Provide a clear reason for this adjustment (min 3 characters)"
              placeholder="e.g., Bonus slots for excellent performance"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseSlotDialog}>Cancel</Button>
          <Button
            onClick={handleAdjustSlots}
            variant="contained"
            disabled={!slotsChange || !slotAdjustmentReason}
          >
            Apply Adjustment
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default SuperAdmin;
