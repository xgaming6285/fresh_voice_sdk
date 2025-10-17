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
  Alert,
  AlertTitle,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Link,
} from "@mui/material";
import {
  AccountBalance as WalletIcon,
  ShoppingCart as CartIcon,
  CheckCircle as CheckIcon,
  Business as BusinessIcon,
  Pending as PendingIcon,
  Cancel as CancelIcon,
} from "@mui/icons-material";
import api from "../services/api";

const PRICE_PER_AGENT = 300;

function Billing() {
  const [billingInfo, setBillingInfo] = useState(null);
  const [paymentRequests, setPaymentRequests] = useState([]);
  const [slotAdjustments, setSlotAdjustments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Dialog state
  const [openDialog, setOpenDialog] = useState(false);
  const [numAgents, setNumAgents] = useState(1);
  const [paymentNotes, setPaymentNotes] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError("");

      const [billingRes, requestsRes, adjustmentsRes] = await Promise.all([
        api.billingAPI.getInfo(),
        api.billingAPI.getRequests(),
        api.billingAPI.getSlotAdjustments(),
      ]);

      setBillingInfo(billingRes.data);
      setPaymentRequests(requestsRes.data);
      setSlotAdjustments(adjustmentsRes.data);
    } catch (err) {
      console.error("Error loading billing data:", err);
      setError(err.response?.data?.detail || "Failed to load billing data");
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDialog = () => {
    setNumAgents(1);
    setPaymentNotes("");
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleSubmitRequest = async () => {
    try {
      setError("");

      await api.billingAPI.createRequest({
        num_agents: numAgents,
        payment_notes: paymentNotes,
      });

      setSuccess(
        `Payment request created for ${numAgents} agent${
          numAgents > 1 ? "s" : ""
        }. Total: $${numAgents * PRICE_PER_AGENT}`
      );
      handleCloseDialog();
      loadData();
    } catch (err) {
      console.error("Error creating payment request:", err);
      setError(
        err.response?.data?.detail || "Failed to create payment request"
      );
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "approved":
        return "success";
      case "pending":
        return "warning";
      case "rejected":
        return "error";
      default:
        return "default";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "approved":
        return <CheckIcon />;
      case "pending":
        return <PendingIcon />;
      case "rejected":
        return <CancelIcon />;
      default:
        return null;
    }
  };

  if (loading) {
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

  const totalCost = numAgents * PRICE_PER_AGENT;
  const hasPendingRequest = paymentRequests.some((r) => r.status === "pending");

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box mb={4}>
        <Typography variant="h4" gutterBottom>
          💳 Billing & Payment
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage your agent slots and payments
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

      {/* Subscription Status Alert */}
      {billingInfo && !billingInfo.is_subscription_active && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <AlertTitle>🚫 Subscription Expired</AlertTitle>
          Your subscription has expired. To continue using our software, please
          purchase agent slots below and complete the payment.
        </Alert>
      )}
      {billingInfo &&
        billingInfo.is_subscription_active &&
        billingInfo.days_remaining !== null &&
        billingInfo.days_remaining <= 7 && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            <AlertTitle>
              ⚠️ Subscription Expiring in {billingInfo.days_remaining} Day
              {billingInfo.days_remaining !== 1 ? "s" : ""}
            </AlertTitle>
            Your subscription will expire on{" "}
            {new Date(billingInfo.subscription_end_date).toLocaleDateString()}.
            Renew now to avoid service interruption.
          </Alert>
        )}

      {/* Billing Info Cards */}
      <Grid container spacing={3} mb={4}>
        {/* Subscription Status Card */}
        <Grid item xs={12} md={3}>
          <Card sx={{ height: "100%", minHeight: 140 }}>
            <CardContent
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              <Box display="flex" alignItems="center">
                <BusinessIcon
                  color={
                    billingInfo?.is_subscription_active ? "success" : "error"
                  }
                  sx={{ mr: 1, fontSize: 32 }}
                />
                <Box>
                  <Chip
                    label={
                      billingInfo?.is_subscription_active ? "ACTIVE" : "EXPIRED"
                    }
                    color={
                      billingInfo?.is_subscription_active ? "success" : "error"
                    }
                    size="small"
                  />
                  <Typography variant="body2" color="text.secondary" mt={0.5}>
                    Subscription Status
                  </Typography>
                  {billingInfo?.subscription_end_date && (
                    <Typography variant="caption" color="text.secondary">
                      {billingInfo.is_subscription_active
                        ? `${billingInfo.days_remaining}d remaining`
                        : "Expired"}
                    </Typography>
                  )}
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ height: "100%", minHeight: 140 }}>
            <CardContent
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              <Box display="flex" alignItems="center">
                <CartIcon color="primary" sx={{ mr: 1, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {billingInfo?.max_agents || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Agent Slots
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ height: "100%", minHeight: 140 }}>
            <CardContent
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              <Box display="flex" alignItems="center">
                <CheckIcon color="success" sx={{ mr: 1, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {billingInfo?.current_agents || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Agents Created
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ height: "100%", minHeight: 140 }}>
            <CardContent
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              <Box display="flex" alignItems="center">
                <PendingIcon color="info" sx={{ mr: 1, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {billingInfo?.available_slots || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Available Slots
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Pricing Card */}
      <Card
        sx={{ mb: 4, bgcolor: "primary.light", color: "primary.contrastText" }}
      >
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <Typography variant="h5" gutterBottom>
                💰 ${PRICE_PER_AGENT}/month per agent
              </Typography>
              <Typography variant="body2">
                Purchase additional agent slots to expand your team. Each agent
                can manage their own leads and campaigns independently.
              </Typography>
            </Grid>
            <Grid item xs={12} md={4} textAlign="right">
              <Button
                variant="contained"
                size="large"
                onClick={handleOpenDialog}
                disabled={hasPendingRequest}
                sx={{
                  bgcolor: "white",
                  color: "primary.main",
                  "&:hover": { bgcolor: "grey.100" },
                }}
              >
                {hasPendingRequest ? "Payment Pending" : "Purchase Agent Slots"}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Payment Wallet Info */}
      {billingInfo?.payment_wallet && (
        <Alert severity="info" icon={<WalletIcon />} sx={{ mb: 4 }}>
          <Typography variant="body2" fontWeight={600} gutterBottom>
            Payment Wallet Address:
          </Typography>
          <Link
            href={billingInfo.payment_wallet}
            target="_blank"
            rel="noopener noreferrer"
            sx={{ wordBreak: "break-all" }}
          >
            {billingInfo.payment_wallet}
          </Link>
          <Typography variant="caption" display="block" mt={1}>
            Send your payment to this wallet and we'll verify it.
          </Typography>
        </Alert>
      )}

      {/* Payment Requests History */}
      <Paper>
        <Box p={2}>
          <Typography variant="h6">Payment Request History</Typography>
        </Box>

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Request ID</TableCell>
                <TableCell>Agent Slots</TableCell>
                <TableCell>Total Amount</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Notes</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paymentRequests.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    <Typography variant="body2" color="text.secondary" py={3}>
                      No payment requests yet
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                paymentRequests.map((request) => (
                  <TableRow key={request.id}>
                    <TableCell>#{request.id}</TableCell>
                    <TableCell>{request.num_agents}</TableCell>
                    <TableCell fontWeight={600}>
                      ${request.total_amount.toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(request.status)}
                        label={request.status.toUpperCase()}
                        size="small"
                        color={getStatusColor(request.status)}
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(request.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      {request.admin_notes || request.payment_notes || "-"}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Slot Adjustments History */}
      {slotAdjustments.length > 0 && (
        <Paper sx={{ mt: 4 }}>
          <Box p={2}>
            <Typography variant="h6">Manual Slot Adjustments</Typography>
            <Typography variant="caption" color="text.secondary">
              Adjustments made by superadmin
            </Typography>
          </Box>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Date</TableCell>
                  <TableCell>Adjusted By</TableCell>
                  <TableCell>Change</TableCell>
                  <TableCell>Previous</TableCell>
                  <TableCell>New Total</TableCell>
                  <TableCell>Reason</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {slotAdjustments.map((adjustment) => (
                  <TableRow key={adjustment.id}>
                    <TableCell>
                      {new Date(adjustment.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>{adjustment.adjusted_by_username}</TableCell>
                    <TableCell>
                      <Chip
                        label={
                          adjustment.slots_change > 0
                            ? `+${adjustment.slots_change}`
                            : adjustment.slots_change
                        }
                        size="small"
                        color={
                          adjustment.slots_change > 0 ? "success" : "error"
                        }
                      />
                    </TableCell>
                    <TableCell>{adjustment.previous_max_agents}</TableCell>
                    <TableCell fontWeight={600}>
                      {adjustment.new_max_agents}
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {adjustment.reason}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* Purchase Dialog */}
      <Dialog
        open={openDialog}
        onClose={handleCloseDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Purchase Agent Slots</DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 2 }}>
            <TextField
              label="Number of Agents"
              type="number"
              value={numAgents}
              onChange={(e) =>
                setNumAgents(Math.max(1, parseInt(e.target.value) || 1))
              }
              fullWidth
              required
              InputProps={{ inputProps: { min: 1 } }}
            />

            <Alert severity="info">
              <Typography variant="body2" fontWeight={600}>
                Total Cost: ${totalCost}
              </Typography>
              <Typography variant="caption">
                ${PRICE_PER_AGENT} × {numAgents} agent{numAgents > 1 ? "s" : ""}
              </Typography>
            </Alert>

            {billingInfo?.payment_wallet && (
              <Alert severity="warning">
                <Typography variant="body2" gutterBottom>
                  <strong>Payment Instructions:</strong>
                </Typography>
                <Typography variant="caption" component="div">
                  1. Send ${totalCost} to the wallet address shown above
                  <br />
                  2. Submit this payment request
                  <br />
                  3. We'll verify your payment manually
                  <br />
                  4. Your agent slots will be activated once approved
                </Typography>
              </Alert>
            )}

            <TextField
              label="Payment Notes (Optional)"
              multiline
              rows={3}
              value={paymentNotes}
              onChange={(e) => setPaymentNotes(e.target.value)}
              fullWidth
              helperText="Add transaction ID or any notes about your payment"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSubmitRequest} variant="contained">
            Submit Payment Request
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default Billing;
