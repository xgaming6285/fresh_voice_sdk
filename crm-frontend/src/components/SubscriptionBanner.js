import React, { useState, useEffect } from "react";
import { Alert, AlertTitle, Button, Box, LinearProgress } from "@mui/material";
import { Warning as WarningIcon } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import api from "../services/api";
import { useAuth } from "../contexts/AuthContext";

function SubscriptionBanner() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [billingInfo, setBillingInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadBillingInfo();
  }, []);

  const loadBillingInfo = async () => {
    // Only load for admins
    if (user?.role !== "admin") {
      setLoading(false);
      return;
    }

    try {
      const res = await api.billingAPI.getInfo();
      setBillingInfo(res.data);
    } catch (err) {
      console.error("Error loading billing info:", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return null;

  // Don't show banner for agents or if no billing info
  if (user?.role === "agent" || !billingInfo) {
    // For agents, check if subscription is inactive by trying a simple operation
    // If they get 403, we'll show them the message in the error
    return null;
  }

  // Admin with no subscription
  if (!billingInfo.is_subscription_active) {
    return (
      <Alert
        severity="error"
        icon={<WarningIcon />}
        sx={{ mb: 3 }}
        action={
          <Button
            color="inherit"
            size="small"
            onClick={() => navigate("/billing")}
          >
            Renew Now
          </Button>
        }
      >
        <AlertTitle>Subscription Expired</AlertTitle>
        Your subscription has expired. To continue using our software, please
        update your payment in the Billing page. You can view your data but
        cannot make any changes until renewal.
      </Alert>
    );
  }

  // Admin with expiring subscription (less than 7 days)
  if (billingInfo.days_remaining !== null && billingInfo.days_remaining <= 7) {
    return (
      <Alert
        severity="warning"
        sx={{ mb: 3 }}
        action={
          <Button
            color="inherit"
            size="small"
            onClick={() => navigate("/billing")}
          >
            Renew
          </Button>
        }
      >
        <AlertTitle>
          Subscription Expiring in {billingInfo.days_remaining} day
          {billingInfo.days_remaining !== 1 ? "s" : ""}
        </AlertTitle>
        Your subscription will expire soon. Renew now to avoid service
        interruption.
      </Alert>
    );
  }

  return null;
}

export default SubscriptionBanner;
