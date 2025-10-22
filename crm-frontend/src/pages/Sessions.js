import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Paper,
  Typography,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  TextField,
  Grid,
  Snackbar,
  Alert,
  CircularProgress,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import {
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  PlayArrow as PlayArrowIcon,
  Description as TranscriptIcon,
  Summarize as SummaryIcon,
  FilterList as FilterIcon,
} from "@mui/icons-material";
import { DateTimePicker } from "@mui/x-date-pickers/DateTimePicker";
import { sessionAPI, voiceAgentAPI, campaignAPI } from "../services/api";
import { formatDateTime } from "../utils/dateUtils";

function Sessions() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeVoiceSessions, setActiveVoiceSessions] = useState([]);
  const [summaries, setSummaries] = useState({});
  const [selectedRows, setSelectedRows] = useState([]);

  // Filter states
  const [campaignFilter, setCampaignFilter] = useState("");
  const [countryFilter, setCountryFilter] = useState("");
  const [interestFilter, setInterestFilter] = useState("");
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);

  // Available options for filters
  const [campaigns, setCampaigns] = useState([]);
  const [countries, setCountries] = useState([]);

  // Bulk operation states
  const [bulkProcessing, setBulkProcessing] = useState(false);
  const [bulkProgress, setBulkProgress] = useState(0);
  const [bulkTotal, setBulkTotal] = useState(0);
  const [bulkDialog, setBulkDialog] = useState({ open: false, type: "" });

  // Snackbar
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: "",
    severity: "success",
  });

  useEffect(() => {
    loadSessions();
    loadActiveVoiceSessions();
    loadCampaigns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [campaignFilter, countryFilter, interestFilter, startDate, endDate]);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const params = {};
      if (campaignFilter) params.campaign_id = campaignFilter;

      const response = await sessionAPI.getAll(params);
      let sessionsData = response.data;

      // Process sessions to extract duration from session_info if needed
      sessionsData = sessionsData.map((session) => {
        // If duration is missing but session_info has duration_seconds, use it
        if (!session.duration && session.session_info?.duration_seconds) {
          return {
            ...session,
            duration: Math.floor(session.session_info.duration_seconds),
          };
        }
        return session;
      });

      // Load summaries for CRM sessions
      await loadSummariesForSessions(sessionsData);

      // Extract unique countries from sessions data (before filtering)
      const uniqueCountries = [
        ...new Set(
          sessionsData.map((session) => session.lead_country).filter(Boolean)
        ),
      ].sort();
      setCountries(uniqueCountries);

      // Apply client-side filters after getting summaries
      sessionsData = applyClientFilters(sessionsData);

      setSessions(sessionsData);
    } catch (error) {
      console.error("Error loading sessions:", error);
    } finally {
      setLoading(false);
    }
  };

  const applyClientFilters = (sessionsData) => {
    let filtered = [...sessionsData];

    // Filter by country
    if (countryFilter) {
      filtered = filtered.filter(
        (session) => session.lead_country === countryFilter
      );
    }

    // Filter by interest status
    if (interestFilter) {
      filtered = filtered.filter((session) => {
        const summary = summaries[session.session_id];
        return summary && summary.status === interestFilter;
      });
    }

    // Filter by date range
    if (startDate) {
      filtered = filtered.filter(
        (session) => new Date(session.started_at) >= new Date(startDate)
      );
    }

    if (endDate) {
      filtered = filtered.filter(
        (session) => new Date(session.started_at) <= new Date(endDate)
      );
    }

    return filtered;
  };

  const loadCampaigns = async () => {
    try {
      const response = await campaignAPI.getAll({});
      setCampaigns(response.data.campaigns || []);
    } catch (error) {
      console.error("Error loading campaigns:", error);
    }
  };

  const loadSummariesForSessions = async (sessionsData) => {
    const summariesData = {};

    // Load summaries for each CRM session in parallel
    await Promise.all(
      sessionsData.map(async (session) => {
        if (!session.session_id) return;
        try {
          const summaryResponse = await voiceAgentAPI.getSummary(
            session.session_id
          );
          summariesData[session.session_id] = summaryResponse.data.summary;
        } catch (error) {
          // Summary not available for this session
          summariesData[session.session_id] = null;
        }
      })
    );

    // Merge with existing summaries
    setSummaries((prev) => ({ ...prev, ...summariesData }));
  };

  const loadActiveVoiceSessions = async () => {
    try {
      const response = await voiceAgentAPI.activeSessions();
      setActiveVoiceSessions(response.data.sessions);
    } catch (error) {
      console.error("Error loading active sessions:", error);
    }
  };

  const handleBulkTranscript = async () => {
    setBulkDialog({ open: false, type: "" });
    setBulkProcessing(true);
    setBulkProgress(0);
    setBulkTotal(selectedRows.length);

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < selectedRows.length; i++) {
      const sessionId = selectedRows[i];
      const session = sessions.find((s) => s.id === sessionId);

      if (!session || !session.session_id) {
        errorCount++;
        setBulkProgress(i + 1);
        continue;
      }

      try {
        await voiceAgentAPI.retranscribe(session.session_id);
        successCount++;
      } catch (error) {
        console.error(
          `Error transcribing session ${session.session_id}:`,
          error
        );
        errorCount++;
      }

      setBulkProgress(i + 1);
    }

    setBulkProcessing(false);
    showSnackbar(
      `Bulk transcription complete: ${successCount} succeeded, ${errorCount} failed`,
      errorCount > 0 ? "warning" : "success"
    );

    // Reload sessions to get updated data
    setTimeout(() => loadSessions(), 2000);
  };

  const handleBulkSummary = async () => {
    setBulkDialog({ open: false, type: "" });
    setBulkProcessing(true);
    setBulkProgress(0);
    setBulkTotal(selectedRows.length);

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < selectedRows.length; i++) {
      const sessionId = selectedRows[i];
      const session = sessions.find((s) => s.id === sessionId);

      if (!session || !session.session_id) {
        errorCount++;
        setBulkProgress(i + 1);
        continue;
      }

      try {
        await voiceAgentAPI.generateSummary(session.session_id, "English");
        successCount++;
      } catch (error) {
        console.error(
          `Error generating summary for session ${session.session_id}:`,
          error
        );
        errorCount++;
      }

      setBulkProgress(i + 1);
    }

    setBulkProcessing(false);
    showSnackbar(
      `Bulk summary generation complete: ${successCount} succeeded, ${errorCount} failed`,
      errorCount > 0 ? "warning" : "success"
    );

    // Reload sessions to get updated summaries
    setTimeout(() => loadSessions(), 2000);
  };

  const showSnackbar = (message, severity) => {
    setSnackbar({ open: true, message, severity });
  };

  const clearFilters = () => {
    setCampaignFilter("");
    setCountryFilter("");
    setInterestFilter("");
    setStartDate(null);
    setEndDate(null);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "answered":
      case "completed":
        return <CheckCircleIcon color="success" fontSize="small" />;
      case "rejected":
      case "no_answer":
      case "failed":
        return <CancelIcon color="error" fontSize="small" />;
      case "in_call":
      case "dialing":
      case "ringing":
        return <PhoneIcon color="primary" fontSize="small" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "answered":
      case "completed":
        return "success";
      case "rejected":
      case "no_answer":
      case "failed":
        return "error";
      case "in_call":
      case "dialing":
      case "ringing":
        return "primary";
      default:
        return "default";
    }
  };

  const columns = [
    {
      field: "started_at",
      headerName: "Date/Time",
      flex: 1.2,
      minWidth: 180,
      valueFormatter: (params) => {
        return formatDateTime(params.value);
      },
    },
    {
      field: "called_number",
      headerName: "Phone Number",
      flex: 1,
      minWidth: 150,
    },
    {
      field: "lead_country",
      headerName: "Country",
      flex: 0.8,
      minWidth: 100,
      renderCell: (params) => params.value || "-",
    },
    {
      field: "status",
      headerName: "Status",
      flex: 0.9,
      minWidth: 130,
      renderCell: (params) => (
        <Chip
          icon={getStatusIcon(params.value)}
          label={params.value.replace("_", " ")}
          size="small"
          color={getStatusColor(params.value)}
        />
      ),
    },
    {
      field: "duration",
      headerName: "Duration",
      flex: 0.7,
      minWidth: 100,
      valueFormatter: (params) => {
        if (!params.value) return "-";
        const minutes = Math.floor(params.value / 60);
        const seconds = params.value % 60;
        return `${minutes}:${seconds.toString().padStart(2, "0")}`;
      },
    },
    {
      field: "campaign_id",
      headerName: "Campaign",
      flex: 0.8,
      minWidth: 120,
      renderCell: (params) => {
        return params.value ? (
          <Button
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              navigate(`/campaigns/${params.value}`);
            }}
          >
            View
          </Button>
        ) : (
          "-"
        );
      },
    },
    {
      field: "interest_status",
      headerName: "Interest Status",
      flex: 0.9,
      minWidth: 120,
      renderCell: (params) => {
        // Display summary interest status for CRM sessions
        const summary = summaries[params.row.session_id];
        if (!summary) return "-";
        return (
          <Chip
            label={summary.status.toUpperCase()}
            size="small"
            color={summary.status === "interested" ? "success" : "default"}
          />
        );
      },
    },
    {
      field: "actions",
      headerName: "Actions",
      flex: 0.9,
      minWidth: 140,
      sortable: false,
      renderCell: (params) => (
        <Button
          size="small"
          variant="contained"
          startIcon={<PlayArrowIcon />}
          onClick={() => navigate(`/sessions/${params.row.session_id}`)}
          sx={{
            borderRadius: "20px",
            textTransform: "none",
            fontWeight: 600,
            padding: "4px 12px",
            background: "linear-gradient(135deg, #90C4F5 0%, #85B4E8 100%)",
            color: "#1565C0",
            boxShadow: "0 2px 8px rgba(144, 196, 245, 0.3)",
            "&:hover": {
              background: "linear-gradient(135deg, #A8D4FF 0%, #90C4F5 100%)",
              boxShadow: "0 4px 12px rgba(144, 196, 245, 0.4)",
              transform: "translateY(-1px)",
            },
            transition: "all 0.2s ease-in-out",
          }}
        >
          Review
        </Button>
      ),
    },
  ];

  return (
    <Box className="fade-in">
      {/* Bulk Processing Progress */}
      {bulkProcessing && (
        <Paper
          className="glass-effect ios-blur-container"
          sx={{
            p: 3,
            mb: 3,
            background:
              "linear-gradient(135deg, rgba(92, 138, 166, 0.08) 0%, rgba(92, 138, 166, 0.03) 100%)",
            border: "1px solid rgba(92, 138, 166, 0.2)",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <CircularProgress size={24} />
            <Typography variant="h6" fontWeight={600}>
              Processing {bulkProgress} of {bulkTotal}...
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={(bulkProgress / bulkTotal) * 100}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Paper>
      )}

      {/* Filters */}
      <Paper
        className="glass-effect ios-blur-container"
        sx={{
          p: 3,
          mb: 3,
          animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
          <FilterIcon />
          <Typography variant="h6" fontWeight={700}>
            Filters
          </Typography>
        </Box>
        <Grid container spacing={1.5}>
          <Grid item xs={12} sm={4} md={2}>
            <DateTimePicker
              label="Start Date/Time"
              value={startDate}
              onChange={setStartDate}
              ampm={false}
              views={["year", "month", "day", "hours", "minutes"]}
              renderInput={(params) => (
                <TextField {...params} fullWidth size="small" />
              )}
              slotProps={{
                textField: {
                  size: "small",
                  fullWidth: true,
                  inputProps: {
                    readOnly: true,
                  },
                },
                actionBar: {
                  actions: ["accept", "cancel", "clear"],
                },
              }}
            />
          </Grid>
          <Grid item xs={12} sm={4} md={2}>
            <DateTimePicker
              label="End Date/Time"
              value={endDate}
              onChange={setEndDate}
              ampm={false}
              views={["year", "month", "day", "hours", "minutes"]}
              renderInput={(params) => (
                <TextField {...params} fullWidth size="small" />
              )}
              slotProps={{
                textField: {
                  size: "small",
                  fullWidth: true,
                  inputProps: {
                    readOnly: true,
                  },
                },
                actionBar: {
                  actions: ["accept", "cancel", "clear"],
                },
              }}
            />
          </Grid>
          <Grid item xs={12} sm={4} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Campaign</InputLabel>
              <Select
                value={campaignFilter}
                onChange={(e) => setCampaignFilter(e.target.value)}
                label="Campaign"
              >
                <MenuItem value="">All</MenuItem>
                {campaigns.map((campaign) => (
                  <MenuItem key={campaign.id} value={campaign.id}>
                    {campaign.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={4} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Country</InputLabel>
              <Select
                value={countryFilter}
                onChange={(e) => setCountryFilter(e.target.value)}
                label="Country"
              >
                <MenuItem value="">All</MenuItem>
                {countries.map((country) => (
                  <MenuItem key={country} value={country}>
                    {country}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={4} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Interest Status</InputLabel>
              <Select
                value={interestFilter}
                onChange={(e) => setInterestFilter(e.target.value)}
                label="Interest Status"
              >
                <MenuItem value="">All</MenuItem>
                <MenuItem value="interested">Interested</MenuItem>
                <MenuItem value="not_interested">Not Interested</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={4} md={2}>
            <Button
              fullWidth
              variant="outlined"
              onClick={clearFilters}
              sx={{ height: "40px" }}
            >
              Clear Filters
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Active Sessions */}
      {activeVoiceSessions.length > 0 && (
        <Paper
          className="glass-effect-colored ios-blur-container"
          sx={{
            p: 3,
            mb: 4,
            background:
              "linear-gradient(135deg, rgba(107, 154, 90, 0.08) 0%, rgba(107, 154, 90, 0.03) 100%)",
            border: "1px solid rgba(107, 154, 90, 0.2)",
            animation:
              "pulse 2s ease-in-out infinite, slideUp 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
            <Typography variant="h6">🔴</Typography>
            <Typography variant="h6" fontWeight={700}>
              Active Voice Sessions
            </Typography>
          </Box>
          <Box>
            {activeVoiceSessions.map((session) => (
              <Box
                key={session.session_id}
                sx={{
                  p: 2,
                  borderBottom: "1px solid",
                  borderColor: "divider",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  "&:last-child": { borderBottom: 0 },
                }}
              >
                <Box display="flex" alignItems="center" gap={2}>
                  <PhoneIcon color="primary" />
                  <Box>
                    <Typography variant="body1">
                      {session.caller_id} → {session.called_number}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Duration: {Math.round(session.duration_seconds)}s
                    </Typography>
                  </Box>
                </Box>
                <Chip label="Active" color="success" size="small" />
              </Box>
            ))}
          </Box>
        </Paper>
      )}

      {/* CRM Sessions */}
      <Paper
        className="glass-effect ios-blur-container"
        sx={{
          mb: 4,
          overflow: "hidden",
          animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
          animationDelay: "0.1s",
          animationFillMode: "both",
          position: "relative",
        }}
      >
        {/* Floating Bulk Action Buttons - Appear when rows are selected */}
        {selectedRows.length > 0 && (
          <Box
            sx={{
              position: "absolute",
              top: 60,
              right: 20,
              zIndex: 1000,
              display: "flex",
              alignItems: "center",
              gap: 2,
              background: "rgba(255, 255, 255, 0.95)",
              backdropFilter: "blur(10px)",
              padding: "12px 16px",
              borderRadius: "12px",
              boxShadow: "0 4px 20px rgba(0, 0, 0, 0.15)",
              border: "1px solid rgba(0, 0, 0, 0.1)",
              animation: "slideInRight 0.3s ease-out",
            }}
          >
            <Chip
              label={`${selectedRows.length} selected`}
              size="small"
              color="primary"
            />
            <Button
              variant="contained"
              size="small"
              startIcon={<TranscriptIcon />}
              onClick={() => setBulkDialog({ open: true, type: "transcript" })}
              disabled={bulkProcessing}
              sx={{
                background: "linear-gradient(135deg, #5C8AA6 0%, #426D87 100%)",
                "&:hover": {
                  background:
                    "linear-gradient(135deg, #6B9FB5 0%, #517C96 100%)",
                },
              }}
            >
              Bulk Transcript
            </Button>
            <Button
              variant="contained"
              size="small"
              startIcon={<SummaryIcon />}
              onClick={() => setBulkDialog({ open: true, type: "summary" })}
              disabled={bulkProcessing}
              sx={{
                background: "linear-gradient(135deg, #6B9A5A 0%, #517542 100%)",
                "&:hover": {
                  background:
                    "linear-gradient(135deg, #7AAA69 0%, #608451 100%)",
                },
              }}
            >
              Bulk Summary
            </Button>
          </Box>
        )}
        <DataGrid
          rows={sessions}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[10, 25, 50]}
          loading={loading}
          autoHeight
          checkboxSelection
          disableRowSelectionOnClick
          onRowSelectionModelChange={(newSelection) => {
            setSelectedRows(newSelection);
          }}
          rowSelectionModel={selectedRows}
          sx={{
            "& .MuiDataGrid-cell": {
              borderBottom: "3px solid rgba(224, 224, 224, 0.8)",
              borderRight: "1px solid rgba(224, 224, 224, 0.3)",
              fontSize: "0.875rem",
              "&:focus": {
                outline: "none",
              },
              "&:focus-within": {
                outline: "none",
              },
            },
            "& .MuiDataGrid-cell:focus-within": {
              outline: "none",
            },
            "& .MuiDataGrid-columnHeader:focus": {
              outline: "none",
            },
            "& .MuiDataGrid-columnHeader:focus-within": {
              outline: "none",
            },
            "& .MuiCheckbox-root": {
              padding: "4px",
            },
            "& .MuiDataGrid-columnHeaders": {
              backgroundColor: "#F3B3F3",
              backdropFilter: "blur(10px)",
              borderBottom: "2px solid rgba(238, 130, 238, 0.5)",
              "& .MuiDataGrid-columnHeaderTitle": {
                fontWeight: 700,
                fontSize: "0.875rem",
              },
            },
            "& .MuiDataGrid-columnSeparator": {
              visibility: "visible !important",
              opacity: "1 !important",
              color: "rgba(255, 255, 255, 0.8)",
            },
            "& .MuiDataGrid-footerContainer": {
              borderTop: "2px solid rgba(200, 92, 60, 0.15)",
              backgroundColor: "rgba(248, 243, 239, 0.8)",
              backdropFilter: "blur(10px)",
            },
            "& .MuiDataGrid-row": {
              backgroundColor: "rgba(255, 255, 255, 0.6)",
              "&:hover": {
                backgroundColor: "rgba(200, 92, 60, 0.08)",
              },
              cursor: "default",
            },
          }}
          slotProps={{
            baseCheckbox: {
              disableRipple: true,
            },
          }}
        />
      </Paper>

      {/* Bulk Operation Confirmation Dialog */}
      <Dialog
        open={bulkDialog.open}
        onClose={() => setBulkDialog({ open: false, type: "" })}
      >
        <DialogTitle>
          Confirm Bulk{" "}
          {bulkDialog.type === "transcript"
            ? "Transcription"
            : "Summary Generation"}
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to{" "}
            {bulkDialog.type === "transcript"
              ? "re-transcribe"
              : "generate summaries for"}{" "}
            {selectedRows.length} selected session
            {selectedRows.length > 1 ? "s" : ""}?
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            This operation may take several minutes depending on the number of
            sessions.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBulkDialog({ open: false, type: "" })}>
            Cancel
          </Button>
          <Button
            onClick={
              bulkDialog.type === "transcript"
                ? handleBulkTranscript
                : handleBulkSummary
            }
            variant="contained"
            color="primary"
          >
            Confirm
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

export default Sessions;
