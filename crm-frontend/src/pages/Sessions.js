import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import {
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  PlayCircle as PlayIcon,
  Refresh as RefreshIcon,
  Headset as HeadsetIcon,
} from "@mui/icons-material";
import { sessionAPI, voiceAgentAPI } from "../services/api";

function Sessions() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [recordings, setRecordings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState("");
  const [activeVoiceSessions, setActiveVoiceSessions] = useState([]);

  useEffect(() => {
    loadSessions();
    loadRecordings();
    loadActiveVoiceSessions();
  }, [statusFilter]);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const params = statusFilter ? { status: statusFilter } : {};
      const response = await sessionAPI.getAll(params);
      setSessions(response.data);
    } catch (error) {
      console.error("Error loading sessions:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadRecordings = async () => {
    try {
      const response = await voiceAgentAPI.recordings();
      setRecordings(response.data.recordings);
    } catch (error) {
      console.error("Error loading recordings:", error);
    }
  };

  const loadActiveVoiceSessions = async () => {
    try {
      const response = await voiceAgentAPI.activeSessions();
      setActiveVoiceSessions(response.data.sessions);
    } catch (error) {
      console.error("Error loading active sessions:", error);
    }
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
      width: 180,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      },
    },
    {
      field: "called_number",
      headerName: "Phone Number",
      width: 150,
    },
    {
      field: "status",
      headerName: "Status",
      width: 130,
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
      width: 100,
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
      width: 120,
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
      field: "transcript_status",
      headerName: "Transcript",
      width: 120,
      renderCell: (params) => {
        if (!params.value) return "-";
        return (
          <Chip
            label={params.value}
            size="small"
            color={params.value === "completed" ? "success" : "default"}
          />
        );
      },
    },
    {
      field: "actions",
      headerName: "Actions",
      width: 120,
      sortable: false,
      renderCell: (params) => (
        <Button
          size="small"
          variant="outlined"
          startIcon={<HeadsetIcon />}
          onClick={() => navigate(`/sessions/${params.row.session_id}`)}
        >
          Review
        </Button>
      ),
    },
  ];

  const recordingColumns = [
    {
      field: "start_time",
      headerName: "Date/Time",
      width: 180,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      },
    },
    {
      field: "caller_id",
      headerName: "Caller",
      width: 150,
    },
    {
      field: "called_number",
      headerName: "Called",
      width: 150,
    },
    {
      field: "duration_seconds",
      headerName: "Duration",
      width: 100,
      valueFormatter: (params) => {
        if (!params.value) return "-";
        const minutes = Math.floor(params.value / 60);
        const seconds = params.value % 60;
        return `${minutes}:${seconds.toString().padStart(2, "0")}`;
      },
    },
    {
      field: "has_transcripts",
      headerName: "Transcripts",
      width: 120,
      renderCell: (params) => (
        <Chip
          label={params.value ? "Available" : "None"}
          size="small"
          color={params.value ? "success" : "default"}
        />
      ),
    },
    {
      field: "audio_files",
      headerName: "Audio Files",
      width: 200,
      renderCell: (params) => {
        const fileCount = Object.keys(params.value || {}).length;
        return `${fileCount} files`;
      },
    },
    {
      field: "actions",
      headerName: "Actions",
      width: 120,
      sortable: false,
      renderCell: (params) => (
        <Button
          size="small"
          variant="outlined"
          startIcon={<PlayIcon />}
          onClick={() => navigate(`/sessions/${params.row.session_id}`)}
        >
          Review
        </Button>
      ),
    },
  ];

  return (
    <Box className="fade-in">
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={4}
      >
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
          📞 Call Sessions
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <FormControl
            size="small"
            sx={{
              minWidth: 150,
              "& .MuiOutlinedInput-root": {
                borderRadius: 2,
                "&:hover fieldset": {
                  borderColor: "primary.main",
                },
              },
            }}
          >
            <InputLabel>Status Filter</InputLabel>
            <Select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              label="Status Filter"
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="answered">Answered</MenuItem>
              <MenuItem value="rejected">Rejected</MenuItem>
              <MenuItem value="no_answer">No Answer</MenuItem>
              <MenuItem value="failed">Failed</MenuItem>
              <MenuItem value="in_call">In Call</MenuItem>
            </Select>
          </FormControl>
          <IconButton
            onClick={() => {
              loadSessions();
              loadRecordings();
              loadActiveVoiceSessions();
            }}
            sx={{
              backgroundColor: "primary.main",
              color: "white",
              "&:hover": {
                backgroundColor: "primary.dark",
                transform: "rotate(180deg)",
              },
              transition: "all 0.5s ease",
            }}
          >
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Active Sessions */}
      {activeVoiceSessions.length > 0 && (
        <Paper
          className="slide-in"
          sx={{
            p: 3,
            mb: 4,
            borderRadius: 3,
            background:
              "linear-gradient(135deg, rgba(107, 154, 90, 0.1) 0%, rgba(255, 255, 255, 0.95) 100%)",
            border: "1px solid rgba(107, 154, 90, 0.3)",
            boxShadow: "0 8px 24px rgba(107, 154, 90, 0.2)",
            animation: "pulse 2s ease-in-out infinite",
          }}
        >
          <Typography variant="h6" gutterBottom fontWeight={700}>
            🔴 Active Voice Sessions
          </Typography>
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
        className="fade-in"
        sx={{
          mb: 4,
          borderRadius: 3,
          background:
            "linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 243, 239, 0.95) 100%)",
          border: "1px solid rgba(200, 92, 60, 0.1)",
          boxShadow: "0 8px 24px rgba(200, 92, 60, 0.12)",
          overflow: "hidden",
        }}
      >
        <Box p={3}>
          <Typography variant="h6" gutterBottom fontWeight={700}>
            📊 CRM Call Sessions
          </Typography>
        </Box>
        <DataGrid
          rows={sessions}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[10, 25, 50]}
          loading={loading}
          autoHeight
          disableSelectionOnClick
        />
      </Paper>

      {/* Voice Agent Recordings */}
      <Paper
        className="fade-in"
        sx={{
          borderRadius: 3,
          background:
            "linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 243, 239, 0.95) 100%)",
          border: "1px solid rgba(200, 92, 60, 0.1)",
          boxShadow: "0 8px 24px rgba(200, 92, 60, 0.12)",
          overflow: "hidden",
        }}
      >
        <Box p={3}>
          <Typography variant="h6" gutterBottom fontWeight={700}>
            🎙️ Voice Agent Recordings
          </Typography>
        </Box>
        <DataGrid
          rows={recordings}
          columns={recordingColumns}
          pageSize={10}
          rowsPerPageOptions={[10, 25, 50]}
          autoHeight
          disableSelectionOnClick
          getRowId={(row) => row.session_id}
        />
      </Paper>
    </Box>
  );
}

export default Sessions;
