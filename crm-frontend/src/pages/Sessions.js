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
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import {
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  PlayCircle as PlayIcon,
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
      flex: 1.2,
      minWidth: 180,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      },
    },
    {
      field: "called_number",
      headerName: "Phone Number",
      flex: 1,
      minWidth: 150,
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
      field: "transcript_status",
      headerName: "Transcript",
      flex: 0.8,
      minWidth: 120,
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
      flex: 0.9,
      minWidth: 140,
      sortable: false,
      renderCell: (params) => (
        <Button
          size="small"
          variant="contained"
          startIcon={<HeadsetIcon />}
          onClick={() => navigate(`/sessions/${params.row.session_id}`)}
          sx={{
            borderRadius: "24px",
            textTransform: "none",
            fontWeight: 600,
            background: "linear-gradient(135deg, #F5A890 0%, #E89B85 100%)",
            color: "#8B4513",
            boxShadow: "0 2px 8px rgba(245, 168, 144, 0.3)",
            "&:hover": {
              background: "linear-gradient(135deg, #FFBFA8 0%, #F5A890 100%)",
              boxShadow: "0 4px 12px rgba(245, 168, 144, 0.4)",
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

  const recordingColumns = [
    {
      field: "start_time",
      headerName: "Date/Time",
      flex: 1.2,
      minWidth: 180,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      },
    },
    {
      field: "caller_id",
      headerName: "Caller",
      flex: 1,
      minWidth: 150,
    },
    {
      field: "called_number",
      headerName: "Called",
      flex: 1,
      minWidth: 150,
    },
    {
      field: "duration_seconds",
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
      field: "has_transcripts",
      headerName: "Transcripts",
      flex: 0.9,
      minWidth: 120,
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
      flex: 1.3,
      minWidth: 140,
      renderCell: (params) => {
        const fileCount = Object.keys(params.value || {}).length;
        return `${fileCount} files`;
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
          startIcon={<PlayIcon />}
          onClick={() => navigate(`/sessions/${params.row.session_id}`)}
          sx={{
            borderRadius: "24px",
            textTransform: "none",
            fontWeight: 600,
            background: "linear-gradient(135deg, #F5A890 0%, #E89B85 100%)",
            color: "#8B4513",
            boxShadow: "0 2px 8px rgba(245, 168, 144, 0.3)",
            "&:hover": {
              background: "linear-gradient(135deg, #FFBFA8 0%, #F5A890 100%)",
              boxShadow: "0 4px 12px rgba(245, 168, 144, 0.4)",
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
        }}
      >
        <Box p={3}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              mb: 2,
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Typography variant="h6">📊</Typography>
              <Typography variant="h6" fontWeight={700}>
                CRM Call Sessions
              </Typography>
            </Box>
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
          </Box>
        </Box>
        <DataGrid
          rows={sessions}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[10, 25, 50]}
          loading={loading}
          autoHeight
          disableSelectionOnClick
          sx={{
            "& .MuiDataGrid-cell": {
              borderBottom: "1px solid rgba(224, 224, 224, 0.2)",
              fontSize: "0.875rem",
            },
            "& .MuiDataGrid-columnHeaders": {
              backgroundColor: "rgba(248, 243, 239, 0.8)",
              backdropFilter: "blur(10px)",
              borderBottom: "2px solid rgba(200, 92, 60, 0.15)",
              fontWeight: 600,
              fontSize: "0.875rem",
            },
            "& .MuiDataGrid-footerContainer": {
              borderTop: "2px solid rgba(200, 92, 60, 0.15)",
              backgroundColor: "rgba(248, 243, 239, 0.8)",
              backdropFilter: "blur(10px)",
            },
            "& .MuiDataGrid-row": {
              "&:hover": {
                backgroundColor: "rgba(200, 92, 60, 0.04)",
              },
            },
          }}
        />
      </Paper>

      {/* Voice Agent Recordings */}
      <Paper
        className="glass-effect ios-blur-container"
        sx={{
          overflow: "hidden",
          animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
          animationDelay: "0.2s",
          animationFillMode: "both",
        }}
      >
        <Box p={3}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
            <Typography variant="h6">🎙️</Typography>
            <Typography variant="h6" fontWeight={700}>
              Voice Agent Recordings
            </Typography>
          </Box>
        </Box>
        <DataGrid
          rows={recordings}
          columns={recordingColumns}
          pageSize={10}
          rowsPerPageOptions={[10, 25, 50]}
          autoHeight
          disableSelectionOnClick
          getRowId={(row) => row.session_id}
          sx={{
            "& .MuiDataGrid-cell": {
              borderBottom: "1px solid rgba(224, 224, 224, 0.2)",
              fontSize: "0.875rem",
            },
            "& .MuiDataGrid-columnHeaders": {
              backgroundColor: "rgba(248, 243, 239, 0.8)",
              backdropFilter: "blur(10px)",
              borderBottom: "2px solid rgba(200, 92, 60, 0.15)",
              fontWeight: 600,
              fontSize: "0.875rem",
            },
            "& .MuiDataGrid-footerContainer": {
              borderTop: "2px solid rgba(200, 92, 60, 0.15)",
              backgroundColor: "rgba(248, 243, 239, 0.8)",
              backdropFilter: "blur(10px)",
            },
            "& .MuiDataGrid-row": {
              "&:hover": {
                backgroundColor: "rgba(200, 92, 60, 0.04)",
              },
            },
          }}
        />
      </Paper>
    </Box>
  );
}

export default Sessions;
