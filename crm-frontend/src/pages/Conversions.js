import React, { useState, useEffect } from "react";
import { Box, Paper, Typography, Chip } from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import {
  Phone as PhoneIcon,
  Email as EmailIcon,
  LocationOn as LocationIcon,
  Male as MaleIcon,
  Female as FemaleIcon,
  QuestionMark as QuestionMarkIcon,
  TrendingUp as TrendingUpIcon,
} from "@mui/icons-material";
import { leadAPI, sessionAPI, voiceAgentAPI } from "../services/api";
import { getRelativeTime, formatTime } from "../utils/dateUtils";

function Conversions() {
  const [interestedLeads, setInterestedLeads] = useState([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);

  useEffect(() => {
    loadInterestedLeads();
  }, [page, pageSize]);

  const loadInterestedLeads = async () => {
    setLoading(true);
    try {
      // Get all leads first
      const leadsResponse = await leadAPI.getAll({ per_page: 1000 });
      const allLeads = leadsResponse.data.leads;

      // Create a map of phone number to lead data for quick lookup
      const leadsMap = new Map();
      allLeads.forEach((lead) => {
        // Store both with and without country code for matching
        leadsMap.set(lead.full_phone, lead);
        leadsMap.set(lead.phone, lead);
      });

      // Get all sessions
      const sessionsResponse = await sessionAPI.getAll({});
      const sessions = sessionsResponse.data;

      // For each session, get the summary to check if interested
      const interestedSessionsMap = new Map();

      for (const session of sessions) {
        if (!session.session_id) continue;

        try {
          const summaryResponse = await voiceAgentAPI.getSummary(
            session.session_id
          );
          const summary = summaryResponse.data.summary;

          if (summary && summary.status === "interested") {
            const phoneKey = session.called_number;

            if (!interestedSessionsMap.has(phoneKey)) {
              interestedSessionsMap.set(phoneKey, {
                interested_count: 1,
                last_interested_at: session.started_at,
                session_id: session.session_id,
                summary: summary,
              });
            } else {
              // Increment interested count if same lead appears multiple times
              const existing = interestedSessionsMap.get(phoneKey);
              existing.interested_count += 1;
              // Update to most recent interested session
              if (
                new Date(session.started_at) >
                new Date(existing.last_interested_at)
              ) {
                existing.last_interested_at = session.started_at;
                existing.session_id = session.session_id;
                existing.summary = summary;
              }
            }
          }
        } catch (error) {
          // Summary not available for this session
          continue;
        }
      }

      // Combine lead data with interested session data
      const interestedLeadsData = [];
      interestedSessionsMap.forEach((sessionData, phoneNumber) => {
        const lead = leadsMap.get(phoneNumber);

        if (lead) {
          interestedLeadsData.push({
            ...lead,
            interested_count: sessionData.interested_count,
            last_interested_at: sessionData.last_interested_at,
            session_id: sessionData.session_id,
            summary: sessionData.summary,
          });
        } else {
          // If lead not found in database, create a minimal entry
          interestedLeadsData.push({
            id: phoneNumber,
            full_name: "Unknown Lead",
            phone: phoneNumber,
            full_phone: phoneNumber,
            country: "—",
            gender: "unknown",
            email: "",
            interested_count: sessionData.interested_count,
            last_interested_at: sessionData.last_interested_at,
            session_id: sessionData.session_id,
            summary: sessionData.summary,
          });
        }
      });

      setInterestedLeads(interestedLeadsData);
    } catch (error) {
      console.error("Error loading interested leads:", error);
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      field: "id",
      headerName: "ID",
      flex: 0.4,
      minWidth: 50,
      renderCell: (params) => (
        <Box
          sx={{
            fontWeight: 600,
            color: "text.secondary",
            fontSize: "0.75rem",
          }}
        >
          #{params.value}
        </Box>
      ),
    },
    {
      field: "full_name",
      headerName: "Name",
      flex: 1.3,
      minWidth: 200,
      renderCell: (params) => {
        const getGenderIcon = () => {
          const gender = params.row.gender;
          if (gender === "male")
            return <MaleIcon sx={{ fontSize: 18, color: "#64B5F6" }} />;
          if (gender === "female")
            return <FemaleIcon sx={{ fontSize: 18, color: "#F48FB1" }} />;
          return (
            <QuestionMarkIcon sx={{ fontSize: 18, color: "text.secondary" }} />
          );
        };

        return (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Box
              sx={{
                p: 0.75,
                borderRadius: "50%",
                background: "rgba(107, 154, 90, 0.12)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {getGenderIcon()}
            </Box>
            <Box>
              <Typography variant="body2" fontWeight={600}>
                {params.value || "—"}
              </Typography>
            </Box>
          </Box>
        );
      },
    },
    {
      field: "email",
      headerName: "Email",
      flex: 1.5,
      minWidth: 220,
      renderCell: (params) => (
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <EmailIcon sx={{ fontSize: 18, color: "text.secondary" }} />
          <Typography
            variant="body2"
            sx={{
              color: params.value ? "text.primary" : "text.disabled",
              fontFamily: "monospace",
              fontSize: "0.875rem",
            }}
          >
            {params.value || "No email"}
          </Typography>
        </Box>
      ),
    },
    {
      field: "full_phone",
      headerName: "Phone",
      flex: 1.1,
      minWidth: 160,
      renderCell: (params) => (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            px: 1.5,
            py: 0.5,
            borderRadius: 2,
            background: "rgba(107, 154, 90, 0.08)",
          }}
        >
          <PhoneIcon sx={{ fontSize: 16, color: "success.main" }} />
          <Typography
            variant="body2"
            fontWeight={500}
            sx={{
              fontFamily: "monospace",
              letterSpacing: "0.5px",
            }}
          >
            {params.value}
          </Typography>
        </Box>
      ),
    },
    {
      field: "country",
      headerName: "Country",
      flex: 0.9,
      minWidth: 120,
      renderCell: (params) => (
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <LocationIcon sx={{ fontSize: 16, color: "text.secondary" }} />
          <Typography variant="body2" fontWeight={500}>
            {params.value || "—"}
          </Typography>
        </Box>
      ),
    },
    {
      field: "gender",
      headerName: "Gender",
      flex: 0.8,
      minWidth: 100,
      renderCell: (params) => (
        <Chip
          label={params.value.charAt(0).toUpperCase() + params.value.slice(1)}
          size="small"
          sx={{
            fontWeight: 600,
            backgroundColor:
              params.value === "male"
                ? "rgba(100, 181, 246, 0.15)"
                : params.value === "female"
                ? "rgba(244, 143, 177, 0.15)"
                : "rgba(0, 0, 0, 0.08)",
            color:
              params.value === "male"
                ? "#1976d2"
                : params.value === "female"
                ? "#d81b60"
                : "text.secondary",
          }}
        />
      ),
    },
    {
      field: "interested_count",
      headerName: "Interest Count",
      flex: 0.8,
      minWidth: 130,
      align: "center",
      headerAlign: "center",
      renderCell: (params) => (
        <Box
          sx={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 0.5,
            px: 1.5,
            py: 0.5,
            borderRadius: 10,
            background: `rgba(107, 154, 90, ${Math.min(
              params.value * 0.15,
              0.4
            )})`,
            border: "1px solid",
            borderColor: "success.light",
          }}
        >
          <TrendingUpIcon sx={{ fontSize: 16, color: "success.dark" }} />
          <Typography
            variant="body2"
            fontWeight={700}
            sx={{
              color: "success.dark",
            }}
          >
            {params.value}
          </Typography>
        </Box>
      ),
    },
    {
      field: "last_interested_at",
      headerName: "Last Interested",
      flex: 1.3,
      minWidth: 180,
      renderCell: (params) => {
        const { timeAgo, color } = getRelativeTime(params.value);

        return (
          <Box>
            <Typography variant="body2" fontWeight={500} color={color}>
              {timeAgo}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatTime(params.value)}
            </Typography>
          </Box>
        );
      },
    },
  ];

  return (
    <Box className="fade-in">
      {/* Header Card */}
      <Paper
        className="glass-effect ios-blur-container"
        sx={{
          p: 3,
          mb: 3,
          background:
            "linear-gradient(135deg, rgba(107, 154, 90, 0.08) 0%, rgba(107, 154, 90, 0.03) 100%)",
          border: "1px solid rgba(107, 154, 90, 0.2)",
          animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Box
            sx={{
              width: 64,
              height: 64,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: 3,
              background:
                "linear-gradient(135deg, rgba(107, 154, 90, 0.2) 0%, rgba(107, 154, 90, 0.1) 100%)",
              boxShadow: "0 4px 16px rgba(107, 154, 90, 0.15)",
            }}
          >
            <img
              src="/transparent-money-bag-1713860414830.webp"
              alt="Conversions"
              style={{ width: 40, height: 40 }}
            />
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h5" fontWeight={700} gutterBottom>
              💰 Interested Leads
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Leads who have shown interest during call sessions. Sort by name,
              country, gender, or interest count.
            </Typography>
          </Box>
          <Box
            sx={{
              textAlign: "center",
              px: 3,
              py: 2,
              borderRadius: 3,
              background: "rgba(107, 154, 90, 0.15)",
            }}
          >
            <Typography variant="h3" fontWeight={800} color="success.dark">
              {interestedLeads.length}
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              fontWeight={600}
            >
              Total Interested
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* Data Grid */}
      <Paper
        className="glass-effect ios-blur-container"
        sx={{
          height: 600,
          width: "100%",
          borderRadius: 3,
          overflow: "hidden",
          position: "relative",
          animation: "springIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
          animationDelay: "0.1s",
          animationFillMode: "both",
        }}
      >
        <DataGrid
          rows={interestedLeads}
          columns={columns}
          pageSize={pageSize}
          rowsPerPageOptions={[25, 50, 100]}
          loading={loading}
          page={page}
          onPageChange={(newPage) => setPage(newPage)}
          onPageSizeChange={(newPageSize) => setPageSize(newPageSize)}
          disableSelectionOnClick
          disableColumnMenu={false}
          rowHeight={72}
          sx={{
            "& .MuiDataGrid-cell": {
              borderBottom: "1px solid rgba(224, 224, 224, 0.15)",
              fontSize: "0.875rem",
              py: 2,
              display: "flex",
              alignItems: "center",
            },
            "& .MuiDataGrid-columnHeaders": {
              backgroundColor: "rgba(239, 246, 237, 0.95)",
              backdropFilter: "blur(20px)",
              borderBottom: "2px solid rgba(107, 154, 90, 0.2)",
              fontWeight: 700,
              fontSize: "0.8125rem",
              letterSpacing: "0.5px",
              textTransform: "uppercase",
              color: "text.secondary",
            },
            "& .MuiDataGrid-columnHeader": {
              "&:focus": {
                outline: "none",
              },
              "&:focus-within": {
                outline: "none",
              },
            },
            "& .MuiDataGrid-footerContainer": {
              borderTop: "2px solid rgba(107, 154, 90, 0.2)",
              backgroundColor: "rgba(239, 246, 237, 0.95)",
              backdropFilter: "blur(20px)",
            },
            "& .MuiDataGrid-row": {
              transition: "all 0.2s ease",
              "&:nth-of-type(even)": {
                backgroundColor: "rgba(239, 246, 237, 0.3)",
              },
              "&:hover": {
                backgroundColor: "rgba(107, 154, 90, 0.08)",
                transform: "translateX(2px)",
                boxShadow: "0 2px 8px rgba(107, 154, 90, 0.15)",
                "& .MuiDataGrid-cell": {
                  borderBottomColor: "transparent",
                },
              },
            },
            "& .MuiDataGrid-cell:focus": {
              outline: "none",
            },
            "& .MuiDataGrid-cell:focus-within": {
              outline: "none",
            },
            "& .MuiDataGrid-columnSeparator": {
              visibility: "visible",
              color: "rgba(107, 154, 90, 0.2)",
            },
            "& .MuiDataGrid-sortIcon": {
              color: "success.main",
            },
            "& .MuiCircularProgress-root": {
              color: "success.main",
            },
          }}
        />
      </Paper>
    </Box>
  );
}

export default Conversions;
