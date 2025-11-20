import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Paper,
  Typography,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Snackbar,
  Alert,
  Skeleton,
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Phone as PhoneIcon,
  Person as PersonIcon,
  Email as EmailIcon,
  LocationOn as LocationIcon,
  AccessTime as AccessTimeIcon,
  TrendingUp as TrendingUpIcon,
  Flag as FlagIcon,
  Male as MaleIcon,
  Female as FemaleIcon,
  QuestionMark as QuestionMarkIcon,
} from "@mui/icons-material";
import { useDropzone } from "react-dropzone";
import { leadAPI, voiceAgentAPI } from "../services/api";
import CustomCallDialog from "../components/CustomCallDialog";
import SubscriptionBanner from "../components/SubscriptionBanner";
import { useAuth } from "../contexts/AuthContext";
import { Tooltip } from "@mui/material";
import { getRelativeTime, formatTime } from "../utils/dateUtils";

function Leads() {
  const { hasActiveSubscription } = useAuth();
  const [leads, setLeads] = useState([]);
  const [totalLeads, setTotalLeads] = useState(0);
  const [loading, setLoading] = useState(false);
  const [paginationModel, setPaginationModel] = useState({
    page: 0,
    pageSize: 25,
  });
  const [openDialog, setOpenDialog] = useState(false);
  const [editingLead, setEditingLead] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: "",
    severity: "success",
  });
  const [importDialog, setImportDialog] = useState(false);
  const [callDialog, setCallDialog] = useState(false);
  const [selectedLead, setSelectedLead] = useState(null);

  const [formData, setFormData] = useState({
    first_name: "",
    last_name: "",
    email: "",
    phone: "",
    country: "",
    country_code: "",
    gender: "unknown",
    address: "",
    notes: "",
  });

  useEffect(() => {
    loadLeads();
  }, [paginationModel.page, paginationModel.pageSize]);

  const loadLeads = async () => {
    setLoading(true);
    try {
      const response = await leadAPI.getAll({
        page: paginationModel.page + 1,
        per_page: paginationModel.pageSize,
      });
      setLeads(response.data.leads);
      setTotalLeads(response.data.total);
    } catch (error) {
      console.error("Error loading leads:", error);
      showSnackbar("Error loading leads", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleAddLead = () => {
    setEditingLead(null);
    setFormData({
      first_name: "",
      last_name: "",
      email: "",
      phone: "",
      country: "",
      country_code: "",
      gender: "unknown",
      address: "",
      notes: "",
    });
    setOpenDialog(true);
  };

  const handleEditLead = (lead) => {
    setEditingLead(lead);
    setFormData({
      first_name: lead.first_name || "",
      last_name: lead.last_name || "",
      email: lead.email || "",
      phone: lead.phone,
      country: lead.country || "",
      country_code: lead.country_code || "",
      gender: lead.gender,
      address: lead.address || "",
      notes: lead.notes || "",
    });
    setOpenDialog(true);
  };

  const handleDeleteLead = async (id) => {
    if (window.confirm("Are you sure you want to delete this lead?")) {
      try {
        await leadAPI.delete(id);
        showSnackbar("Lead deleted successfully", "success");
        loadLeads();
      } catch (error) {
        console.error("Error deleting lead:", error);
        showSnackbar("Error deleting lead", "error");
      }
    }
  };

  const handleSaveLead = async () => {
    try {
      if (editingLead) {
        await leadAPI.update(editingLead.id, formData);
        showSnackbar("Lead updated successfully", "success");
      } else {
        await leadAPI.create(formData);
        showSnackbar("Lead created successfully", "success");
      }
      setOpenDialog(false);
      loadLeads();
    } catch (error) {
      console.error("Error saving lead:", error);
      showSnackbar("Error saving lead", "error");
    }
  };

  const handleCallLead = (lead) => {
    setSelectedLead(lead);
    setCallDialog(true);
  };

  const handleMakeCustomCall = async (
    lead,
    callConfig,
    greetingFile,
    greetingTranscript
  ) => {
    try {
      await voiceAgentAPI.makeCall(
        lead.full_phone,
        callConfig,
        greetingFile,
        greetingTranscript
      );
      showSnackbar(
        `Custom call initiated to ${lead.full_name} at ${lead.full_phone}`,
        "success"
      );
    } catch (error) {
      console.error("Error making call:", error);
      showSnackbar("Error making call", "error");
      throw error; // Re-throw to handle in dialog
    }
  };

  const handleFileUpload = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      try {
        const response = await leadAPI.importCSV(file);
        showSnackbar(
          `Imported ${response.data.imported} leads successfully`,
          "success"
        );
        setImportDialog(false);
        loadLeads();
      } catch (error) {
        console.error("Error importing leads:", error);
        showSnackbar("Error importing leads", "error");
      }
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: handleFileUpload,
    accept: {
      "text/csv": [".csv"],
    },
    maxFiles: 1,
  });

  const showSnackbar = (message, severity) => {
    setSnackbar({ open: true, message, severity });
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
      field: "owner_name",
      headerName: "Added By",
      flex: 1,
      minWidth: 110,
      renderCell: (params) => (
        <Chip
          icon={<PersonIcon sx={{ fontSize: 16 }} />}
          label={params.value || "Unknown"}
          size="small"
          sx={{
            fontWeight: 600,
            backdropFilter: "blur(10px)",
            border: "1px solid",
            borderColor: "primary.light",
            "& .MuiChip-icon": {
              color: "primary.main",
            },
          }}
          color="primary"
          variant="outlined"
        />
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
                background: "rgba(200, 92, 60, 0.08)",
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
            background: "rgba(92, 138, 166, 0.08)",
          }}
        >
          <PhoneIcon sx={{ fontSize: 16, color: "info.main" }} />
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
      field: "call_count",
      headerName: "Calls",
      flex: 0.6,
      minWidth: 80,
      align: "center",
      headerAlign: "center",
      renderCell: (params) => (
        <Box
          sx={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            px: 1.5,
            py: 0.5,
            borderRadius: 10,
            background:
              params.value > 0
                ? `rgba(107, 154, 90, ${Math.min(params.value * 0.1, 0.3)})`
                : "rgba(0, 0, 0, 0.05)",
            border: "1px solid",
            borderColor: params.value > 0 ? "success.light" : "divider",
          }}
        >
          <Typography
            variant="body2"
            fontWeight={700}
            sx={{
              color: params.value > 0 ? "success.dark" : "text.secondary",
            }}
          >
            {params.value}
          </Typography>
        </Box>
      ),
    },
    {
      field: "last_called_at",
      headerName: "Last Called",
      flex: 1.3,
      minWidth: 180,
      renderCell: (params) => {
        if (!params.value) {
          return (
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              <AccessTimeIcon sx={{ fontSize: 16, color: "text.disabled" }} />
              <Typography variant="body2" color="text.disabled">
                Never called
              </Typography>
            </Box>
          );
        }

        const { timeAgo, color } = getRelativeTime(params.value);

        return (
          <Box>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              <AccessTimeIcon sx={{ fontSize: 16, color }} />
              <Typography variant="body2" fontWeight={500} color={color}>
                {timeAgo}
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              {formatTime(params.value)}
            </Typography>
          </Box>
        );
      },
    },
    {
      field: "actions",
      headerName: "Actions",
      flex: 1,
      minWidth: 150,
      sortable: false,
      renderCell: (params) => (
        <Box sx={{ display: "flex", gap: 0.5 }}>
          <IconButton
            size="small"
            onClick={() => handleCallLead(params.row)}
            color="primary"
            sx={{
              transition: "all 0.2s ease",
              "&:hover": {
                transform: "scale(1.1)",
                backgroundColor: "rgba(200, 92, 60, 0.1)",
              },
            }}
          >
            <PhoneIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={() => handleEditLead(params.row)}
            sx={{
              transition: "all 0.2s ease",
              "&:hover": {
                transform: "scale(1.1)",
                backgroundColor: "rgba(139, 94, 60, 0.1)",
              },
            }}
          >
            <EditIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={() => handleDeleteLead(params.row.id)}
            color="error"
            sx={{
              transition: "all 0.2s ease",
              "&:hover": {
                transform: "scale(1.1)",
                backgroundColor: "rgba(199, 84, 80, 0.1)",
              },
            }}
          >
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Box>
      ),
    },
  ];

  const handleExportTemplate = () => {
    const csvContent = `first_name,last_name,email,phone,country,prefix,gender,address
John,Doe,john.doe@example.com,555-1234,USA,+1,male,123 Main St
Jane,Smith,jane.smith@example.com,555-5678,UK,+44,female,456 High St`;

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "leads_template.csv";
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <Box className="fade-in" sx={{ height: 'calc(100vh - 100px)', display: 'flex', flexDirection: 'column' }}>
      <SubscriptionBanner />
      <Paper
        className="glass-effect ios-blur-container"
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          width: "100%",
          borderRadius: 3,
          overflow: "hidden",
          position: "relative",
          "& .MuiDataGrid-root": {
            border: "none",
            flex: 1,
          },
          "& .MuiDataGrid-main": {
            overflow: "auto",
          },
          "& .MuiDataGrid-virtualScroller": {
            overflow: "auto !important",
          },
        }}
      >
        <Box
          sx={{
            position: "absolute",
            top: 8,
            right: 16,
            zIndex: 10,
          }}
        >
          <Tooltip
            title={
              !hasActiveSubscription()
                ? "Subscription required to add leads"
                : "Add new lead"
            }
          >
            <span>
              <IconButton
                onClick={handleAddLead}
                disabled={!hasActiveSubscription()}
                sx={{
                  color: "#C85C3C",
                  "&:hover": {
                    backgroundColor: "rgba(200, 92, 60, 0.08)",
                    transform: "scale(1.1) rotate(90deg)",
                  },
                  transition: "all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
                  "&.Mui-disabled": {
                    color: "rgba(200, 92, 60, 0.3)",
                  },
                }}
              >
                <AddIcon />
              </IconButton>
            </span>
          </Tooltip>
        </Box>
        <DataGrid
          rows={leads}
          columns={columns}
          paginationModel={paginationModel}
          onPaginationModelChange={setPaginationModel}
          pageSizeOptions={[25, 50, 100]}
          paginationMode="server"
          rowCount={totalLeads}
          loading={loading}
          disableSelectionOnClick
          disableColumnMenu
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
              backgroundColor: "rgba(248, 243, 239, 0.95)",
              backdropFilter: "blur(20px)",
              borderBottom: "2px solid rgba(200, 92, 60, 0.2)",
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
              borderTop: "2px solid rgba(200, 92, 60, 0.2)",
              backgroundColor: "rgba(248, 243, 239, 0.95)",
              backdropFilter: "blur(20px)",
            },
            "& .MuiDataGrid-row": {
              transition: "all 0.2s ease",
              "&:nth-of-type(even)": {
                backgroundColor: "rgba(248, 243, 239, 0.3)",
              },
              "&:hover": {
                backgroundColor: "rgba(200, 92, 60, 0.06)",
                transform: "translateX(2px)",
                boxShadow: "0 2px 8px rgba(200, 92, 60, 0.1)",
                "& .MuiDataGrid-cell": {
                  borderBottomColor: "transparent",
                },
              },
            },
            "& .MuiDataGrid-virtualScrollerContent": {
              minWidth: "100% !important",
            },
            "& .MuiDataGrid-columnHeadersInner": {
              minWidth: "100% !important",
            },
            "& .MuiDataGrid-cell:focus": {
              outline: "none",
            },
            "& .MuiDataGrid-cell:focus-within": {
              outline: "none",
            },
            "& .MuiDataGrid-columnSeparator": {
              visibility: "hidden",
            },
            "& .MuiDataGrid-menuIcon": {
              visibility: "hidden",
            },
            "& .MuiDataGrid-sortIcon": {
              color: "primary.main",
            },
            "& .MuiCircularProgress-root": {
              color: "primary.main",
            },
          }}
        />
      </Paper>

      {/* Add/Edit Lead Dialog */}
      <Dialog
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>{editingLead ? "Edit Lead" : "Add New Lead"}</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              margin="normal"
              label="First Name"
              value={formData.first_name}
              onChange={(e) =>
                setFormData({ ...formData, first_name: e.target.value })
              }
            />
            <TextField
              fullWidth
              margin="normal"
              label="Last Name"
              value={formData.last_name}
              onChange={(e) =>
                setFormData({ ...formData, last_name: e.target.value })
              }
            />
            <TextField
              fullWidth
              margin="normal"
              label="Email"
              type="email"
              value={formData.email}
              onChange={(e) =>
                setFormData({ ...formData, email: e.target.value })
              }
            />
            <TextField
              fullWidth
              margin="normal"
              label="Phone (without country code)"
              value={formData.phone}
              onChange={(e) =>
                setFormData({ ...formData, phone: e.target.value })
              }
              required
            />
            <TextField
              fullWidth
              margin="normal"
              label="Country"
              value={formData.country}
              onChange={(e) =>
                setFormData({ ...formData, country: e.target.value })
              }
            />
            <TextField
              fullWidth
              margin="normal"
              label="Country Code (e.g., +1)"
              value={formData.country_code}
              onChange={(e) =>
                setFormData({ ...formData, country_code: e.target.value })
              }
            />
            <FormControl fullWidth margin="normal">
              <InputLabel>Gender</InputLabel>
              <Select
                value={formData.gender}
                onChange={(e) =>
                  setFormData({ ...formData, gender: e.target.value })
                }
                label="Gender"
              >
                <MenuItem value="male">Male</MenuItem>
                <MenuItem value="female">Female</MenuItem>
                <MenuItem value="other">Other</MenuItem>
                <MenuItem value="unknown">Unknown</MenuItem>
              </Select>
            </FormControl>
            <TextField
              fullWidth
              margin="normal"
              label="Address"
              multiline
              rows={2}
              value={formData.address}
              onChange={(e) =>
                setFormData({ ...formData, address: e.target.value })
              }
            />
            <TextField
              fullWidth
              margin="normal"
              label="Notes"
              multiline
              rows={2}
              value={formData.notes}
              onChange={(e) =>
                setFormData({ ...formData, notes: e.target.value })
              }
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveLead} variant="contained">
            {editingLead ? "Update" : "Create"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Import Dialog */}
      <Dialog
        open={importDialog}
        onClose={() => setImportDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Import Leads from CSV</DialogTitle>
        <DialogContent>
          <Box
            {...getRootProps()}
            sx={{
              border: "2px dashed #ccc",
              borderRadius: 2,
              p: 4,
              textAlign: "center",
              cursor: "pointer",
              mt: 2,
              "&:hover": {
                borderColor: "primary.main",
              },
            }}
          >
            <input {...getInputProps()} />
            <UploadIcon sx={{ fontSize: 48, color: "text.secondary", mb: 2 }} />
            <Typography variant="h6">
              Drag and drop a CSV file here, or click to select
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              CSV should include: first_name, last_name, email, phone, country,
              prefix, gender, address
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImportDialog(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

      {/* Custom Call Dialog */}
      <CustomCallDialog
        open={callDialog}
        onClose={() => {
          setCallDialog(false);
          setSelectedLead(null);
        }}
        lead={selectedLead}
        onMakeCall={handleMakeCustomCall}
      />

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

export default Leads;
