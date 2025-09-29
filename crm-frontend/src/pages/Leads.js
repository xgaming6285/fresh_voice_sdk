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
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Phone as PhoneIcon,
} from "@mui/icons-material";
import { useDropzone } from "react-dropzone";
import { leadAPI, voiceAgentAPI } from "../services/api";

function Leads() {
  const [leads, setLeads] = useState([]);
  const [totalLeads, setTotalLeads] = useState(0);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingLead, setEditingLead] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: "",
    severity: "success",
  });
  const [importDialog, setImportDialog] = useState(false);

  const [formData, setFormData] = useState({
    lead_type: "cold",
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
  }, [page, pageSize]);

  const loadLeads = async () => {
    setLoading(true);
    try {
      const response = await leadAPI.getAll({
        page: page + 1,
        per_page: pageSize,
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
      lead_type: "cold",
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
      lead_type: lead.lead_type,
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

  const handleCallLead = async (lead) => {
    try {
      await voiceAgentAPI.makeCall(lead.full_phone);
      showSnackbar(
        `Calling ${lead.full_name} at ${lead.full_phone}`,
        "success"
      );
    } catch (error) {
      console.error("Error making call:", error);
      showSnackbar("Error making call", "error");
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
    { field: "id", headerName: "ID", width: 70 },
    {
      field: "lead_type",
      headerName: "Type",
      width: 100,
      renderCell: (params) => (
        <Chip
          label={params.value.toUpperCase()}
          size="small"
          color={
            params.value === "ftd"
              ? "success"
              : params.value === "live"
              ? "primary"
              : params.value === "cold"
              ? "default"
              : "secondary"
          }
        />
      ),
    },
    { field: "full_name", headerName: "Name", width: 200 },
    { field: "email", headerName: "Email", width: 200 },
    { field: "full_phone", headerName: "Phone", width: 150 },
    { field: "country", headerName: "Country", width: 120 },
    {
      field: "call_count",
      headerName: "Calls",
      width: 80,
      align: "center",
    },
    {
      field: "last_called_at",
      headerName: "Last Called",
      width: 180,
      valueFormatter: (params) => {
        return params.value ? new Date(params.value).toLocaleString() : "Never";
      },
    },
    {
      field: "actions",
      headerName: "Actions",
      width: 180,
      sortable: false,
      renderCell: (params) => (
        <Box>
          <IconButton
            size="small"
            onClick={() => handleCallLead(params.row)}
            color="primary"
          >
            <PhoneIcon />
          </IconButton>
          <IconButton size="small" onClick={() => handleEditLead(params.row)}>
            <EditIcon />
          </IconButton>
          <IconButton
            size="small"
            onClick={() => handleDeleteLead(params.row.id)}
            color="error"
          >
            <DeleteIcon />
          </IconButton>
        </Box>
      ),
    },
  ];

  const handleExportTemplate = () => {
    const csvContent = `lead_type,first_name,last_name,email,phone,country,prefix,gender,address
cold,John,Doe,john.doe@example.com,555-1234,USA,+1,male,123 Main St
ftd,Jane,Smith,jane.smith@example.com,555-5678,UK,+44,female,456 High St`;

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "leads_template.csv";
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <Box>
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={3}
      >
        <Typography variant="h4">Leads</Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExportTemplate}
            sx={{ mr: 1 }}
          >
            Download Template
          </Button>
          <Button
            variant="outlined"
            startIcon={<UploadIcon />}
            onClick={() => setImportDialog(true)}
            sx={{ mr: 1 }}
          >
            Import CSV
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleAddLead}
          >
            Add Lead
          </Button>
        </Box>
      </Box>

      <Paper sx={{ height: 600, width: "100%" }}>
        <DataGrid
          rows={leads}
          columns={columns}
          pageSize={pageSize}
          rowsPerPageOptions={[25, 50, 100]}
          paginationMode="server"
          rowCount={totalLeads}
          loading={loading}
          page={page}
          onPageChange={(newPage) => setPage(newPage)}
          onPageSizeChange={(newPageSize) => setPageSize(newPageSize)}
          disableSelectionOnClick
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
            <FormControl fullWidth margin="normal">
              <InputLabel>Lead Type</InputLabel>
              <Select
                value={formData.lead_type}
                onChange={(e) =>
                  setFormData({ ...formData, lead_type: e.target.value })
                }
                label="Lead Type"
              >
                <MenuItem value="cold">Cold</MenuItem>
                <MenuItem value="ftd">FTD</MenuItem>
                <MenuItem value="filler">Filler</MenuItem>
                <MenuItem value="live">Live</MenuItem>
              </Select>
            </FormControl>
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
              CSV should include: lead_type, first_name, last_name, email,
              phone, country, prefix, gender, address
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImportDialog(false)}>Cancel</Button>
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

export default Leads;
