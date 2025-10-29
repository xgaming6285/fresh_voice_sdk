import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Box,
  Button,
  Paper,
  Typography,
  Grid,
  Chip,
  Tabs,
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Checkbox,
  FormControlLabel,
  FormGroup,
  LinearProgress,
  Alert,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
} from "@mui/material";
import {
  ArrowBack as ArrowBackIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Edit as EditIcon,
  Add as AddIcon,
  FilterList as FilterIcon,
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
} from "@mui/icons-material";
import { DataGrid } from "@mui/x-data-grid";
import { campaignAPI, leadAPI } from "../services/api";
import { formatDateTime } from "../utils/dateUtils";
import CampaignCallConfigDialog from "../components/CampaignCallConfigDialog";

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`campaign-tabpanel-${index}`}
      aria-labelledby={`campaign-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function CampaignDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [campaign, setCampaign] = useState(null);
  const [campaignLeads, setCampaignLeads] = useState([]);
  const [totalCampaignLeads, setTotalCampaignLeads] = useState(0);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [openAddLeadsDialog, setOpenAddLeadsDialog] = useState(false);
  const [openFilterDialog, setOpenFilterDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [callConfigDialog, setCallConfigDialog] = useState(false);
  const [availableLeads, setAvailableLeads] = useState([]);
  const [selectedLeadIds, setSelectedLeadIds] = useState([]);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);

  const [formData, setFormData] = useState({
    name: "",
    description: "",
    bot_config: {
      greeting_delay: 1,
      max_call_duration: 300,
      voice_speed: 1.0,
    },
    dialing_config: {
      concurrent_calls: 1,
      retry_attempts: 2,
    },
  });

  const [filterCriteria, setFilterCriteria] = useState({
    countries: [],
    lead_types: [],
    never_called: false,
    min_call_count: "",
    max_call_count: "",
  });

  useEffect(() => {
    loadCampaignData();
  }, [id]);

  useEffect(() => {
    if (tabValue === 1) {
      loadCampaignLeads();
    }
  }, [tabValue, page, pageSize]);

  const loadCampaignData = async () => {
    setLoading(true);
    try {
      const response = await campaignAPI.getById(id);
      setCampaign(response.data);
    } catch (error) {
      console.error("Error loading campaign:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadCampaignLeads = async () => {
    try {
      const response = await campaignAPI.getLeads(id, {
        page: page + 1,
        per_page: pageSize,
      });
      setCampaignLeads(response.data.campaign_leads);
      setTotalCampaignLeads(response.data.total);
    } catch (error) {
      console.error("Error loading campaign leads:", error);
    }
  };

  const loadAvailableLeads = async () => {
    try {
      const response = await leadAPI.getAll({ per_page: 1000 });
      setAvailableLeads(response.data.leads);
    } catch (error) {
      console.error("Error loading available leads:", error);
    }
  };

  const handleStartCampaign = async () => {
    // Show call config dialog before starting
    setCallConfigDialog(true);
  };

  const handleStartCampaignWithConfig = async (callConfig) => {
    try {
      await campaignAPI.start(id, callConfig);
      setCallConfigDialog(false);
      loadCampaignData();
    } catch (error) {
      console.error("Error starting campaign:", error);
    }
  };

  const handlePauseCampaign = async () => {
    try {
      await campaignAPI.pause(id);
      loadCampaignData();
    } catch (error) {
      console.error("Error pausing campaign:", error);
    }
  };

  const handleOpenEditDialog = () => {
    setFormData({
      name: campaign.name,
      description: campaign.description || "",
      bot_config: campaign.bot_config,
      dialing_config: campaign.dialing_config,
    });
    setOpenEditDialog(true);
  };

  const handleSaveCampaign = async () => {
    try {
      await campaignAPI.update(id, formData);
      setOpenEditDialog(false);
      loadCampaignData();
    } catch (error) {
      console.error("Error updating campaign:", error);
    }
  };

  const handleAddSelectedLeads = async () => {
    try {
      await campaignAPI.addLeads(id, selectedLeadIds);
      setOpenAddLeadsDialog(false);
      setSelectedLeadIds([]);
      loadCampaignData();
      if (tabValue === 1) {
        loadCampaignLeads();
      }
    } catch (error) {
      console.error("Error adding leads:", error);
    }
  };

  const handleAddFilteredLeads = async () => {
    try {
      const filter = {
        countries:
          filterCriteria.countries.length > 0 ? filterCriteria.countries : null,
        lead_types:
          filterCriteria.lead_types.length > 0
            ? filterCriteria.lead_types
            : null,
        never_called: filterCriteria.never_called,
        min_call_count: filterCriteria.min_call_count
          ? parseInt(filterCriteria.min_call_count)
          : null,
        max_call_count: filterCriteria.max_call_count
          ? parseInt(filterCriteria.max_call_count)
          : null,
      };

      await campaignAPI.addFilteredLeads(id, filter);
      setOpenFilterDialog(false);
      setFilterCriteria({
        countries: [],
        lead_types: [],
        never_called: false,
        min_call_count: "",
        max_call_count: "",
      });
      loadCampaignData();
      if (tabValue === 1) {
        loadCampaignLeads();
      }
    } catch (error) {
      console.error("Error adding filtered leads:", error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "draft":
        return "default";
      case "ready":
        return "info";
      case "running":
        return "success";
      case "paused":
        return "warning";
      default:
        return "default";
    }
  };

  const getCallStatusIcon = (status) => {
    switch (status) {
      case "answered":
        return <CheckCircleIcon color="success" fontSize="small" />;
      case "rejected":
      case "no_answer":
        return <CancelIcon color="error" fontSize="small" />;
      case "in_call":
      case "dialing":
      case "ringing":
        return <PhoneIcon color="primary" fontSize="small" />;
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
        minHeight="400px"
      >
        <LinearProgress sx={{ width: "50%" }} />
      </Box>
    );
  }

  if (!campaign) {
    return (
      <Box>
        <Typography variant="h5">Campaign not found</Typography>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate("/campaigns")}
        >
          Back to Campaigns
        </Button>
      </Box>
    );
  }

  return (
    <Box className="fade-in">
      <Box display="flex" alignItems="center" mb={4}>
        <IconButton
          onClick={() => navigate("/campaigns")}
          sx={{
            mr: 2,
            backgroundColor: "primary.main",
            color: "white",
            "&:hover": {
              backgroundColor: "primary.dark",
              transform: "translateX(-4px)",
            },
            transition: "all 0.3s ease",
          }}
        >
          <ArrowBackIcon />
        </IconButton>
        <Box flexGrow={1}>
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
            {campaign.name}
          </Typography>
          {campaign.description && (
            <Typography variant="body1" color="text.secondary">
              {campaign.description}
            </Typography>
          )}
        </Box>
        <Chip
          label={campaign.status.toUpperCase()}
          color={getStatusColor(campaign.status)}
          sx={{ mr: 2 }}
        />
        <Button
          variant="outlined"
          startIcon={<EditIcon />}
          onClick={handleOpenEditDialog}
          sx={{ mr: 1 }}
        >
          Edit
        </Button>
        {campaign.status === "draft" || campaign.status === "ready" ? (
          <Button
            variant="contained"
            startIcon={<PlayIcon />}
            onClick={handleStartCampaign}
            disabled={campaign.total_leads === 0}
          >
            Start Campaign
          </Button>
        ) : campaign.status === "running" ? (
          <Button
            variant="outlined"
            startIcon={<PauseIcon />}
            onClick={handlePauseCampaign}
          >
            Pause
          </Button>
        ) : campaign.status === "paused" ? (
          <Button
            variant="contained"
            startIcon={<PlayIcon />}
            onClick={handleStartCampaign}
          >
            Resume
          </Button>
        ) : null}
      </Box>

      {/* Campaign Stats */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Total Leads
            </Typography>
            <Typography variant="h4">{campaign.total_leads}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Calls Made
            </Typography>
            <Typography variant="h4">{campaign.leads_called}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Answered
            </Typography>
            <Typography variant="h4" color="success.main">
              {campaign.leads_answered}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Success Rate
            </Typography>
            <Typography variant="h4">
              {campaign.leads_called > 0
                ? `${Math.round(
                    (campaign.leads_answered / campaign.leads_called) * 100
                  )}%`
                : "0%"}
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper>
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
        >
          <Tab label="Overview" />
          <Tab label="Leads" />
          <Tab label="Configuration" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Campaign Information
              </Typography>
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell>Created</TableCell>
                    <TableCell>{formatDateTime(campaign.created_at)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Started</TableCell>
                    <TableCell>
                      {campaign.started_at
                        ? formatDateTime(campaign.started_at)
                        : "Not started"}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Rejected Calls</TableCell>
                    <TableCell>{campaign.leads_rejected}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Failed Calls</TableCell>
                    <TableCell>{campaign.leads_failed}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              {campaign.status === "draft" || campaign.status === "ready" ? (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Add leads to this campaign before starting
                </Alert>
              ) : campaign.status === "running" ? (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Campaign is actively making calls
                </Alert>
              ) : campaign.status === "paused" ? (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Campaign is paused. You can add more leads or resume calling.
                </Alert>
              ) : null}
              <Box>
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={() => {
                    loadAvailableLeads();
                    setOpenAddLeadsDialog(true);
                  }}
                  sx={{ mb: 1, mr: 1 }}
                >
                  Add Selected Leads
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<FilterIcon />}
                  onClick={() => setOpenFilterDialog(true)}
                  sx={{ mb: 1 }}
                >
                  Add Filtered Leads
                </Button>
              </Box>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            mb={2}
          >
            <Typography variant="h6">Campaign Leads</Typography>
            <IconButton onClick={loadCampaignLeads}>
              <RefreshIcon />
            </IconButton>
          </Box>

          <DataGrid
            rows={campaignLeads}
            columns={[
              { field: "lead_name", headerName: "Name", width: 200 },
              { field: "lead_phone", headerName: "Phone", width: 150 },
              { field: "lead_country", headerName: "Country", width: 120 },
              {
                field: "status",
                headerName: "Status",
                width: 130,
                renderCell: (params) => (
                  <Box display="flex" alignItems="center">
                    {getCallStatusIcon(params.value)}
                    <Typography variant="body2" sx={{ ml: 1 }}>
                      {params.value.replace("_", " ")}
                    </Typography>
                  </Box>
                ),
              },
              {
                field: "call_attempts",
                headerName: "Attempts",
                width: 100,
                align: "center",
              },
              {
                field: "last_attempt_at",
                headerName: "Last Attempt",
                width: 180,
                valueFormatter: (params) => {
                  return params.value
                    ? new Date(params.value).toLocaleString()
                    : "-";
                },
              },
              {
                field: "call_duration",
                headerName: "Duration",
                width: 100,
                valueFormatter: (params) => {
                  return params.value ? `${params.value}s` : "-";
                },
              },
            ]}
            pageSize={pageSize}
            rowsPerPageOptions={[25, 50, 100]}
            paginationMode="server"
            rowCount={totalCampaignLeads}
            page={page}
            onPageChange={(newPage) => setPage(newPage)}
            onPageSizeChange={(newPageSize) => setPageSize(newPageSize)}
            disableSelectionOnClick
            autoHeight
          />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            Bot Configuration
          </Typography>
          <pre>{JSON.stringify(campaign.bot_config, null, 2)}</pre>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Dialing Configuration
          </Typography>
          <pre>{JSON.stringify(campaign.dialing_config, null, 2)}</pre>
        </TabPanel>
      </Paper>

      {/* Add Selected Leads Dialog */}
      <Dialog
        open={openAddLeadsDialog}
        onClose={() => setOpenAddLeadsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Add Leads to Campaign</DialogTitle>
        <DialogContent>
          <DataGrid
            rows={availableLeads}
            columns={[
              { field: "full_name", headerName: "Name", width: 200 },
              { field: "full_phone", headerName: "Phone", width: 150 },
              { field: "country", headerName: "Country", width: 120 },
              { field: "lead_type", headerName: "Type", width: 100 },
              { field: "call_count", headerName: "Previous Calls", width: 120 },
            ]}
            checkboxSelection
            onRowSelectionModelChange={(newSelection) => {
              setSelectedLeadIds(newSelection);
            }}
            rowSelectionModel={selectedLeadIds}
            autoHeight
            pageSize={10}
            rowsPerPageOptions={[10]}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenAddLeadsDialog(false)}>Cancel</Button>
          <Button
            onClick={handleAddSelectedLeads}
            variant="contained"
            disabled={selectedLeadIds.length === 0}
          >
            Add {selectedLeadIds.length} Leads
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add Filtered Leads Dialog */}
      <Dialog
        open={openFilterDialog}
        onClose={() => setOpenFilterDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add Filtered Leads</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <FormControl fullWidth margin="normal">
              <InputLabel>Countries</InputLabel>
              <Select
                multiple
                value={filterCriteria.countries}
                onChange={(e) =>
                  setFilterCriteria({
                    ...filterCriteria,
                    countries: e.target.value,
                  })
                }
                label="Countries"
              >
                <MenuItem value="USA">USA</MenuItem>
                <MenuItem value="UK">UK</MenuItem>
                <MenuItem value="Canada">Canada</MenuItem>
                <MenuItem value="Australia">Australia</MenuItem>
                <MenuItem value="Germany">Germany</MenuItem>
                <MenuItem value="France">France</MenuItem>
                <MenuItem value="Spain">Spain</MenuItem>
                <MenuItem value="Italy">Italy</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth margin="normal">
              <InputLabel>Lead Types</InputLabel>
              <Select
                multiple
                value={filterCriteria.lead_types}
                onChange={(e) =>
                  setFilterCriteria({
                    ...filterCriteria,
                    lead_types: e.target.value,
                  })
                }
                label="Lead Types"
              >
                <MenuItem value="cold">Cold</MenuItem>
                <MenuItem value="ftd">FTD</MenuItem>
                <MenuItem value="filler">Filler</MenuItem>
                <MenuItem value="live">Live</MenuItem>
              </Select>
            </FormControl>

            <FormControlLabel
              control={
                <Checkbox
                  checked={filterCriteria.never_called}
                  onChange={(e) =>
                    setFilterCriteria({
                      ...filterCriteria,
                      never_called: e.target.checked,
                    })
                  }
                />
              }
              label="Never Called Before"
              sx={{ mt: 2 }}
            />

            <TextField
              fullWidth
              margin="normal"
              label="Min Call Count"
              type="number"
              value={filterCriteria.min_call_count}
              onChange={(e) =>
                setFilterCriteria({
                  ...filterCriteria,
                  min_call_count: e.target.value,
                })
              }
              inputProps={{ min: 0 }}
            />

            <TextField
              fullWidth
              margin="normal"
              label="Max Call Count"
              type="number"
              value={filterCriteria.max_call_count}
              onChange={(e) =>
                setFilterCriteria({
                  ...filterCriteria,
                  max_call_count: e.target.value,
                })
              }
              inputProps={{ min: 0 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenFilterDialog(false)}>Cancel</Button>
          <Button onClick={handleAddFilteredLeads} variant="contained">
            Add Filtered Leads
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Campaign Dialog */}
      <Dialog
        open={openEditDialog}
        onClose={() => setOpenEditDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Edit Campaign</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              margin="normal"
              label="Campaign Name"
              value={formData.name}
              onChange={(e) =>
                setFormData({ ...formData, name: e.target.value })
              }
              required
            />
            <TextField
              fullWidth
              margin="normal"
              label="Description"
              multiline
              rows={3}
              value={formData.description}
              onChange={(e) =>
                setFormData({ ...formData, description: e.target.value })
              }
            />

            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              Bot Configuration
            </Typography>
            <TextField
              fullWidth
              margin="normal"
              label="Greeting Delay (seconds)"
              type="number"
              value={formData.bot_config.greeting_delay}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  bot_config: {
                    ...formData.bot_config,
                    greeting_delay: parseInt(e.target.value),
                  },
                })
              }
              inputProps={{ min: 0, max: 10 }}
            />
            <TextField
              fullWidth
              margin="normal"
              label="Max Call Duration (seconds)"
              type="number"
              value={formData.bot_config.max_call_duration}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  bot_config: {
                    ...formData.bot_config,
                    max_call_duration: parseInt(e.target.value),
                  },
                })
              }
              inputProps={{ min: 30, max: 3600 }}
            />

            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              Dialing Configuration
            </Typography>
            <TextField
              fullWidth
              margin="normal"
              label="Concurrent Calls"
              type="number"
              value={formData.dialing_config.concurrent_calls}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  dialing_config: {
                    ...formData.dialing_config,
                    concurrent_calls: parseInt(e.target.value),
                  },
                })
              }
              inputProps={{ min: 1, max: 10 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenEditDialog(false)}>Cancel</Button>
          <Button
            onClick={handleSaveCampaign}
            variant="contained"
            disabled={!formData.name}
          >
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Campaign Call Config Dialog */}
      <CampaignCallConfigDialog
        open={callConfigDialog}
        onClose={() => setCallConfigDialog(false)}
        campaign={campaign}
        onStartCampaign={handleStartCampaignWithConfig}
      />
    </Box>
  );
}

export default CampaignDetail;
