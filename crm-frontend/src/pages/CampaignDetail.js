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
  Stop as StopIcon,
  Add as AddIcon,
  FilterList as FilterIcon,
  Phone as PhoneIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
} from "@mui/icons-material";
import { DataGrid } from "@mui/x-data-grid";
import { campaignAPI, leadAPI } from "../services/api";

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
  const [availableLeads, setAvailableLeads] = useState([]);
  const [selectedLeadIds, setSelectedLeadIds] = useState([]);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);

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
    try {
      await campaignAPI.start(id);
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

  const handleStopCampaign = async () => {
    if (window.confirm("Are you sure you want to stop this campaign?")) {
      try {
        await campaignAPI.stop(id);
        loadCampaignData();
      } catch (error) {
        console.error("Error stopping campaign:", error);
      }
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
      case "completed":
        return "primary";
      case "cancelled":
        return "error";
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

  const progressPercentage =
    campaign.total_leads > 0
      ? Math.round((campaign.leads_called / campaign.total_leads) * 100)
      : 0;

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={3}>
        <IconButton onClick={() => navigate("/campaigns")} sx={{ mr: 2 }}>
          <ArrowBackIcon />
        </IconButton>
        <Box flexGrow={1}>
          <Typography variant="h4">{campaign.name}</Typography>
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
          <>
            <Button
              variant="outlined"
              startIcon={<PauseIcon />}
              onClick={handlePauseCampaign}
              sx={{ mr: 1 }}
            >
              Pause
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={handleStopCampaign}
            >
              Stop
            </Button>
          </>
        ) : campaign.status === "paused" ? (
          <>
            <Button
              variant="contained"
              startIcon={<PlayIcon />}
              onClick={handleStartCampaign}
              sx={{ mr: 1 }}
            >
              Resume
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={handleStopCampaign}
            >
              Stop
            </Button>
          </>
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

      {/* Progress Bar */}
      {campaign.status === "running" && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body1">Campaign Progress</Typography>
            <Typography variant="body1">{progressPercentage}%</Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={progressPercentage}
            sx={{ height: 8 }}
          />
        </Paper>
      )}

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
                    <TableCell>
                      {new Date(campaign.created_at).toLocaleString()}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Started</TableCell>
                    <TableCell>
                      {campaign.started_at
                        ? new Date(campaign.started_at).toLocaleString()
                        : "Not started"}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Completed</TableCell>
                    <TableCell>
                      {campaign.completed_at
                        ? new Date(campaign.completed_at).toLocaleString()
                        : "Not completed"}
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
              ) : campaign.status === "completed" ? (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Campaign completed successfully
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
                  disabled={
                    campaign.status !== "draft" && campaign.status !== "ready"
                  }
                >
                  Add Selected Leads
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<FilterIcon />}
                  onClick={() => setOpenFilterDialog(true)}
                  sx={{ mb: 1 }}
                  disabled={
                    campaign.status !== "draft" && campaign.status !== "ready"
                  }
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
    </Box>
  );
}

export default CampaignDetail;
