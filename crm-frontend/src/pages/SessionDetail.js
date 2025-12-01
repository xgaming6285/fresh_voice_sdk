import React, { useState, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Box,
  Paper,
  Typography,
  Grid,
  IconButton,
  Button,
  Tabs,
  Tab,
  Chip,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
} from "@mui/material";
import {
  ArrowBack as ArrowBackIcon,
  PlayArrow as PlayIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Mic as MicIcon,
  Phone as PhoneIcon,
  Headset as HeadsetIcon,
  Description as DescriptionIcon,
} from "@mui/icons-material";
import { sessionAPI, voiceAgentAPI } from "../services/api";
import { formatDateTime } from "../utils/dateUtils";

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`session-tabpanel-${index}`}
      aria-labelledby={`session-tab-${index}`}
      {...other}
      style={{ height: '100%', overflow: 'auto' }}
    >
      {value === index && children}
    </div>
  );
}

function SessionDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [recording, setRecording] = useState(null);
  const [transcripts, setTranscripts] = useState({});
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [retranscribing, setRetranscribing] = useState(false);
  const [generatingSummary, setGeneratingSummary] = useState(false);
  const [languageDialogOpen, setLanguageDialogOpen] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("English");

  // Helper to get recording URL with token (for local recordings)
  const getRecordingUrl = (audioPath) => {
    return `http://localhost:8000/static/${audioPath}`;
  };

  // Helper to get PBX recording URL using asterisk_linkedid
  const getPBXRecordingUrl = (linkedId) => {
    if (!linkedId) return null;
    return `http://192.168.50.50/play.php?api=R0SHJIU9w55wRR&uniq=${linkedId}`;
  };

  const loadSessionData = useCallback(async () => {
    setLoading(true);
    try {
      // Load session from CRM database - this already includes transcripts, analysis, and audio_files
      const crmResponse = await sessionAPI.getById(id);
      const sessionData = crmResponse.data;
      setSession(sessionData);

      // Check what data is already available in session response
      const hasTranscripts = sessionData.transcripts && Object.keys(sessionData.transcripts).length > 0;
      const hasAnalysis = !!sessionData.analysis;
      const hasAudioFiles = !!sessionData.audio_files;
      const hasPBXRecording = !!sessionData.asterisk_linkedid;

      // Set data that's already available immediately
      if (hasTranscripts) {
        setTranscripts(sessionData.transcripts);
      }
      if (hasAnalysis) {
        setSummary(sessionData.analysis);
      }
      if (hasAudioFiles) {
        setRecording({ session_id: id, audio_files: sessionData.audio_files });
      }

      // Build array of fallback promises for missing data (parallel execution)
      const fallbackPromises = [];

      if (!hasTranscripts) {
        fallbackPromises.push(
          voiceAgentAPI.transcripts(id)
            .then(res => ({ type: 'transcripts', data: res.data.transcripts || {} }))
            .catch(() => ({ type: 'transcripts', data: {} }))
        );
      }

      if (!hasAnalysis) {
        fallbackPromises.push(
          voiceAgentAPI.getSummary(id)
            .then(res => ({ type: 'summary', data: res.data.summary }))
            .catch(() => ({ type: 'summary', data: null }))
        );
      }

      if (!hasAudioFiles && !hasPBXRecording) {
        fallbackPromises.push(
          voiceAgentAPI.recordings()
            .then(res => {
              const foundRecording = res.data.recordings.find(r => r.session_id === id);
              return { type: 'recording', data: foundRecording || null };
            })
            .catch(() => ({ type: 'recording', data: null }))
        );
      }

      // Execute all fallback requests in parallel
      if (fallbackPromises.length > 0) {
        const results = await Promise.all(fallbackPromises);
        results.forEach(result => {
          switch (result.type) {
            case 'transcripts':
              if (Object.keys(result.data).length > 0) setTranscripts(result.data);
              break;
            case 'summary':
              if (result.data) setSummary(result.data);
              break;
            case 'recording':
              if (result.data) setRecording(result.data);
              break;
            default:
              break;
          }
        });
      }
    } catch (error) {
      console.error("Error loading session data:", error);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    loadSessionData();
  }, [loadSessionData]);

  const handleRetranscribe = async () => {
    setRetranscribing(true);
    try {
      await voiceAgentAPI.retranscribe(id);
      // Wait a bit for transcription to start
      setTimeout(() => {
        loadSessionData();
        setRetranscribing(false);
      }, 3000);
    } catch (error) {
      console.error("Error retranscribing:", error);
      setRetranscribing(false);
    }
  };

  const handleOpenLanguageDialog = () => {
    setLanguageDialogOpen(true);
  };

  const handleCloseLanguageDialog = () => {
    setLanguageDialogOpen(false);
  };

  const handleGenerateSummary = async () => {
    setLanguageDialogOpen(false);
    setGeneratingSummary(true);
    try {
      await voiceAgentAPI.generateSummary(id, selectedLanguage);
      // Wait a bit for summary generation to complete
      setTimeout(() => {
        loadSessionData();
        setGeneratingSummary(false);
      }, 5000);
    } catch (error) {
      console.error("Error generating summary:", error);
      setGeneratingSummary(false);
    }
  };

  const handlePlayAudio = (audioType) => {
    // Check if we have PBX recording (asterisk_linkedid)
    const pbxUrl = getPBXRecordingUrl(session?.asterisk_linkedid);
    if (pbxUrl) {
      window.open(pbxUrl, "_blank");
      return;
    }

    // Fallback to local recordings
    const audioFile = recording?.audio_files?.[audioType];
    if (audioFile?.path) {
      window.open(getRecordingUrl(audioFile.path), "_blank");
    }
  };

  const handleDownloadAudio = (audioType) => {
    // Check if we have PBX recording (asterisk_linkedid)
    const pbxUrl = getPBXRecordingUrl(session?.asterisk_linkedid);
    if (pbxUrl) {
      const link = document.createElement("a");
      link.href = pbxUrl;
      link.download = `recording_${session.asterisk_linkedid}.mp3`;
      link.click();
      return;
    }

    // Fallback to local recordings
    const audioFile = recording?.audio_files?.[audioType];
    if (audioFile?.path) {
      const link = document.createElement("a");
      link.href = getRecordingUrl(audioFile.path);
      link.download = audioFile.filename;
      link.click();
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

  if (!session && !recording) {
    return (
      <Box>
        <Typography variant="h5">Session not found</Typography>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate("/sessions")}
        >
          Back to Sessions
        </Button>
      </Box>
    );
  }

  const sessionData = session || recording;
  // Try to get duration from multiple sources
  let duration = sessionData.duration || sessionData.duration_seconds;
  // If not found, check session_info
  if (!duration && sessionData.session_info) {
    duration = sessionData.session_info.duration_seconds;
  }

  return (
    <Box sx={{ height: 'calc(100vh - 80px)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <Box 
        display="flex" 
        alignItems="center" 
        mb={2}
        sx={{
          p: 2,
          background: "rgba(255, 255, 255, 0.72)",
          backdropFilter: "blur(20px)",
          borderRadius: 2,
          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.05)",
        }}
      >
        <IconButton 
          onClick={() => navigate("/sessions")} 
          sx={{ 
            mr: 2,
            background: "rgba(200, 92, 60, 0.08)",
            "&:hover": {
              background: "rgba(200, 92, 60, 0.15)",
            }
          }}
        >
          <ArrowBackIcon />
        </IconButton>
        <Box flexGrow={1}>
          <Typography variant="h5" fontWeight={700} color="primary.main">
            Session Review
          </Typography>
          <Typography variant="body2" color="text.secondary" fontWeight={500}>
            Session ID: {id}
          </Typography>
        </Box>
        <IconButton 
          onClick={loadSessionData} 
          sx={{
            background: "rgba(92, 138, 166, 0.08)",
            "&:hover": {
              background: "rgba(92, 138, 166, 0.15)",
            }
          }}
        >
          <RefreshIcon />
        </IconButton>
      </Box>

      {/* Session Overview Cards */}
      <Grid container spacing={2} mb={2}>
        <Grid item xs={12} sm={6} md={3}>
          <Card 
            className="glass-effect"
            sx={{ 
              height: "100%",
              transition: "all 0.3s ease",
              "&:hover": {
                transform: "translateY(-4px)",
                boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
              }
            }}
          >
            <CardContent sx={{ p: 2 }}>
              <Box display="flex" alignItems="center" mb={1.5} gap={1}>
                <Box
                  sx={{
                    p: 1,
                    borderRadius: "50%",
                    background: "rgba(200, 92, 60, 0.1)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <PhoneIcon sx={{ fontSize: 20, color: "primary.main" }} />
                </Box>
                <Typography variant="subtitle2" color="text.secondary" fontWeight={700}>
                  Phone Numbers
                </Typography>
              </Box>
              <Typography variant="body2" fontWeight={600}>
                From: <strong>{sessionData.caller_id || "Unknown"}</strong>
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                To: <strong>{sessionData.called_number || "Unknown"}</strong>
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card 
            className="glass-effect"
            sx={{ 
              height: "100%",
              transition: "all 0.3s ease",
              "&:hover": {
                transform: "translateY(-4px)",
                boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
              }
            }}
          >
            <CardContent sx={{ p: 2 }}>
              <Box display="flex" alignItems="center" mb={1.5} gap={1}>
                <Box
                  sx={{
                    p: 1,
                    borderRadius: "50%",
                    background: "rgba(92, 138, 166, 0.1)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <MicIcon sx={{ fontSize: 20, color: "info.main" }} />
                </Box>
                <Typography variant="subtitle2" color="text.secondary" fontWeight={700}>
                  Call Duration
                </Typography>
              </Box>
              <Typography variant="h5" fontWeight={700}>
                {duration
                  ? `${Math.floor(duration / 60)}:${Math.floor(duration % 60)
                      .toString()
                      .padStart(2, "0")}`
                  : "N/A"}
              </Typography>
              {sessionData.talk_time && (
                <Typography variant="body2" color="text.secondary" fontWeight={500}>
                  Talk time: {Math.floor(sessionData.talk_time / 60)}:
                  {Math.floor(sessionData.talk_time % 60)
                    .toString()
                    .padStart(2, "0")}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card 
            className="glass-effect"
            sx={{ 
              height: "100%",
              transition: "all 0.3s ease",
              "&:hover": {
                transform: "translateY(-4px)",
                boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
              }
            }}
          >
            <CardContent sx={{ p: 2 }}>
              <Box display="flex" alignItems="center" mb={1.5} gap={1}>
                <Box
                  sx={{
                    p: 1,
                    borderRadius: "50%",
                    background: "rgba(107, 154, 90, 0.1)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <HeadsetIcon sx={{ fontSize: 20, color: "success.main" }} />
                </Box>
                <Typography variant="subtitle2" color="text.secondary" fontWeight={700}>
                  Call Status
                </Typography>
              </Box>
              <Chip
                label={(sessionData.status || "completed").toUpperCase()}
                color={sessionData.status === "answered" ? "success" : "error"}
                sx={{ fontWeight: 700, fontSize: "0.8rem" }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card 
            className="glass-effect"
            sx={{ 
              height: "100%",
              transition: "all 0.3s ease",
              "&:hover": {
                transform: "translateY(-4px)",
                boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
              }
            }}
          >
            <CardContent sx={{ p: 2 }}>
              <Box display="flex" alignItems="center" mb={1.5} gap={1}>
                <Box
                  sx={{
                    p: 1,
                    borderRadius: "50%",
                    background: "rgba(139, 94, 60, 0.1)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <DescriptionIcon sx={{ fontSize: 20, color: "warning.main" }} />
                </Box>
                <Typography variant="subtitle2" color="text.secondary" fontWeight={700}>
                  Transcripts
                </Typography>
              </Box>
              <Chip
                label={
                  Object.keys(transcripts).length > 0
                    ? "Available"
                    : "Not Available"
                }
                color={
                  Object.keys(transcripts).length > 0 ? "success" : "default"
                }
                size="small"
                sx={{ fontWeight: 700, fontSize: "0.75rem" }}
              />
              {Object.keys(transcripts).length > 0 && (
                <Button
                  size="small"
                  onClick={handleRetranscribe}
                  disabled={retranscribing}
                  sx={{ 
                    mt: 1,
                    fontSize: "0.75rem",
                    textTransform: "none",
                    fontWeight: 600,
                  }}
                >
                  {retranscribing ? "Retranscribing..." : "Retranscribe"}
                </Button>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper 
        className="glass-effect"
        sx={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column',
          overflow: 'hidden',
          borderRadius: 2,
        }}
      >
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
          sx={{
            borderBottom: "2px solid rgba(200, 92, 60, 0.1)",
            "& .MuiTab-root": {
              fontWeight: 600,
              textTransform: "none",
              fontSize: "0.95rem",
            },
            "& .Mui-selected": {
              color: "primary.main",
            },
          }}
        >
          <Tab label="Audio Recordings" />
          <Tab label="Transcripts" />
          <Tab label="Session Details" />
        </Tabs>

        <Box sx={{ flex: 1, overflow: 'auto', p: 3 }}>
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h6" gutterBottom fontWeight={700} color="primary.main">
              Audio Recordings
            </Typography>

            {session?.asterisk_linkedid ? (
              // PBX Recording Display (Single MP3 stream)
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Card className="glass-effect" sx={{ borderRadius: 2 }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                        <PhoneIcon color="primary" />
                        <Typography variant="h6" fontWeight={700}>
                          Call Recording (PBX)
                        </Typography>
                      </Box>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        fontWeight={500}
                        gutterBottom
                        sx={{ mb: 2 }}
                      >
                        Linked ID: {session.asterisk_linkedid}
                      </Typography>
                      <Box 
                        sx={{ 
                          p: 2, 
                          borderRadius: 2, 
                          background: "rgba(248, 243, 239, 0.5)",
                          mb: 2,
                        }}
                      >
                        <audio
                          controls
                          style={{ width: "100%" }}
                          src={getPBXRecordingUrl(session.asterisk_linkedid)}
                        >
                          Your browser does not support the audio element.
                        </audio>
                      </Box>
                      <Box display="flex" gap={1}>
                        <Button
                          variant="outlined"
                          startIcon={<PlayIcon />}
                          onClick={() => handlePlayAudio(null)}
                          sx={{ 
                            borderRadius: 2,
                            textTransform: "none",
                            fontWeight: 600,
                          }}
                        >
                          Open in New Tab
                        </Button>
                        <Button
                          variant="contained"
                          startIcon={<DownloadIcon />}
                          onClick={() => handleDownloadAudio(null)}
                          sx={{ 
                            borderRadius: 2,
                            textTransform: "none",
                            fontWeight: 600,
                          }}
                        >
                          Download
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            ) : recording?.audio_files ? (
              // Local Recording Display (3 WAV files)
              <Grid container spacing={2}>
                {Object.entries(recording.audio_files).map(([type, file]) => (
                  <Grid item xs={12} md={4} key={type}>
                    <Card className="glass-effect" sx={{ height: "100%", borderRadius: 2 }}>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="h6" fontWeight={700} gutterBottom>
                          {type.charAt(0).toUpperCase() +
                            type.slice(1).replace("_", " ")}{" "}
                          Audio
                        </Typography>
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          fontWeight={500}
                          gutterBottom
                        >
                          File: {file.filename}
                        </Typography>
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          fontWeight={500}
                          gutterBottom
                        >
                          Size: {file.size_mb} MB
                        </Typography>
                        <Box mt={2} display="flex" flexDirection="column" gap={1}>
                          <Button
                            variant="outlined"
                            startIcon={<PlayIcon />}
                            onClick={() => handlePlayAudio(type)}
                            fullWidth
                            sx={{ 
                              borderRadius: 2,
                              textTransform: "none",
                              fontWeight: 600,
                            }}
                          >
                            Play
                          </Button>
                          <Button
                            variant="contained"
                            startIcon={<DownloadIcon />}
                            onClick={() => handleDownloadAudio(type)}
                            fullWidth
                            sx={{ 
                              borderRadius: 2,
                              textTransform: "none",
                              fontWeight: 600,
                            }}
                          >
                            Download
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Alert severity="info" sx={{ borderRadius: 2 }}>
                No audio recordings available for this session
              </Alert>
            )}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom fontWeight={700} color="primary.main">
              Call Transcripts
            </Typography>

            {Object.keys(transcripts).length > 0 ? (
              <Box>
                {Object.entries(transcripts).map(([type, transcript]) => (
                  <Box key={type} mb={3}>
                    <Box
                      display="flex"
                      justifyContent="space-between"
                      alignItems="center"
                      mb={2}
                    >
                      <Typography variant="h6" fontWeight={700}>
                        {type.charAt(0).toUpperCase() +
                          type.slice(1).replace("_", " ")}{" "}
                        Transcript
                      </Typography>
                      <Box display="flex" gap={1}>
                        {transcript.language && (
                          <Chip
                            label={`Language: ${transcript.language}`}
                            size="small"
                            sx={{ fontWeight: 600 }}
                          />
                        )}
                        {transcript.confidence && (
                          <Chip
                            label={`Confidence: ${(
                              transcript.confidence * 100
                            ).toFixed(1)}%`}
                            size="small"
                            color="success"
                            sx={{ fontWeight: 600 }}
                          />
                        )}
                      </Box>
                    </Box>
                    <Paper
                      className="glass-effect"
                      sx={{
                        p: 2.5,
                        backgroundColor: "rgba(248, 243, 239, 0.4)",
                        maxHeight: 400,
                        overflow: "auto",
                        borderRadius: 2,
                      }}
                    >
                    {transcript.success === false ? (
                      <Alert severity="error">
                        Transcription failed:{" "}
                        {transcript.error || "Unknown error"}
                      </Alert>
                    ) : transcript.conversation &&
                      Array.isArray(transcript.conversation) ? (
                      // Display as conversation if structured data is available
                      <Box>
                        {transcript.conversation.map((turn, index) => (
                          <Box
                            key={index}
                            sx={{
                              mb: 1.5,
                              display: "flex",
                              flexDirection:
                                turn.speaker === "agent"
                                  ? "row"
                                  : "row-reverse",
                            }}
                          >
                            <Box
                              sx={{
                                maxWidth: "75%",
                                p: 1.5,
                                borderRadius: 2,
                                backgroundColor:
                                  turn.speaker === "agent"
                                    ? "primary.light"
                                    : "secondary.light",
                                color:
                                  turn.speaker === "agent"
                                    ? "primary.contrastText"
                                    : "secondary.contrastText",
                              }}
                            >
                              <Typography
                                variant="caption"
                                sx={{
                                  display: "block",
                                  mb: 0.5,
                                  fontWeight: "bold",
                                  opacity: 0.8,
                                }}
                              >
                                {turn.speaker === "agent" ? "AGENT" : "USER"}
                              </Typography>
                              <Typography
                                variant="body2"
                                style={{ whiteSpace: "pre-wrap" }}
                              >
                                {turn.text}
                              </Typography>
                            </Box>
                          </Box>
                        ))}
                      </Box>
                    ) : (
                      // Fallback to plain text if no structured data
                      <Typography
                        variant="body2"
                        style={{ whiteSpace: "pre-wrap" }}
                      >
                        {transcript.content ||
                          "No transcript content available"}
                      </Typography>
                    )}
                  </Paper>
                  {transcript.transcribed_at && (
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ mt: 1, display: "block" }}
                    >
                      Transcribed at:{" "}
                      {formatDateTime(transcript.transcribed_at)}
                    </Typography>
                  )}
                </Box>
              ))}
            </Box>
            ) : (
              <Box>
                <Alert severity="info" sx={{ mb: 2, borderRadius: 2 }}>
                  No transcripts available for this session.
                </Alert>
                <Button
                  variant="contained"
                  onClick={handleRetranscribe}
                  disabled={retranscribing}
                  startIcon={retranscribing ? null : <RefreshIcon />}
                  sx={{
                    borderRadius: 2,
                    textTransform: "none",
                    fontWeight: 700,
                    px: 3,
                    py: 1,
                  }}
                >
                  {retranscribing ? "Transcribing..." : "Generate Transcripts"}
                </Button>
              </Box>
            )}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <Typography variant="h6" gutterBottom fontWeight={700} color="primary.main">
              Session Details
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card className="glass-effect" sx={{ height: "100%", borderRadius: 2 }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography
                      variant="h6"
                      fontWeight={700}
                      gutterBottom
                      sx={{ mb: 2 }}
                    >
                      Session Information
                    </Typography>
                    <Box display="flex" flexDirection="column" gap={1.5}>
                      <Box>
                        <Typography variant="body2" color="text.secondary" fontWeight={600}>
                          Session ID
                        </Typography>
                        <Typography variant="body1" fontWeight={500}>
                          {id}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="body2" color="text.secondary" fontWeight={600}>
                          Started
                        </Typography>
                        <Typography variant="body1" fontWeight={500}>
                          {sessionData.start_time
                            ? formatDateTime(sessionData.start_time)
                            : "Unknown"}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="body2" color="text.secondary" fontWeight={600}>
                          Ended
                        </Typography>
                        <Typography variant="body1" fontWeight={500}>
                          {sessionData.end_time
                            ? formatDateTime(sessionData.end_time)
                            : "Unknown"}
                        </Typography>
                      </Box>
                      {sessionData.campaign_id && (
                        <Box>
                          <Typography variant="body2" color="text.secondary" fontWeight={600}>
                            Campaign
                          </Typography>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={() =>
                              navigate(`/campaigns/${sessionData.campaign_id}`)
                            }
                            sx={{
                              mt: 0.5,
                              borderRadius: 2,
                              textTransform: "none",
                              fontWeight: 600,
                            }}
                          >
                            View Campaign
                          </Button>
                        </Box>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card className="glass-effect" sx={{ height: "100%", borderRadius: 2 }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography
                      variant="h6"
                      fontWeight={700}
                      gutterBottom
                      sx={{ mb: 2 }}
                    >
                      AI Summary
                    </Typography>
                    {summary ? (
                      <Box>
                        <Chip
                          label={summary.status.toUpperCase()}
                          size="medium"
                          color={
                            summary.status === "interested" ? "success" : "default"
                          }
                          sx={{ mb: 2, fontWeight: 700 }}
                        />
                        <Typography variant="body2" fontWeight={600} gutterBottom>
                          Summary:
                        </Typography>
                        <Paper
                          sx={{
                            p: 2,
                            mt: 1,
                            backgroundColor: "rgba(248, 243, 239, 0.4)",
                            maxHeight: 200,
                            overflow: "auto",
                            borderRadius: 1.5,
                          }}
                        >
                          <Typography variant="body2" fontWeight={500}>
                            {summary.summary}
                          </Typography>
                        </Paper>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={handleOpenLanguageDialog}
                          disabled={generatingSummary}
                          sx={{
                            mt: 2,
                            borderRadius: 2,
                            textTransform: "none",
                            fontWeight: 600,
                          }}
                        >
                          {generatingSummary
                            ? "Regenerating..."
                            : "Regenerate Summary"}
                        </Button>
                      </Box>
                    ) : (
                      <Box>
                        <Alert severity="info" sx={{ mb: 2, borderRadius: 2 }}>
                          No AI summary available for this session
                        </Alert>
                        <Button
                          variant="contained"
                          size="small"
                          onClick={handleOpenLanguageDialog}
                          disabled={
                            generatingSummary || Object.keys(transcripts).length === 0
                          }
                          sx={{
                            borderRadius: 2,
                            textTransform: "none",
                            fontWeight: 700,
                            px: 3,
                          }}
                        >
                          {generatingSummary ? "Generating..." : "Generate Summary"}
                        </Button>
                        {Object.keys(transcripts).length === 0 && (
                          <Typography
                            variant="caption"
                            display="block"
                            sx={{ mt: 1 }}
                            color="text.secondary"
                            fontWeight={500}
                          >
                            Transcripts are required to generate a summary
                          </Typography>
                        )}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {sessionData.key_points && sessionData.key_points.length > 0 && (
              <Box mt={3}>
                <Card className="glass-effect" sx={{ borderRadius: 2 }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography
                      variant="h6"
                      fontWeight={700}
                      gutterBottom
                      sx={{ mb: 2 }}
                    >
                      Key Points from Conversation
                    </Typography>
                    <Box component="ul" sx={{ pl: 2 }}>
                      {sessionData.key_points.map((point, index) => (
                        <Typography
                          component="li"
                          variant="body2"
                          fontWeight={500}
                          key={index}
                          sx={{ mb: 1 }}
                        >
                          {point}
                        </Typography>
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Box>
            )}
          </TabPanel>
        </Box>
      </Paper>

      {/* Language Selection Dialog */}
      <Dialog
        open={languageDialogOpen}
        onClose={handleCloseLanguageDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Select Summary Language</DialogTitle>
        <DialogContent>
          <TextField
            select
            label="Language"
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            fullWidth
            margin="normal"
            helperText="Choose the language for the AI-generated summary"
          >
            <MenuItem value="English">English</MenuItem>
            <MenuItem value="Bulgarian">Bulgarian (Български)</MenuItem>
            <MenuItem value="Spanish">Spanish (Español)</MenuItem>
            <MenuItem value="French">French (Français)</MenuItem>
            <MenuItem value="German">German (Deutsch)</MenuItem>
            <MenuItem value="Italian">Italian (Italiano)</MenuItem>
            <MenuItem value="Portuguese">Portuguese (Português)</MenuItem>
            <MenuItem value="Russian">Russian (Русский)</MenuItem>
            <MenuItem value="Chinese">Chinese (中文)</MenuItem>
            <MenuItem value="Japanese">Japanese (日本語)</MenuItem>
            <MenuItem value="Korean">Korean (한국어)</MenuItem>
            <MenuItem value="Arabic">Arabic (العربية)</MenuItem>
            <MenuItem value="Hindi">Hindi (हिन्दी)</MenuItem>
            <MenuItem value="Turkish">Turkish (Türkçe)</MenuItem>
            <MenuItem value="Polish">Polish (Polski)</MenuItem>
            <MenuItem value="Dutch">Dutch (Nederlands)</MenuItem>
            <MenuItem value="Swedish">Swedish (Svenska)</MenuItem>
            <MenuItem value="Norwegian">Norwegian (Norsk)</MenuItem>
            <MenuItem value="Danish">Danish (Dansk)</MenuItem>
            <MenuItem value="Finnish">Finnish (Suomi)</MenuItem>
          </TextField>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseLanguageDialog}>Cancel</Button>
          <Button
            onClick={handleGenerateSummary}
            variant="contained"
            disabled={generatingSummary}
          >
            Generate
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default SessionDetail;
