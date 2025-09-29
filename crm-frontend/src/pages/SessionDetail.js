import React, { useState, useEffect } from "react";
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
  Divider,
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

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`session-tabpanel-${index}`}
      aria-labelledby={`session-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function SessionDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [recording, setRecording] = useState(null);
  const [transcripts, setTranscripts] = useState({});
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [retranscribing, setRetranscribing] = useState(false);

  useEffect(() => {
    loadSessionData();
  }, [id]);

  const loadSessionData = async () => {
    setLoading(true);
    try {
      // Try to load from CRM database first
      try {
        const crmResponse = await sessionAPI.getById(id);
        setSession(crmResponse.data);
      } catch (error) {
        console.log("Session not in CRM database, checking recordings...");
      }

      // Load recording data
      const recordingsResponse = await voiceAgentAPI.recordings();
      const foundRecording = recordingsResponse.data.recordings.find(
        (r) => r.session_id === id
      );
      if (foundRecording) {
        setRecording(foundRecording);
      }

      // Load transcripts
      try {
        const transcriptsResponse = await voiceAgentAPI.transcripts(id);
        setTranscripts(transcriptsResponse.data.transcripts || {});
      } catch (error) {
        console.log("No transcripts available");
      }
    } catch (error) {
      console.error("Error loading session data:", error);
    } finally {
      setLoading(false);
    }
  };

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

  const handlePlayAudio = (audioType) => {
    const audioFile = recording?.audio_files?.[audioType];
    if (audioFile?.path) {
      // In a real app, you'd implement audio playback here
      // For now, we'll open the file in a new window
      window.open(`http://localhost:8000/static/${audioFile.path}`, "_blank");
    }
  };

  const handleDownloadAudio = (audioType) => {
    const audioFile = recording?.audio_files?.[audioType];
    if (audioFile?.path) {
      // Create download link
      const link = document.createElement("a");
      link.href = `http://localhost:8000/static/${audioFile.path}`;
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
  const duration = sessionData.duration || sessionData.duration_seconds;

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={3}>
        <IconButton onClick={() => navigate("/sessions")} sx={{ mr: 2 }}>
          <ArrowBackIcon />
        </IconButton>
        <Box flexGrow={1}>
          <Typography variant="h4">Session Review</Typography>
          <Typography variant="body1" color="text.secondary">
            Session ID: {id}
          </Typography>
        </Box>
        <IconButton onClick={loadSessionData} sx={{ mr: 1 }}>
          <RefreshIcon />
        </IconButton>
      </Box>

      {/* Session Overview */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <PhoneIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Phone Numbers
                </Typography>
              </Box>
              <Typography variant="body1">
                From: {sessionData.caller_id || "Unknown"}
              </Typography>
              <Typography variant="body1">
                To: {sessionData.called_number || "Unknown"}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <MicIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Call Duration
                </Typography>
              </Box>
              <Typography variant="h5">
                {duration
                  ? `${Math.floor(duration / 60)}:${(duration % 60)
                      .toString()
                      .padStart(2, "0")}`
                  : "N/A"}
              </Typography>
              {sessionData.talk_time && (
                <Typography variant="body2" color="text.secondary">
                  Talk time: {Math.floor(sessionData.talk_time / 60)}:
                  {(sessionData.talk_time % 60).toString().padStart(2, "0")}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <HeadsetIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Call Status
                </Typography>
              </Box>
              <Chip
                label={(sessionData.status || "completed").toUpperCase()}
                color={sessionData.status === "answered" ? "success" : "error"}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <DescriptionIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
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
              />
              {Object.keys(transcripts).length > 0 && (
                <Button
                  size="small"
                  onClick={handleRetranscribe}
                  disabled={retranscribing}
                  sx={{ mt: 1 }}
                >
                  {retranscribing ? "Retranscribing..." : "Retranscribe"}
                </Button>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper>
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
        >
          <Tab label="Audio Recordings" />
          <Tab label="Transcripts" />
          <Tab label="Session Details" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Typography variant="h6" gutterBottom>
            Audio Recordings
          </Typography>

          {recording?.audio_files ? (
            <Grid container spacing={2}>
              {Object.entries(recording.audio_files).map(([type, file]) => (
                <Grid item xs={12} md={4} key={type}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        {type.charAt(0).toUpperCase() +
                          type.slice(1).replace("_", " ")}{" "}
                        Audio
                      </Typography>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        gutterBottom
                      >
                        File: {file.filename}
                      </Typography>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        gutterBottom
                      >
                        Size: {file.size_mb} MB
                      </Typography>
                      <Box mt={2}>
                        <Button
                          startIcon={<PlayIcon />}
                          onClick={() => handlePlayAudio(type)}
                          sx={{ mr: 1 }}
                        >
                          Play
                        </Button>
                        <Button
                          startIcon={<DownloadIcon />}
                          onClick={() => handleDownloadAudio(type)}
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
            <Alert severity="info">
              No audio recordings available for this session
            </Alert>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Typography variant="h6" gutterBottom>
            Call Transcripts
          </Typography>

          {transcriptLoading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <LinearProgress sx={{ width: "50%" }} />
            </Box>
          ) : Object.keys(transcripts).length > 0 ? (
            <Box>
              {Object.entries(transcripts).map(([type, transcript]) => (
                <Box key={type} mb={3}>
                  <Box
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    mb={2}
                  >
                    <Typography variant="subtitle1">
                      {type.charAt(0).toUpperCase() +
                        type.slice(1).replace("_", " ")}{" "}
                      Transcript
                    </Typography>
                    <Box>
                      {transcript.language && (
                        <Chip
                          label={`Language: ${transcript.language}`}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                      )}
                      {transcript.confidence && (
                        <Chip
                          label={`Confidence: ${(
                            transcript.confidence * 100
                          ).toFixed(1)}%`}
                          size="small"
                        />
                      )}
                    </Box>
                  </Box>
                  <Paper
                    sx={{
                      p: 2,
                      backgroundColor: "grey.50",
                      maxHeight: 400,
                      overflow: "auto",
                    }}
                  >
                    {transcript.success === false ? (
                      <Alert severity="error">
                        Transcription failed:{" "}
                        {transcript.error || "Unknown error"}
                      </Alert>
                    ) : (
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
                      {new Date(transcript.transcribed_at).toLocaleString()}
                    </Typography>
                  )}
                </Box>
              ))}
            </Box>
          ) : (
            <Box>
              <Alert severity="info" sx={{ mb: 2 }}>
                No transcripts available for this session.
              </Alert>
              <Button
                variant="contained"
                onClick={handleRetranscribe}
                disabled={retranscribing}
                startIcon={retranscribing ? null : <RefreshIcon />}
              >
                {retranscribing ? "Transcribing..." : "Generate Transcripts"}
              </Button>
            </Box>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            Session Details
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography
                variant="subtitle2"
                color="text.secondary"
                gutterBottom
              >
                Session Information
              </Typography>
              <Box mb={2}>
                <Typography variant="body2">
                  <strong>Session ID:</strong> {id}
                </Typography>
                <Typography variant="body2">
                  <strong>Started:</strong>{" "}
                  {sessionData.start_time
                    ? new Date(sessionData.start_time).toLocaleString()
                    : "Unknown"}
                </Typography>
                <Typography variant="body2">
                  <strong>Ended:</strong>{" "}
                  {sessionData.end_time
                    ? new Date(sessionData.end_time).toLocaleString()
                    : "Unknown"}
                </Typography>
                {sessionData.campaign_id && (
                  <Typography variant="body2">
                    <strong>Campaign:</strong>{" "}
                    <Button
                      size="small"
                      onClick={() =>
                        navigate(`/campaigns/${sessionData.campaign_id}`)
                      }
                    >
                      View Campaign
                    </Button>
                  </Typography>
                )}
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography
                variant="subtitle2"
                color="text.secondary"
                gutterBottom
              >
                AI Analysis
              </Typography>
              {sessionData.sentiment_score !== null &&
              sessionData.sentiment_score !== undefined ? (
                <Box>
                  <Typography variant="body2">
                    <strong>Sentiment Score:</strong>{" "}
                    {sessionData.sentiment_score.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Interest Level:</strong>{" "}
                    {sessionData.interest_level || "N/A"}/10
                  </Typography>
                  <Typography variant="body2">
                    <strong>Follow-up Required:</strong>{" "}
                    {sessionData.follow_up_required ? "Yes" : "No"}
                  </Typography>
                  {sessionData.follow_up_notes && (
                    <Box mt={1}>
                      <Typography variant="body2">
                        <strong>Follow-up Notes:</strong>
                      </Typography>
                      <Typography variant="body2">
                        {sessionData.follow_up_notes}
                      </Typography>
                    </Box>
                  )}
                </Box>
              ) : (
                <Alert severity="info">
                  AI analysis not available for this session
                </Alert>
              )}
            </Grid>
          </Grid>

          {sessionData.key_points && sessionData.key_points.length > 0 && (
            <Box mt={3}>
              <Typography
                variant="subtitle2"
                color="text.secondary"
                gutterBottom
              >
                Key Points from Conversation
              </Typography>
              <Box component="ul">
                {sessionData.key_points.map((point, index) => (
                  <Typography component="li" variant="body2" key={index}>
                    {point}
                  </Typography>
                ))}
              </Box>
            </Box>
          )}
        </TabPanel>
      </Paper>
    </Box>
  );
}

export default SessionDetail;
