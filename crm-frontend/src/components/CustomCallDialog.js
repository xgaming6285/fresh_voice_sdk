import React, { useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Typography,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  CircularProgress,
} from "@mui/material";
import {
  ExpandMore as ExpandMoreIcon,
  Business as BusinessIcon,
  Person as PersonIcon,
  ShoppingCart as ProductIcon,
  Settings as SettingsIcon,
  Phone as PhoneIcon,
  Language as LanguageIcon,
} from "@mui/icons-material";

const CustomCallDialog = ({ open, onClose, lead, onMakeCall }) => {
  const [callConfig, setCallConfig] = useState({
    company_name: "QuantumAI",
    caller_name: "John",
    product_name: "ArtroFlex joint pain relief cream",
    additional_prompt: "",
    call_urgency: "medium", // low, medium, high
    call_objective: "sales", // sales, followup, survey, appointment
    main_benefits: "natural ingredients, fast pain relief, no side effects",
    special_offer: "50% off today only, free shipping, 30-day guarantee",
    objection_strategy: "understanding", // understanding, aggressive, educational
    voice_name: "Puck", // Gemini voice selection
    greeting_instruction: "", // Custom greeting text
  });

  const [loading, setLoading] = useState(false);
  const [greetingStatus, setGreetingStatus] = useState(""); // "generating", "ready", ""

  const handleMakeCall = async () => {
    setLoading(true);
    setGreetingStatus("generating");

    try {
      // Step 1: Generate custom greeting
      console.log("Generating custom greeting...");
      const greetingResponse = await fetch(
        "http://localhost:8000/api/generate_greeting",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            phone_number: lead.full_phone,
            call_config: callConfig,
          }),
        }
      );

      if (!greetingResponse.ok) {
        const error = await greetingResponse.json();
        console.error("Failed to generate greeting:", error);

        // Check if it's because greeting generator is not available
        if (greetingResponse.status === 503) {
          // Proceed without custom greeting
          console.log(
            "Greeting generator not available, using default greeting"
          );
          await onMakeCall(lead, callConfig);
        } else {
          throw new Error(error.detail || "Failed to generate greeting");
        }
      } else {
        // Greeting generated successfully
        const greetingData = await greetingResponse.json();
        console.log("Greeting generated:", greetingData);
        setGreetingStatus("ready");

        // Step 2: Make call with custom greeting
        await onMakeCall(lead, callConfig, greetingData.greeting_file);
      }

      onClose();
      // Reset form
      setCallConfig({
        company_name: "QuantumAI",
        caller_name: "John",
        product_name: "ArtroFlex joint pain relief cream",
        additional_prompt: "",
        call_urgency: "medium",
        call_objective: "sales",
        main_benefits: "natural ingredients, fast pain relief, no side effects",
        special_offer: "50% off today only, free shipping, 30-day guarantee",
        objection_strategy: "understanding",
        voice_name: "Puck",
        greeting_instruction: "",
      });
    } catch (error) {
      console.error("Error making call:", error);
      alert("Failed to make call: " + error.message);
    } finally {
      setLoading(false);
      setGreetingStatus("");
    }
  };

  const getObjectiveIcon = (objective) => {
    switch (objective) {
      case "sales":
        return <ProductIcon />;
      case "followup":
        return <PhoneIcon />;
      case "survey":
        return <SettingsIcon />;
      case "appointment":
        return <PersonIcon />;
      default:
        return <PhoneIcon />;
    }
  };

  // Detect country/language from lead phone number for preview
  const getDetectedLanguage = (phone) => {
    if (!phone) return "Unknown";

    // Simple phone prefix detection for preview
    if (phone.startsWith("+359") || phone.includes("359"))
      return "Bulgarian 🇧🇬";
    if (phone.startsWith("+40") && !phone.startsWith("+401"))
      return "Romanian 🇷🇴";
    if (phone.startsWith("+30")) return "Greek 🇬🇷";
    if (phone.startsWith("+49")) return "German 🇩🇪";
    if (phone.startsWith("+33")) return "French 🇫🇷";
    if (phone.startsWith("+34")) return "Spanish 🇪🇸";
    if (phone.startsWith("+39")) return "Italian 🇮🇹";
    if (phone.startsWith("+1")) return "English 🇺🇸";
    if (phone.startsWith("+44")) return "English 🇬🇧";

    return "Auto-detect 🌍";
  };

  const generatePromptPreview = () => {
    const {
      company_name,
      caller_name,
      product_name,
      call_objective,
      call_urgency,
      main_benefits,
      special_offer,
      objection_strategy,
    } = callConfig;

    let basePrompt = `You are ${caller_name} from ${company_name}, a highly persuasive sales representative for ${product_name}. `;

    if (call_objective === "sales") {
      basePrompt +=
        "You are making sales calls to sell this product. Focus on converting prospects into customers by highlighting product benefits and closing the sale. ";
    } else if (call_objective === "followup") {
      basePrompt +=
        "You are following up on a previous interaction. Be friendly and check on their interest while guiding toward a purchase decision. ";
    } else if (call_objective === "survey") {
      basePrompt +=
        "You are conducting a survey but also identifying sales opportunities. Ask relevant questions while presenting the product benefits. ";
    } else if (call_objective === "appointment") {
      basePrompt +=
        "You are cold calling to set appointments or qualify leads. Focus on building rapport, understanding their needs, and scheduling a follow-up meeting or call. ";
    }

    if (main_benefits) {
      basePrompt += `Key benefits to emphasize: ${main_benefits}. `;
    }

    if (special_offer) {
      basePrompt += `Current offers: ${special_offer}. `;
    }

    if (call_urgency === "high") {
      basePrompt +=
        "Create MAXIMUM URGENCY - emphasize time-sensitive offers and limited availability. ";
    } else if (call_urgency === "medium") {
      basePrompt +=
        "Create moderate urgency with special offers and time-sensitive deals. ";
    } else {
      basePrompt +=
        "Be persistent but not overly aggressive. Focus on building rapport and trust. ";
    }

    if (objection_strategy === "understanding") {
      basePrompt +=
        "Handle objections with empathy and understanding. Listen to their concerns and address them thoughtfully. ";
    } else if (objection_strategy === "educational") {
      basePrompt +=
        "Handle objections by providing educational information and facts to overcome doubts. ";
    } else if (objection_strategy === "aggressive") {
      basePrompt +=
        "Handle objections persistently. Push back on concerns and maintain strong sales pressure. ";
    }

    basePrompt +=
      "Always try to close the sale and handle objections professionally.";

    return basePrompt;
  };

  if (!lead) return null;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 3,
          overflow: "hidden",
        },
      }}
    >
      <DialogTitle
        sx={{
          background: "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
          color: "white",
          display: "flex",
          alignItems: "center",
          gap: 1,
        }}
      >
        <PhoneIcon />
        <Box>
          <Typography variant="h6" component="div">
            Custom Call Setup
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.9 }}>
            Calling {lead.full_name} at {lead.full_phone}
          </Typography>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 3 }}>
        {/* Language Detection Info */}
        <Alert
          severity="info"
          icon={<LanguageIcon />}
          sx={{ mb: 3, borderRadius: 2 }}
        >
          <Box>
            <Typography variant="body2" fontWeight={600}>
              Detected Language: {getDetectedLanguage(lead.full_phone)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              The AI will automatically speak in the appropriate language based
              on the phone number
            </Typography>
          </Box>
        </Alert>

        {/* Basic Call Configuration */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="h6"
            gutterBottom
            sx={{ display: "flex", alignItems: "center", gap: 1 }}
          >
            <SettingsIcon color="primary" />
            Call Configuration
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Company Name"
                value={callConfig.company_name}
                onChange={(e) =>
                  setCallConfig({ ...callConfig, company_name: e.target.value })
                }
                InputProps={{
                  startAdornment: (
                    <BusinessIcon sx={{ mr: 1, color: "text.secondary" }} />
                  ),
                }}
                helperText="The company you're representing"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Caller Name"
                value={callConfig.caller_name}
                onChange={(e) =>
                  setCallConfig({ ...callConfig, caller_name: e.target.value })
                }
                InputProps={{
                  startAdornment: (
                    <PersonIcon sx={{ mr: 1, color: "text.secondary" }} />
                  ),
                }}
                helperText="Your name (e.g., 'John', 'Sarah')"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Product/Service Name"
                value={callConfig.product_name}
                onChange={(e) =>
                  setCallConfig({ ...callConfig, product_name: e.target.value })
                }
                InputProps={{
                  startAdornment: (
                    <ProductIcon sx={{ mr: 1, color: "text.secondary" }} />
                  ),
                }}
                helperText="What you're selling (e.g., 'ArtroFlex joint pain relief cream')"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Call Objective</InputLabel>
                <Select
                  value={callConfig.call_objective}
                  onChange={(e) =>
                    setCallConfig({
                      ...callConfig,
                      call_objective: e.target.value,
                    })
                  }
                  label="Call Objective"
                >
                  <MenuItem value="sales">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <ProductIcon fontSize="small" />
                      Direct Sales
                    </Box>
                  </MenuItem>
                  <MenuItem value="followup">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <PhoneIcon fontSize="small" />
                      Follow-up Call
                    </Box>
                  </MenuItem>
                  <MenuItem value="survey">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <SettingsIcon fontSize="small" />
                      Survey/Research
                    </Box>
                  </MenuItem>
                  <MenuItem value="appointment">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <PersonIcon fontSize="small" />
                      Appointment setter/Cold calling
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Call Urgency</InputLabel>
                <Select
                  value={callConfig.call_urgency}
                  onChange={(e) =>
                    setCallConfig({
                      ...callConfig,
                      call_urgency: e.target.value,
                    })
                  }
                  label="Call Urgency"
                >
                  <MenuItem value="low">
                    <Chip
                      label="Low - Build Rapport"
                      color="success"
                      size="small"
                    />
                  </MenuItem>
                  <MenuItem value="medium">
                    <Chip
                      label="Medium - Moderate Urgency"
                      color="warning"
                      size="small"
                    />
                  </MenuItem>
                  <MenuItem value="high">
                    <Chip
                      label="High - Maximum Urgency"
                      color="error"
                      size="small"
                    />
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Box>

        {/* Greeting Configuration */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="h6"
            gutterBottom
            sx={{ display: "flex", alignItems: "center", gap: 1 }}
          >
            <PhoneIcon color="primary" />
            Greeting Configuration
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Voice Selection</InputLabel>
                <Select
                  value={callConfig.voice_name}
                  onChange={(e) =>
                    setCallConfig({
                      ...callConfig,
                      voice_name: e.target.value,
                    })
                  }
                  label="Voice Selection"
                >
                  <MenuItem value="Puck">Puck (Male, Confident)</MenuItem>
                  <MenuItem value="Charon">
                    Charon (Male, Authoritative)
                  </MenuItem>
                  <MenuItem value="Kore">Kore (Female, Warm)</MenuItem>
                  <MenuItem value="Fenrir">Fenrir (Male, Energetic)</MenuItem>
                  <MenuItem value="Aoede">
                    Aoede (Female, Professional)
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Custom Greeting Text"
                multiline
                rows={3}
                value={callConfig.greeting_instruction}
                onChange={(e) =>
                  setCallConfig({
                    ...callConfig,
                    greeting_instruction: e.target.value,
                  })
                }
                helperText="Enter the exact greeting the bot should say (leave empty for auto-generated greeting based on language and product)"
                placeholder="e.g., 'Здравейте! Аз съм Maria от компания QuantumAI. Обаждам се във връзка с АртроФлекс. Интересувате ли се да научите повече?'"
              />
            </Grid>
          </Grid>
        </Box>

        {/* Sales Configuration */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="h6"
            gutterBottom
            sx={{ display: "flex", alignItems: "center", gap: 1 }}
          >
            <ProductIcon color="primary" />
            Sales Strategy
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Main Product Benefits"
                value={callConfig.main_benefits}
                onChange={(e) =>
                  setCallConfig({
                    ...callConfig,
                    main_benefits: e.target.value,
                  })
                }
                helperText="Key selling points to emphasize (e.g., 'natural ingredients, fast pain relief, no side effects')"
                placeholder="natural ingredients, fast pain relief, no side effects"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Special Offers/Deals"
                value={callConfig.special_offer}
                onChange={(e) =>
                  setCallConfig({
                    ...callConfig,
                    special_offer: e.target.value,
                  })
                }
                helperText="Current promotions and deals to mention"
                placeholder="50% off today only, free shipping, 30-day guarantee"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Objection Handling Style</InputLabel>
                <Select
                  value={callConfig.objection_strategy}
                  onChange={(e) =>
                    setCallConfig({
                      ...callConfig,
                      objection_strategy: e.target.value,
                    })
                  }
                  label="Objection Handling Style"
                >
                  <MenuItem value="understanding">
                    <Chip
                      label="Understanding - Empathetic Approach"
                      color="success"
                      size="small"
                    />
                  </MenuItem>
                  <MenuItem value="educational">
                    <Chip
                      label="Educational - Informative Approach"
                      color="info"
                      size="small"
                    />
                  </MenuItem>
                  <MenuItem value="aggressive">
                    <Chip
                      label="Aggressive - Persistent Push"
                      color="error"
                      size="small"
                    />
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Box>

        {/* Advanced Settings */}
        <Accordion sx={{ borderRadius: 2, "&:before": { display: "none" } }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" fontWeight={600}>
              Advanced Prompt Settings
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TextField
              fullWidth
              label="Additional Prompt Instructions"
              multiline
              rows={4}
              value={callConfig.additional_prompt}
              onChange={(e) =>
                setCallConfig({
                  ...callConfig,
                  additional_prompt: e.target.value,
                })
              }
              helperText="Add specific instructions, talking points, or special offers (optional)"
              placeholder="e.g., 'Mention the 50% discount expires tonight', 'Ask about their previous experience with joint pain', 'Focus on natural ingredients'"
            />
          </AccordionDetails>
        </Accordion>

        {/* Prompt Preview */}
        <Box sx={{ mt: 3, p: 2, bgcolor: "grey.50", borderRadius: 2 }}>
          <Typography
            variant="subtitle2"
            gutterBottom
            color="primary"
            fontWeight={600}
          >
            📝 Generated Prompt Preview (Base):
          </Typography>
          <Typography
            variant="body2"
            sx={{
              fontFamily: "monospace",
              fontSize: "0.8rem",
              lineHeight: 1.6,
              color: "text.secondary",
              maxHeight: 120,
              overflow: "auto",
              p: 1,
              bgcolor: "white",
              borderRadius: 1,
              border: "1px solid",
              borderColor: "divider",
            }}
          >
            {generatePromptPreview()}
            {callConfig.additional_prompt && (
              <>
                <br />
                <br />
                <strong>Additional Instructions:</strong>{" "}
                {callConfig.additional_prompt}
              </>
            )}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mt: 1, display: "block" }}
          >
            Note: This will be automatically translated to{" "}
            {getDetectedLanguage(lead.full_phone)} when the call is made
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3, gap: 1 }}>
        <Button onClick={onClose} variant="outlined" disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleMakeCall}
          variant="contained"
          disabled={
            loading ||
            !callConfig.company_name ||
            !callConfig.caller_name ||
            !callConfig.product_name ||
            !callConfig.main_benefits ||
            !callConfig.special_offer
          }
          startIcon={
            greetingStatus === "generating" ? (
              <CircularProgress size={20} color="inherit" />
            ) : (
              getObjectiveIcon(callConfig.call_objective)
            )
          }
          sx={{
            background:
              greetingStatus === "generating"
                ? "linear-gradient(135deg, #666 0%, #444 100%)"
                : "linear-gradient(135deg, #C85C3C 0%, #A0462A 100%)",
            minWidth: 180,
          }}
        >
          {loading && greetingStatus === "generating"
            ? "Generating Greeting..."
            : loading
            ? "Calling..."
            : `Make ${
                callConfig.call_objective === "appointment"
                  ? "Appointment"
                  : callConfig.call_objective
              } Call`}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CustomCallDialog;
