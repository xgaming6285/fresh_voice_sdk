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
    company_name: "",
    caller_name: "",
    product_name: "",
    additional_prompt: "",
    call_objective: "sales", // sales, followup, survey, appointment, confirm_order, promotion_offer, ai_sales_services, companions_services
    main_benefits: "",
    special_offer: "",
    objection_strategy: "understanding", // understanding, aggressive, educational
    voice_name: "Puck", // Gemini voice selection
    greeting_instruction: "", // Custom greeting text
  });

  const [loading, setLoading] = useState(false);
  const [greetingStatus, setGreetingStatus] = useState(""); // "generating", "ready", ""

  // Apply hardcoded demo configurations for specific objectives
  const applyDemoConfig = (objective) => {
    switch (objective) {
      case "promotion_offer":
        return {
          company_name: "JBet Casino",
          caller_name: "Maria",
          product_name:
            "JBet Casino - Ексклузивна Оферта за 100 Безплатни Завъртания",
          main_benefits:
            "100 безплатни завъртания при регистрация, бонус при първи депозит, сигурно лицензирано казино, 24/7 клиентска поддръжка",
          special_offer:
            "Регистрирайте се сега и получете 100 безплатни завъртания незабавно! Освен това, направете първия си депозит и получете специален приветствен бонус. Ще ви изпратим съобщение с линк към уебсайта и детайли за бонуса.",
          objection_strategy: "understanding",
        };
      case "confirm_order":
        return {
          company_name: "Technopolis",
          caller_name: "Stefan",
          product_name: 'Microwave "Jira" - Order #6845175',
          main_benefits:
            "order verification, confirm customer identity, verify delivery address, 3 business days delivery time, warranty included",
          special_offer:
            "Delivery in 3 business days to the registered address. Professional installation available upon request.",
          objection_strategy: "understanding",
        };
      case "appointment":
        return {
          company_name: "Pikolo Digital Marketing",
          caller_name: "Alexandra",
          product_name: "Digital Marketing Services",
          main_benefits:
            "increase online visibility, drive more traffic to your business, professional SEO and social media management, proven results",
          special_offer:
            "Free consultation and business analysis. Let us show you how we can help grow your business with our proven digital marketing strategies.",
          objection_strategy: "educational",
        };
      case "ai_sales_services":
        return {
          company_name: "PropTech AI Solutions",
          caller_name: "Viktor",
          product_name: "AI Асистент за Автоматизация на Недвижими Имоти",
          main_benefits:
            "автоматично отговаряне на клиентски запитвания 24/7, интелигентно квалифициране на купувачи, автоматизирано насрочване на огледи, AI генериране на описания на имоти, виртуални асистенти за първоначален контакт, освобождаване на време за продажби",
          special_offer:
            "Безплатна 15-минутна демонстрация как AI може да автоматизира Вашата агенция. Ще Ви покажем как да обработвате 10 пъти повече запитвания без да наемате допълнителен персонал. Специална оферта: първите 30 дни безплатно за ранни клиенти. Кога можем да насрочим кратка среща - утре следобед или в петък сутрин?",
          objection_strategy: "educational",
        };
      case "companions_services":
        return {
          company_name: "Elite Companions Sofia",
          caller_name: "Diana",
          product_name: "Премиум Ескорт Услуги",
          main_benefits:
            "дискретност и професионализъм, елегантни и образовани дами, гъвкави срещи по график, безопасност и поверителност гарантирани, VIP придружаване за събития",
          special_offer:
            "Първа среща със специална отстъпка 20%. Работим 24/7, гарантираме пълна дискретност. Можем да организираме среща в удобно за Вас време - днес вечерта или утре? Разполагаме с портфолио на нашите дами, което можем да Ви изпратим дискретно.",
          objection_strategy: "understanding",
        };
      default:
        return {
          company_name: "",
          caller_name: "",
          product_name: "",
          main_benefits: "",
          special_offer: "",
          objection_strategy: "understanding",
        };
    }
  };

  // Handle call objective change and apply demo config
  const handleObjectiveChange = (newObjective) => {
    const demoConfig = applyDemoConfig(newObjective);
    setCallConfig({
      ...callConfig,
      call_objective: newObjective,
      ...demoConfig,
    });
  };

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

        // Step 2: Make call with custom greeting AND greeting transcript for context
        // ✅ Pass greeting transcript so Gemini knows what was already said
        await onMakeCall(
          lead,
          callConfig,
          greetingData.greeting_file,
          greetingData.transcript || greetingData.greeting_text
        );
      }

      onClose();
      // Reset form
      setCallConfig({
        company_name: "",
        caller_name: "",
        product_name: "",
        additional_prompt: "",
        call_objective: "sales",
        main_benefits: "",
        special_offer: "",
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
      main_benefits,
      special_offer,
      objection_strategy,
    } = callConfig;

    let basePrompt = "";

    // Handle confirm_order differently - it's customer support, not sales
    if (call_objective === "confirm_order") {
      basePrompt = `You are ${caller_name} from ${company_name}, a customer support representative. `;
      basePrompt +=
        `You are calling to confirm a customer's existing order. The customer has ordered ${product_name}. ` +
        "FOLLOW THIS FLOW EXACTLY: " +
        `1) Greet and introduce yourself: 'Hello, I'm calling from ${company_name}.' ` +
        "2) Ask if you're speaking with the right person by their full name. If they say NO, apologize and say you're looking for [customer name], then end call politely. If YES, continue. " +
        "3) Inform about the order: 'You have an order for a microwave \"Jira\" with order number 6845175.' Ask if they confirm this order. " +
        "4) If they confirm, say 'Okay, we will make a delivery in 3 business days.' Then confirm ALL details: 'Just to confirm, you are [full name] with phone number [phone], and the delivery address is [address]?' " +
        "5) Wait for their confirmation of details. If correct, thank them and end call. If incorrect, ask for correct information. " +
        "Be professional, clear, courteous, and helpful throughout the call. ";

      if (special_offer) {
        basePrompt += `Additional information: ${special_offer}. `;
      }

      basePrompt +=
        "If they have questions or concerns, answer them patiently and professionally.";
    } else {
      // For sales-oriented calls
      basePrompt = `You are ${caller_name} from ${company_name}, a highly persuasive sales representative for ${product_name}. `;

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
      } else if (call_objective === "promotion_offer") {
        basePrompt +=
          "You are calling to present a special casino promotional offer. Inform them about the 100 free spins offer and first deposit bonus. Explain that you will send them a message with website details and registration instructions. Be enthusiastic but not pushy. ";
      }

      if (main_benefits) {
        basePrompt += `Key benefits to emphasize: ${main_benefits}. `;
      }

      if (special_offer) {
        basePrompt += `Current offers: ${special_offer}. `;
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
    }

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
        {/* Demo Configuration Info */}
        {(callConfig.call_objective === "promotion_offer" ||
          callConfig.call_objective === "confirm_order" ||
          callConfig.call_objective === "appointment" ||
          callConfig.call_objective === "ai_sales_services" ||
          callConfig.call_objective === "companions_services") && (
          <Alert severity="success" sx={{ mb: 2, borderRadius: 2 }}>
            <Box>
              <Typography variant="body2" fontWeight={600}>
                🎬 Demo Configuration Active
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Pre-configured settings have been applied for{" "}
                {callConfig.call_objective === "promotion_offer"
                  ? "JBet Casino Promotion"
                  : callConfig.call_objective === "confirm_order"
                  ? "Technopolis Order Confirmation"
                  : callConfig.call_objective === "appointment"
                  ? "Pikolo Digital Marketing"
                  : callConfig.call_objective === "ai_sales_services"
                  ? "PropTech AI for Real Estate"
                  : "Elite Companions Services"}
                . You can edit any field if needed.
              </Typography>
            </Box>
          </Alert>
        )}

        {/* Order Confirmation Flow Info */}
        {callConfig.call_objective === "confirm_order" && (
          <Alert severity="info" sx={{ mb: 2, borderRadius: 2 }}>
            <Box>
              <Typography variant="body2" fontWeight={600}>
                📋 Order Confirmation Call Flow
              </Typography>
              <Typography
                variant="caption"
                color="text.secondary"
                component="div"
              >
                The AI will follow this sequence:
                <br />
                1️⃣ Verify identity: "Are you [Lead Name]?"
                <br />
                2️⃣ Mention order: Microwave "Jira" #6845175
                <br />
                3️⃣ Confirm details: Name, phone, and delivery address
                <br />
                4️⃣ Inform delivery: 3 business days
                <br />
                <br />
                💡 The lead's name and phone from your database will be used
                automatically.
              </Typography>
            </Box>
          </Alert>
        )}

        {/* AI Sales Services Flow Info */}
        {callConfig.call_objective === "ai_sales_services" && (
          <Alert severity="info" sx={{ mb: 2, borderRadius: 2 }}>
            <Box>
              <Typography variant="body2" fontWeight={600}>
                🏠 AI Real Estate Automation Cold Call Flow
              </Typography>
              <Typography
                variant="caption"
                color="text.secondary"
                component="div"
              >
                Professional cold calling script for real estate agents in
                Bulgarian:
                <br />
                1️⃣ Pattern Interrupt: Professional intro, mention real estate AI
                <br />
                2️⃣ Pain Points: Lost leads, manual property descriptions, missed
                calls
                <br />
                3️⃣ Solution: 24/7 AI assistant, auto property descriptions,
                buyer qualification
                <br />
                4️⃣ Qualify: Ask about their current lead volume and response
                time
                <br />
                5️⃣ Schedule: Book 15-minute demo showing 10x more leads handled
                <br />
                6️⃣ Handle Objections: "Too expensive", "Already have assistant",
                "No time"
                <br />
                <br />
                🎁 Special Offer: First 30 days free for early adopters
                <br />
                📅 If interested and appointment is agreed, the AI will save the
                appointment time (e.g. "tomorrow afternoon", "Friday 10am") in
                the session notes.
              </Typography>
            </Box>
          </Alert>
        )}

        {/* Companions Services Flow Info */}
        {callConfig.call_objective === "companions_services" && (
          <Alert severity="info" sx={{ mb: 2, borderRadius: 2 }}>
            <Box>
              <Typography variant="body2" fontWeight={600}>
                💎 Premium Companions Services Call Flow
              </Typography>
              <Typography
                variant="caption"
                color="text.secondary"
                component="div"
              >
                Professional and discreet approach:
                <br />
                1️⃣ Greeting: Polite, discreet introduction
                <br />
                2️⃣ Service Overview: Premium escort services, VIP events
                <br />
                3️⃣ Benefits: Discretion, elegance, professionalism, 24/7
                availability
                <br />
                4️⃣ Booking: Flexible scheduling, tonight or tomorrow
                <br />
                5️⃣ Portfolio: Offer to send photos discreetly
                <br />
                <br />
                🔒 Emphasis on privacy, safety, and professionalism throughout
                the conversation.
              </Typography>
            </Box>
          </Alert>
        )}

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
                helperText={
                  callConfig.call_objective === "promotion_offer" ||
                  callConfig.call_objective === "confirm_order" ||
                  callConfig.call_objective === "appointment" ||
                  callConfig.call_objective === "ai_sales_services" ||
                  callConfig.call_objective === "companions_services"
                    ? "Demo configuration applied - you can edit if needed"
                    : "The company you're representing"
                }
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
                helperText={
                  callConfig.call_objective === "promotion_offer" ||
                  callConfig.call_objective === "confirm_order" ||
                  callConfig.call_objective === "appointment" ||
                  callConfig.call_objective === "ai_sales_services" ||
                  callConfig.call_objective === "companions_services"
                    ? "Demo configuration applied - you can edit if needed"
                    : "Your name (e.g., 'John', 'Sarah')"
                }
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
                helperText={
                  callConfig.call_objective === "promotion_offer" ||
                  callConfig.call_objective === "confirm_order" ||
                  callConfig.call_objective === "appointment" ||
                  callConfig.call_objective === "ai_sales_services" ||
                  callConfig.call_objective === "companions_services"
                    ? "Demo configuration applied - you can edit if needed"
                    : "What you're selling (e.g., 'ArtroFlex joint pain relief cream')"
                }
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Call Objective</InputLabel>
                <Select
                  value={callConfig.call_objective}
                  onChange={(e) => handleObjectiveChange(e.target.value)}
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
                  <MenuItem value="confirm_order">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <SettingsIcon fontSize="small" />
                      Confirm Order
                    </Box>
                  </MenuItem>
                  <MenuItem value="promotion_offer">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <ProductIcon fontSize="small" />
                      Promotion Offer
                    </Box>
                  </MenuItem>
                  <MenuItem value="ai_sales_services">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <SettingsIcon fontSize="small" />
                      AI Real Estate Services (Bulgarian)
                    </Box>
                  </MenuItem>
                  <MenuItem value="companions_services">
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <PersonIcon fontSize="small" />
                      Companions Services (Bulgarian)
                    </Box>
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
                placeholder="e.g., 'Здравейте! Аз съм Maria от компания PropTechAI. Обаждам се във връзка с АртроФлекс. Интересувате ли се да научите повече?'"
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
                helperText={
                  callConfig.call_objective === "promotion_offer" ||
                  callConfig.call_objective === "confirm_order" ||
                  callConfig.call_objective === "appointment" ||
                  callConfig.call_objective === "ai_sales_services" ||
                  callConfig.call_objective === "companions_services"
                    ? "Demo configuration applied - you can edit if needed"
                    : "Key selling points to emphasize (e.g., 'natural ingredients, fast pain relief, no side effects')"
                }
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
                helperText={
                  callConfig.call_objective === "promotion_offer" ||
                  callConfig.call_objective === "confirm_order" ||
                  callConfig.call_objective === "appointment" ||
                  callConfig.call_objective === "ai_sales_services" ||
                  callConfig.call_objective === "companions_services"
                    ? "Demo configuration applied - you can edit if needed"
                    : "Current promotions and deals to mention"
                }
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
                  : callConfig.call_objective === "confirm_order"
                  ? "Confirm Order"
                  : callConfig.call_objective === "promotion_offer"
                  ? "Promotion"
                  : callConfig.call_objective === "ai_sales_services"
                  ? "Real Estate AI"
                  : callConfig.call_objective === "companions_services"
                  ? "Companions"
                  : callConfig.call_objective
              } Call`}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CustomCallDialog;
