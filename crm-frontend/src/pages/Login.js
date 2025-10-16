import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import {
  Box,
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
} from "@mui/material";
import {
  Login as LoginIcon,
  Visibility,
  VisibilityOff,
  Person,
} from "@mui/icons-material";
import AnimatedBackground from "../components/AnimatedBackground";

const Login = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    const result = await login(formData.username, formData.password);

    if (result.success) {
      navigate("/dashboard");
    } else {
      setError(result.error);
    }

    setLoading(false);
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <AnimatedBackground />

      <Container maxWidth="sm" className="scale-in">
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          {/* Logo/Header */}
          <Box
            className="bounce-in float"
            sx={{
              mb: 4,
              textAlign: "center",
            }}
          >
            <Typography
              variant="h2"
              className="wave"
              sx={{
                fontSize: "4rem",
                mb: 1,
              }}
            >
              🎙️
            </Typography>
            <Typography
              variant="h3"
              className="text-shimmer"
              sx={{
                fontWeight: 800,
                mb: 1,
              }}
            >
              Voice Agent CRM
            </Typography>
            <Typography
              variant="body1"
              sx={{
                color: "text.secondary",
                fontWeight: 500,
              }}
            >
              Welcome back! Sign in to continue
            </Typography>
          </Box>

          <Paper
            className="glass-effect hover-lift gradient-border"
            elevation={0}
            sx={{
              p: 4,
              width: "100%",
              animation: "springIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
              animationDelay: "0.2s",
              animationFillMode: "both",
            }}
          >
            {error && (
              <Alert
                severity="error"
                className="slide-in-right"
                sx={{
                  mb: 3,
                  borderRadius: 3,
                  animation: "bounceIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)",
                }}
              >
                {error}
              </Alert>
            )}

            <form onSubmit={handleSubmit}>
              <TextField
                label="Username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                fullWidth
                required
                margin="normal"
                autoFocus
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Person color="primary" />
                    </InputAdornment>
                  ),
                }}
                sx={{
                  mb: 2,
                  "& .MuiOutlinedInput-root": {
                    transition: "all 0.3s ease",
                    "&:hover": {
                      transform: "translateY(-2px)",
                    },
                  },
                }}
              />

              <TextField
                label="Password"
                name="password"
                type={showPassword ? "text" : "password"}
                value={formData.password}
                onChange={handleChange}
                fullWidth
                required
                margin="normal"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                        className="hover-scale"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                sx={{
                  mb: 3,
                  "& .MuiOutlinedInput-root": {
                    transition: "all 0.3s ease",
                    "&:hover": {
                      transform: "translateY(-2px)",
                    },
                  },
                }}
              />

              <Button
                type="submit"
                variant="contained"
                fullWidth
                size="large"
                disabled={loading}
                startIcon={loading ? null : <LoginIcon />}
                className="ripple-container"
                sx={{
                  mt: 2,
                  mb: 3,
                  py: 1.5,
                  fontSize: "1.1rem",
                  position: "relative",
                  overflow: "hidden",
                  "&::before": {
                    content: '""',
                    position: "absolute",
                    top: 0,
                    left: "-100%",
                    width: "100%",
                    height: "100%",
                    background:
                      "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)",
                    transition: "left 0.5s",
                  },
                  "&:hover::before": {
                    left: "100%",
                  },
                }}
              >
                {loading ? (
                  <Box className="loading-dots">
                    <span />
                    <span />
                    <span />
                  </Box>
                ) : (
                  "Sign In"
                )}
              </Button>

              <Box
                sx={{
                  textAlign: "center",
                  pt: 2,
                  borderTop: "1px solid",
                  borderColor: "divider",
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  Don't have an account?{" "}
                  <Link
                    to="/register"
                    style={{
                      textDecoration: "none",
                      color: "#C85C3C",
                      fontWeight: 600,
                      transition: "all 0.2s ease",
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.textDecoration = "underline";
                      e.target.style.color = "#E07B5F";
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.textDecoration = "none";
                      e.target.style.color = "#C85C3C";
                    }}
                  >
                    Register here
                  </Link>
                </Typography>
              </Box>
            </form>
          </Paper>

          {/* Footer */}
          <Box
            className="fade-in"
            sx={{
              mt: 4,
              textAlign: "center",
              animation: "fadeIn 1s ease-out",
              animationDelay: "0.5s",
              animationFillMode: "both",
            }}
          >
            <Typography variant="caption" color="text.secondary">
              © 2025 Voice Agent CRM. All rights reserved.
            </Typography>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Login;
