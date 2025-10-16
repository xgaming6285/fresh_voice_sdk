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
  Grid,
  InputAdornment,
  IconButton,
  LinearProgress,
} from "@mui/material";
import {
  PersonAdd as PersonAddIcon,
  Visibility,
  VisibilityOff,
  Person,
  Email,
  Lock,
} from "@mui/icons-material";
import AnimatedBackground from "../components/AnimatedBackground";

const Register = () => {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
    first_name: "",
    last_name: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError("");
  };

  const getPasswordStrength = () => {
    const password = formData.password;
    if (!password) return 0;
    let strength = 0;
    if (password.length >= 6) strength += 25;
    if (password.length >= 10) strength += 25;
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength += 25;
    if (/\d/.test(password)) strength += 25;
    return strength;
  };

  const getPasswordStrengthColor = () => {
    const strength = getPasswordStrength();
    if (strength < 50) return "error";
    if (strength < 75) return "warning";
    return "success";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    // Validation
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }

    setLoading(true);

    const userData = {
      username: formData.username,
      email: formData.email,
      password: formData.password,
      first_name: formData.first_name,
      last_name: formData.last_name,
    };

    const result = await register(userData);

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
        py: 4,
      }}
    >
      <AnimatedBackground />

      <Container maxWidth="md" className="scale-in">
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
              mb: 3,
              textAlign: "center",
            }}
          >
            <Typography
              variant="h2"
              className="wave"
              sx={{
                fontSize: "3.5rem",
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
                fontSize: "2.5rem",
              }}
            >
              Join Voice Agent CRM
            </Typography>
            <Typography
              variant="body1"
              sx={{
                color: "text.secondary",
                fontWeight: 500,
              }}
            >
              Create an admin account to get started
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
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
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
                    className="stagger-1"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        transition: "all 0.3s ease",
                        "&:hover": {
                          transform: "translateY(-2px)",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Email"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleChange}
                    fullWidth
                    required
                    margin="normal"
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Email color="primary" />
                        </InputAdornment>
                      ),
                    }}
                    className="stagger-2"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        transition: "all 0.3s ease",
                        "&:hover": {
                          transform: "translateY(-2px)",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="First Name"
                    name="first_name"
                    value={formData.first_name}
                    onChange={handleChange}
                    fullWidth
                    margin="normal"
                    className="stagger-3"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        transition: "all 0.3s ease",
                        "&:hover": {
                          transform: "translateY(-2px)",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Last Name"
                    name="last_name"
                    value={formData.last_name}
                    onChange={handleChange}
                    fullWidth
                    margin="normal"
                    className="stagger-4"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        transition: "all 0.3s ease",
                        "&:hover": {
                          transform: "translateY(-2px)",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    value={formData.password}
                    onChange={handleChange}
                    fullWidth
                    required
                    margin="normal"
                    helperText="Minimum 6 characters"
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Lock color="primary" />
                        </InputAdornment>
                      ),
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
                    className="stagger-5"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        transition: "all 0.3s ease",
                        "&:hover": {
                          transform: "translateY(-2px)",
                        },
                      },
                    }}
                  />
                  {formData.password && (
                    <Box sx={{ mt: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={getPasswordStrength()}
                        color={getPasswordStrengthColor()}
                        sx={{
                          height: 6,
                          borderRadius: 3,
                          animation: "slideInRight 0.3s ease-out",
                        }}
                      />
                      <Typography
                        variant="caption"
                        color={`${getPasswordStrengthColor()}.main`}
                        sx={{ mt: 0.5, display: "block" }}
                      >
                        Password strength:{" "}
                        {getPasswordStrength() < 50
                          ? "Weak"
                          : getPasswordStrength() < 75
                          ? "Medium"
                          : "Strong"}
                      </Typography>
                    </Box>
                  )}
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Confirm Password"
                    name="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    fullWidth
                    required
                    margin="normal"
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Lock color="primary" />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() =>
                              setShowConfirmPassword(!showConfirmPassword)
                            }
                            edge="end"
                            className="hover-scale"
                          >
                            {showConfirmPassword ? (
                              <VisibilityOff />
                            ) : (
                              <Visibility />
                            )}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                    className="stagger-5"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        transition: "all 0.3s ease",
                        "&:hover": {
                          transform: "translateY(-2px)",
                        },
                      },
                    }}
                  />
                </Grid>
              </Grid>

              <Button
                type="submit"
                variant="contained"
                fullWidth
                size="large"
                disabled={loading}
                startIcon={loading ? null : <PersonAddIcon />}
                className="ripple-container"
                sx={{
                  mt: 3,
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
                  "Create Account"
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
                  Already have an account?{" "}
                  <Link
                    to="/login"
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
                    Login here
                  </Link>
                </Typography>
              </Box>
            </form>
          </Paper>

          {/* Footer */}
          <Box
            className="fade-in"
            sx={{
              mt: 3,
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

export default Register;
