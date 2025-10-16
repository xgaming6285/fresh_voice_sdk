import React, { useEffect, useState } from "react";
import { Box } from "@mui/material";

const PageTransition = ({ children }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Trigger animation on mount
    setIsVisible(true);

    return () => {
      setIsVisible(false);
    };
  }, []);

  return (
    <Box
      sx={{
        animation: isVisible
          ? "fadeIn 0.5s ease-out, slideUp 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)"
          : "none",
        animationFillMode: "both",
      }}
    >
      {children}
    </Box>
  );
};

export default PageTransition;
