import React from "react";
import { Box, Skeleton, Grid, Card, CardContent } from "@mui/material";

export const DashboardSkeleton = () => (
  <Box sx={{ mt: 4 }}>
    <Grid container spacing={3} mb={3}>
      {[1, 2, 3, 4].map((item) => (
        <Grid item xs={12} sm={6} md={3} key={item}>
          <Card
            className="glass-effect"
            sx={{ animation: "pulse 1.5s ease-in-out infinite" }}
          >
            <CardContent>
              <Skeleton variant="text" width="60%" height={24} />
              <Skeleton variant="text" width="40%" height={48} sx={{ mt: 1 }} />
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Card className="glass-effect">
          <CardContent>
            <Skeleton variant="text" width="40%" height={32} />
            <Skeleton
              variant="rectangular"
              height={300}
              sx={{ mt: 2, borderRadius: 2 }}
            />
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card className="glass-effect">
          <CardContent>
            <Skeleton variant="text" width="40%" height={32} />
            <Skeleton
              variant="circular"
              width={250}
              height={250}
              sx={{ mt: 2, mx: "auto" }}
            />
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  </Box>
);

export const TableSkeleton = ({ rows = 5 }) => (
  <Box>
    {[...Array(rows)].map((_, index) => (
      <Box
        key={index}
        sx={{
          p: 2,
          borderBottom: "1px solid",
          borderColor: "divider",
          animation: "pulse 1.5s ease-in-out infinite",
          animationDelay: `${index * 0.1}s`,
        }}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={3}>
            <Skeleton variant="text" width="80%" />
          </Grid>
          <Grid item xs={3}>
            <Skeleton variant="text" width="60%" />
          </Grid>
          <Grid item xs={2}>
            <Skeleton variant="text" width="50%" />
          </Grid>
          <Grid item xs={2}>
            <Skeleton variant="text" width="70%" />
          </Grid>
          <Grid item xs={2}>
            <Skeleton
              variant="rectangular"
              width="100%"
              height={32}
              sx={{ borderRadius: 2 }}
            />
          </Grid>
        </Grid>
      </Box>
    ))}
  </Box>
);

export const CardSkeleton = () => (
  <Card className="glass-effect">
    <CardContent>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Skeleton variant="text" width="40%" height={32} />
        <Skeleton variant="circular" width={40} height={40} />
      </Box>
      <Skeleton variant="text" width="60%" />
      <Skeleton variant="text" width="80%" sx={{ mt: 1 }} />
      <Skeleton variant="text" width="50%" sx={{ mt: 1 }} />
    </CardContent>
  </Card>
);

export default { DashboardSkeleton, TableSkeleton, CardSkeleton };
