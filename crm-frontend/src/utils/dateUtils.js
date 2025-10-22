/**
 * Date and time utility functions for proper timezone handling
 */

/**
 * Parse a date string from the backend (assumed to be in UTC) and return a Date object
 * This ensures proper timezone conversion regardless of how the backend sends the date
 *
 * @param {string|Date} dateString - Date string or Date object from backend
 * @returns {Date} - JavaScript Date object with proper timezone
 */
export function parseUTCDate(dateString) {
  if (!dateString) return null;

  // If already a Date object, return it
  if (dateString instanceof Date) return dateString;

  // If the string already has timezone info (Z or +/-offset), use it directly
  if (dateString.endsWith("Z") || dateString.match(/[+-]\d{2}:\d{2}$/)) {
    return new Date(dateString);
  }

  // Otherwise, append 'Z' to indicate it's UTC
  return new Date(dateString + "Z");
}

/**
 * Format a date to local string with both date and time
 *
 * @param {string|Date} dateString - Date string or Date object
 * @param {Object} options - Intl.DateTimeFormat options
 * @returns {string} - Formatted date string in user's local timezone
 */
export function formatDateTime(dateString, options = {}) {
  const date = parseUTCDate(dateString);
  if (!date) return "Unknown";

  const defaultOptions = {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    ...options,
  };

  return date.toLocaleString(undefined, defaultOptions);
}

/**
 * Format just the date part
 *
 * @param {string|Date} dateString - Date string or Date object
 * @returns {string} - Formatted date string
 */
export function formatDate(dateString) {
  const date = parseUTCDate(dateString);
  if (!date) return "Unknown";

  return date.toLocaleDateString();
}

/**
 * Format just the time part
 *
 * @param {string|Date} dateString - Date string or Date object
 * @returns {string} - Formatted time string
 */
export function formatTime(dateString) {
  const date = parseUTCDate(dateString);
  if (!date) return "Unknown";

  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/**
 * Get relative time string (e.g., "2 hours ago", "Just now")
 *
 * @param {string|Date} dateString - Date string or Date object
 * @returns {Object} - Object with timeAgo string and color for UI
 */
export function getRelativeTime(dateString) {
  const date = parseUTCDate(dateString);
  if (!date) return { timeAgo: "Never", color: "text.disabled" };

  const now = new Date();
  const diffMs = now - date;
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  let timeAgo = "";
  let color = "text.primary";

  if (diffSeconds < 60) {
    timeAgo = "Just now";
    color = "success.main";
  } else if (diffMinutes < 60) {
    timeAgo = `${diffMinutes}m ago`;
    color = "success.main";
  } else if (diffHours < 24) {
    timeAgo = `${diffHours}h ago`;
    color = "info.main";
  } else if (diffDays < 7) {
    timeAgo = `${diffDays}d ago`;
    color = "warning.main";
  } else {
    timeAgo = formatDate(dateString);
    color = "text.secondary";
  }

  return { timeAgo, color };
}

/**
 * Get full display info for a timestamp
 * Includes relative time and formatted timestamp
 *
 * @param {string|Date} dateString - Date string or Date object
 * @returns {Object} - Object with display information
 */
export function getDateTimeDisplay(dateString) {
  const date = parseUTCDate(dateString);
  if (!date) {
    return {
      relative: { timeAgo: "Never", color: "text.disabled" },
      formatted: "Unknown",
      date: null,
    };
  }

  return {
    relative: getRelativeTime(dateString),
    formatted: formatDateTime(dateString),
    formattedDate: formatDate(dateString),
    formattedTime: formatTime(dateString),
    date: date,
  };
}
