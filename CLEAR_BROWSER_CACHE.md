# 🧹 Clear Browser Cache - IMPORTANT!

## Problem

The browser might be caching old call session data in localStorage or browser cache, causing old recordings to appear even after fixing the backend.

## Solution

Clear all browser data for the CRM application.

## Steps to Clear Cache

### Option 1: Clear localStorage (Quick)

1. Open the CRM app: http://localhost:3000
2. Press `F12` to open DevTools
3. Go to **Console** tab
4. Run these commands:

```javascript
localStorage.clear();
sessionStorage.clear();
location.reload();
```

### Option 2: Clear All Site Data (Recommended)

1. Open the CRM app: http://localhost:3000
2. Press `F12` to open DevTools
3. Go to **Application** tab (Chrome) or **Storage** tab (Firefox)
4. In the left sidebar, find **Storage** section
5. Click **"Clear site data"** button
6. Confirm and refresh the page

### Option 3: Hard Refresh

1. Close all CRM tabs
2. Press `Ctrl + Shift + Delete` to open Clear Browsing Data
3. Select:
   - ✅ Cookies and other site data
   - ✅ Cached images and files
4. Time range: **Last hour** (or All time if needed)
5. Click **Clear data**
6. Open CRM again: http://localhost:3000

### Option 4: Incognito/Private Window (For Testing)

1. Open browser in Incognito/Private mode
2. Go to: http://localhost:3000
3. Test with fresh cache

## What to Check After Clearing

1. **Login as Admin 1**:

   - Create a campaign
   - Make some calls
   - View Sessions page → Should see your recordings

2. **Logout and Login as Admin 2** (different admin):

   - View Sessions page → Should see ZERO recordings from Admin 1
   - Should only see empty list or own recordings

3. **Login as Agent of Admin 1**:
   - View Sessions page → Should only see agent's own recordings
   - Should NOT see admin's recordings or other agents' recordings

## Expected Results

✅ **Agent**: Only sees their own call recordings  
✅ **Admin**: Only sees their own + their agents' recordings  
✅ **New Admin**: Sees ZERO recordings (empty list)  
✅ **Superadmin**: Sees all recordings from all users

## If Problem Persists

If you still see recordings that don't belong to you after clearing cache:

1. **Check Backend Logs**:

   - Look for the API call: `GET /api/crm/sessions`
   - Check what data is being returned

2. **Check Database**:

   ```bash
   python -c "import sqlite3; conn = sqlite3.connect('voice_agent_crm.db'); print(conn.execute('SELECT id, owner_id, session_id FROM call_sessions').fetchall())"
   ```

3. **Verify Owner IDs**:

   - Make sure all call_sessions have a valid owner_id (not NULL)
   - owner_id should match a real user ID in the users table

4. **Check API Response** (in Browser DevTools → Network tab):
   - Find the `/api/crm/sessions` request
   - Check the response data
   - Verify owner_id values

## Notes

- The fix adds `owner_id.isnot(None)` filter to ensure no orphaned sessions leak through
- Browser cache can persist API responses and cause confusion
- Always test in incognito mode when debugging caching issues
- The frontend might also have React state caching - full page refresh helps

## Done! 🎉

After clearing cache, new admins should NOT see any old recordings!
