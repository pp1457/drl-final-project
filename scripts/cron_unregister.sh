#!/bin/bash
# Remove the DRL_FINAL_PROJ cron entry. Other crontab lines untouched.
set -euo pipefail
EXISTING=$(crontab -l 2>/dev/null | grep -v 'DRL_FINAL_PROJ$' || true)
printf '%s\n' "$EXISTING" | crontab -
echo "DRL_FINAL_PROJ cron entry removed. Current crontab:"
crontab -l 2>/dev/null || echo "(empty)"
