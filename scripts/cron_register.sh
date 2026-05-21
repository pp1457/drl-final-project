#!/bin/bash
# Install or update the cron entry that runs cron_check.sh every 5 minutes.
# Safe to re-run -- replaces any prior entry with the same marker comment.
set -euo pipefail

PROJ="/tmp2/$USER/DRL_final_project"
ENTRY="*/5 * * * * $PROJ/scripts/cron_check.sh > /dev/null 2>&1  # DRL_FINAL_PROJ"

# Get existing crontab (empty if none); drop any prior DRL_FINAL_PROJ line.
EXISTING=$(crontab -l 2>/dev/null | grep -v 'DRL_FINAL_PROJ$' || true)

# Append new entry and reinstall.
printf '%s\n%s\n' "$EXISTING" "$ENTRY" | crontab -

echo "cron entry installed:"
crontab -l | grep DRL_FINAL_PROJ
echo
echo "first cron run will land within 5 minutes."
echo "view status:    cat \$HOME/drl_status_latest.txt"
echo "history:        tail -20 \$HOME/drl_status_history.log"
echo "alerts:         cat \$HOME/drl_ALERT.txt  (file exists iff something's wrong)"
echo
echo "remove with:    $PROJ/scripts/cron_unregister.sh"
