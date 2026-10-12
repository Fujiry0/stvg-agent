#!/usr/bin/env bash
# Restart the tool server if its /health stops answering (checked every minute; restarts are logged).
# While logs/tools_watchdog.pause exists (servers stopped on purpose) it does nothing.
cd "$(dirname "$0")/.."
RSS_CAP_GB=${RSS_CAP_GB:-60}
fails=0
while true; do
  if [ -f logs/tools_watchdog.pause ]; then fails=0; sleep 60; continue; fi
  if curl -sf --max-time 30 http://127.0.0.1:8101/health > /dev/null; then fails=0; else fails=$((fails + 1)); fi
  # Memory cap: restart before the OOM killer picks the server (agents wait for it and resend, see server_wait.py)
  pid=$(pgrep -f '[s]tvg_agent.tools_server' | head -1)
  if [ -n "$pid" ] && [ "$(ps -o rss= -p $pid 2>/dev/null || echo 0)" -gt $((RSS_CAP_GB * 1048576)) ]; then
    echo "$(date '+%F %T') tool server RSS over ${RSS_CAP_GB} GB, restarting" >> logs/tools_watchdog.log
    scripts/restart_tools.sh >> logs/tools_watchdog.log 2>&1 && echo "$(date '+%F %T') restarted" >> logs/tools_watchdog.log
    fails=0; sleep 60; continue
  fi
  # Host memory pressure (swapping) makes /health answer slowly although the server is fine; restarting then only loses
  # the queued requests. Wait, up to 15 more minutes.
  avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if [ $fails -ge 2 ] && [ -n "$pid" ] && [ "${avail_kb:-999999999}" -lt $((4 * 1048576)) ] && [ $fails -lt 17 ]; then
    [ $fails -eq 2 ] && echo "$(date '+%F %T') /health slow under memory pressure (${avail_kb} kB available), waiting" >> logs/tools_watchdog.log
    sleep 60; continue
  fi
  if [ $fails -ge 2 ]; then
    echo "$(date '+%F %T') tool server down, restarting" >> logs/tools_watchdog.log
    [ -f logs/tools_server.log ] && cp logs/tools_server.log "logs/tools_server_crash_$(date +%m%d_%H%M).log"
    scripts/restart_tools.sh >> logs/tools_watchdog.log 2>&1 && echo "$(date '+%F %T') restarted" >> logs/tools_watchdog.log
    fails=0
  fi
  sleep 60
done
