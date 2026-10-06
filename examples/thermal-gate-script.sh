#!/bin/bash
# Example custom thermal gate. Point `thermal.command` at it:
#
#   [thermal]
#   command = "/path/to/thermal-gate-script.sh"
#
# Contract: called as `<command> <profile>`; exit 0 = go, exit 1 = use a lighter
# tier (honoured for the `medium` profile, otherwise treated as blocked),
# exit 2 = blocked. Print a reason on stdout; the router logs it.

profile="${1:-standard}"
load=$(sysctl -n vm.loadavg | awk '{print $2}')
cores=$(sysctl -n hw.logicalcpu)

if awk -v l="$load" -v c="$cores" 'BEGIN { exit !(l > 0.8 * c) }'; then
  echo "SKIP_LLM: load $load on $cores cores"
  exit 2
fi
if [ "$profile" = "medium" ] && awk -v l="$load" -v c="$cores" 'BEGIN { exit !(l > 0.5 * c) }'; then
  echo "DOWNGRADE: load $load on $cores cores"
  exit 1
fi
exit 0
