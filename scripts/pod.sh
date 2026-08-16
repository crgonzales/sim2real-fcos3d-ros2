#!/usr/bin/env bash
# Run a command on the Runpod pod over the SSH proxy.
#
#   ./scripts/pod.sh 'nvidia-smi'
#   ./scripts/pod.sh 'tail -20 /workspace/setup.log'
#
# Why this wrapper exists -- three quirks of the Runpod SSH proxy:
#   1. It REQUIRES a PTY, so `ssh host cmd` fails with
#      "Your SSH client doesn't support PTY". We use `ssh -tt` and feed the
#      command on stdin instead of as an argv.
#   2. For the same reason scp/sftp do NOT work (no sftp subsystem, and scp
#      can't allocate a PTY). Move files with scripts/push.sh (base64) or git.
#   3. Proxy auth uses the ACCOUNT-level SSH key from
#      runpod.io -> Settings -> SSH Public Keys, NOT the pod's PUBLIC_KEY env
#      var. Secure-cloud pods have no public IP, so there is no direct-SSH
#      fallback -- the account key is mandatory.
#
# Contributor: Carlos Gonzales
set -euo pipefail

POD_USER="${POD_USER:-5vx5v0ztxddnu9-64411ea0}"
POD_HOST="${POD_HOST:-ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"

[ $# -ge 1 ] || { echo "usage: $0 '<remote command>'" >&2; exit 2; }

# The proxy prints a login banner and the PTY echoes the command back, so we
# fence the real output between sentinels and print only what's inside.
printf 'echo __B__; %s; echo __E__; exit\n' "$*" | ssh -tt \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ConnectTimeout=30 \
    -o ServerAliveInterval=10 \
    -i "$KEY" \
    "${POD_USER}@${POD_HOST}" 2>&1 \
  | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\r//g' \
  | awk '/^__E__/{p=0} p; /^__B__/{p=1}'
