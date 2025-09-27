#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURE these once per script-run ===
USER="${USER:-sftpuser}"          # login used on the server
HOST="${HOST:-24.130.208.7}"     # target IP
PORT="${PORT:-22}"               # target port
KEY="${KEY:-$HOME/.ssh/id_ed25519_sftp_one}"  # local private key path

# === ensure .ssh perms ===
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# === generate key if missing ===
if [[ ! -f "$KEY" ]]; then
  ssh-keygen -t ed25519 -a 100 -f "$KEY" -N "" -C "${USER}@${HOST}-$(date +%Y%m%d)" >/dev/null
  chmod 600 "$KEY"
  chmod 644 "$KEY.pub"
  echo "Created key: $KEY"
fi

# === show public key for you to paste into server's SFTP/public-key UI ===
echo
echo "----- PUBLIC KEY (paste this into server) -----"
cat "${KEY}.pub"
echo "-----------------------------------------------"
echo

# === quick connectivity test (uses the key above) ===
SFTP_OPTS=(
  -o IdentitiesOnly=yes
  -o IdentityFile="$KEY"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -P "$PORT"
)

echo "Testing SFTP to ${USER}@${HOST}:${PORT} ..."
if sftp "${SFTP_OPTS[@]}" "${USER}@${HOST}" <<< $'quit' >/dev/null 2>&1; then
  echo "SFTP OK — public-key auth accepted."
  exit 0
else
  echo "SFTP failed — server didn't accept the key (or network issue)."
  echo "After you paste the public key on the server, re-run this script to test."
  exit 1
fi
