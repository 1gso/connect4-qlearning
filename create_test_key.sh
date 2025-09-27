#!/usr/bin/env bash
set -euo pipefail

USER="Zarmot"
HOST="24.130.208.7"
PORT="22"
KEY="$HOME/.ssh/id_ed25519_bitvise"
ALIAS="my-sftp"
CONFIG="$HOME/.ssh/config"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# Fresh key each run (simple)
rm -f "$KEY" "$KEY.pub"
ssh-keygen -t ed25519 -a 100 -C "${USER}@${HOST}-$(date +%Y%m%d)" -f "$KEY" -N ""

# Ensure ~/.ssh/config and add a Host block if missing
touch "$CONFIG"
chmod 600 "$CONFIG"
if ! grep -qE "^Host[[:space:]]+$ALIAS\b" "$CONFIG"; then
  {
    echo
    echo "Host $ALIAS"
    echo "  HostName $HOST"
    echo "  User $USER"
    echo "  Port $PORT"
    echo "  IdentityFile $KEY"
    echo "  IdentitiesOnly yes"
  } >> "$CONFIG"
fi

echo
echo "========== COPY THIS LINE INTO BITVISE (exactly one line) =========="
cat "$KEY.pub"
echo "===================================================================="
echo "Fingerprint:"
ssh-keygen -lf "$KEY"
echo

read -rp "Press Enter AFTER you paste & Save the key in Bitvise to test... " _

echo ":: Testing via alias: $ALIAS"
if sftp -b /dev/null \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "$ALIAS" >/dev/null 2>&1; then
  echo "✅ Success. You can now use:  ssh $ALIAS   or   sftp $ALIAS"
  exit 0
else
  echo "❌ Auth failed. Run:  ssh -vvv $ALIAS"
  exit 1
fi
