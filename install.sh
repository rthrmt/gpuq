#!/usr/bin/env bash
set -euo pipefail

# GPU Job Queue (gpuq) Installation Script
# This script installs the gpuq daemon and CLI, sets up system directories,
# and installs the systemd service unit.
#
# gpuq is installed into a dedicated system virtualenv at /opt/gpuq/venv and its
# console scripts are symlinked into /usr/local/bin, so every local user gets
# global `gpuq` / `gpuqd` commands without activating anything. Using a venv also
# sidesteps PEP 668 ("externally-managed-environment"), which blocks a plain
# system-wide `pip install` on modern Debian/Ubuntu/Fedora.

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run as root."
    echo "Please run: sudo $0"
    exit 1
fi

VENV_DIR="/opt/gpuq/venv"
BIN_DIR="/usr/local/bin"

echo "=========================================="
echo "Installing gpuq (GPU Job Queue)"
echo "=========================================="

# Step 1: Create system directories. These hold the DB and daemon log, which
# only root (the daemon) touches — users never read them directly, so no group
# is needed. Access to gpuq is via the world-accessible socket, not these dirs.
echo "[1/6] Creating system directories..."
install -d -o root -g root -m 0755 /var/lib/gpuq
install -d -o root -g root -m 0755 /var/log/gpuq
echo "    ✓ /var/lib/gpuq (owner root:root)"
echo "    ✓ /var/log/gpuq (owner root:root)"

# Step 2: Install the package into a dedicated system venv
echo "[2/6] Installing gpuq into a system virtualenv..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A venv is never "externally managed", so pip works regardless of PEP 668.
if ! python3 -m venv "$VENV_DIR"; then
    echo "    ✗ Error: failed to create virtualenv at $VENV_DIR"
    echo "      On Debian/Ubuntu you may need: sudo apt install python3-venv"
    exit 1
fi
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip > /dev/null 2>&1 || true
"$VENV_DIR/bin/python" -m pip install --quiet "$SCRIPT_DIR"
# The venv is created under root's umask, which on some systems (e.g. 0027)
# produces a 0750 /opt/gpuq that non-root users cannot traverse — so the
# symlinked `gpuq` command fails with "Permission denied" for everyone but
# root. Open read+traverse for all (capital X = execute on dirs/binaries only).
# The tree holds no secrets, only the stdlib-only package and its venv.
chmod -R a+rX /opt/gpuq
echo "    ✓ Installed into $VENV_DIR"

# Symlink the console scripts onto a system PATH location so every user can run
# `gpuq` / `gpuqd` directly with no venv activation.
GPUQ_BIN="$BIN_DIR/gpuq"
GPUQD_BIN="$BIN_DIR/gpuqd"
ln -sf "$VENV_DIR/bin/gpuq" "$GPUQ_BIN"
ln -sf "$VENV_DIR/bin/gpuqd" "$GPUQD_BIN"
echo "    ✓ Linked $GPUQ_BIN and $GPUQD_BIN -> $VENV_DIR/bin"

# Step 3: Install the systemd unit, pointing ExecStart at the venv's gpuqd
echo "[3/6] Installing systemd unit..."
# Use the real venv binary (not the symlink) so the unit is robust even if
# /usr/local/bin is later cleaned up.
EXEC_BIN="$VENV_DIR/bin/gpuqd"
if [[ "$EXEC_BIN" != "/usr/local/bin/gpuqd" ]]; then
    echo "    ℹ Pointing ExecStart at $EXEC_BIN"
    SYSTEMD_FILE="/tmp/gpuq.service.$$"
    sed "s|ExecStart=/usr/local/bin/gpuqd|ExecStart=$EXEC_BIN|g" \
        "$SCRIPT_DIR/systemd/gpuq.service" > "$SYSTEMD_FILE"
    install -m 0644 "$SYSTEMD_FILE" /etc/systemd/system/gpuq.service
    rm -f "$SYSTEMD_FILE"
else
    install -m 0644 "$SCRIPT_DIR/systemd/gpuq.service" /etc/systemd/system/gpuq.service
fi
echo "    ✓ Systemd unit installed: /etc/systemd/system/gpuq.service"

# Step 4: Install configuration file
echo "[4/6] Installing configuration..."
if [[ ! -f /etc/gpuq/gpuq.conf ]]; then
    mkdir -p /etc/gpuq
    install -m 0644 "$SCRIPT_DIR/etc/gpuq.conf.example" /etc/gpuq/gpuq.conf
    echo "    ✓ Config installed: /etc/gpuq/gpuq.conf (from example)"
else
    echo "    ℹ Config already exists: /etc/gpuq/gpuq.conf (not overwriting)"
fi

# Step 5: Reload systemd daemon
echo "[5/6] Reloading systemd..."
systemctl daemon-reload
echo "    ✓ Systemd configuration reloaded"

# Step 6: Print next steps
echo "[6/6] Installation complete!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo ""
echo "1. Enable and start the service:"
echo "   sudo systemctl enable --now gpuq"
echo ""
echo "2. Verify the service is running:"
echo "   sudo systemctl status gpuq"
echo "   gpuq status"
echo ""
echo "All local users can use 'gpuq' immediately once the service is running —"
echo "the socket is world-accessible (mode 0666), so no group membership is needed."
echo "=========================================="
