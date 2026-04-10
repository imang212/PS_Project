#!/usr/bin/env bash
# set: chmod +x auto_setup_repair_hailo_rpi5.sh
# run: ./auto_setup_repair_hailo_rpi5.sh
# see live progress after restart: tail -f ~/hailo_setup.log
# Auto setup and repair script for Hailo Tappas on Raspberry Pi 5 AI HAT
set -e
# Define variables
STATE_FILE="$HOME/.hailo_setup_state"
LOG_FILE="$HOME/hailo_setup.log"
USER_NAME=$(whoami)
HOME_DIR=$HOME
SERVICE_NAME="hailo-setup-resume"

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}
get_stage() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo "0"
    fi
}
set_stage() {
    echo "$1" > "$STATE_FILE"
    log "Moving to stage $1"
}
setup_auto_resume() {
    SCRIPT_PATH="$(realpath "$0")"
    
    # Create systemd service file
    sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Hailo Setup Auto-Resume Service
After=network.target

[Service]
Type=oneshot
User=$USER_NAME
WorkingDirectory=$HOME_DIR
ExecStart=/bin/bash $SCRIPT_PATH
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
EOF

    # Enable the service
    sudo systemctl daemon-reload
    sudo systemctl enable ${SERVICE_NAME}.service
    
    log "Auto-resume enabled via systemd service: ${SERVICE_NAME}"
    log "Logs will be written to: $LOG_FILE"
    log "To monitor progress after reboot, run: tail -f $LOG_FILE"
    log "To check service status: sudo systemctl status ${SERVICE_NAME}"
}
remove_auto_resume() {
    if systemctl list-unit-files | grep -q "${SERVICE_NAME}.service"; then
        sudo systemctl disable ${SERVICE_NAME}.service
        sudo rm -f /etc/systemd/system/${SERVICE_NAME}.service
        sudo systemctl daemon-reload
        log "Auto-resume service disabled and removed"
    fi
}

echo "Hailo RPi5 Setup Script - Resumable Version"
echo "Log file: $LOG_FILE"
echo "State file: $STATE_FILE"
echo ""
STAGE=$(get_stage)
log "Starting setup/repair script at stage $STAGE"

case $STAGE in
    0)
        log "[Stage 0] Initial setup - Enabling PCIe Gen 3.0..."
        setup_auto_resume
        echo ""
        echo "IMPORTANT: After reboot, monitor progress with:"
        echo "  tail -f $LOG_FILE"
        echo ""
        sleep 3
        # Determine the correct config.txt location
        CONFIG_FILE="/boot/config.txt"
        if [ -f "/boot/firmware/config.txt" ]; then
            CONFIG_FILE="/boot/firmware/config.txt"
        fi
        log "Checking PCIe Gen 3.0 setting in $CONFIG_FILE..."
        if ! grep -q "^dtparam=pciex1_gen=3$" "$CONFIG_FILE"; then
            log "Entry not found. Enabling PCIe Gen 3.0..."
            echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_FILE" > /dev/null
        else
            log "PCIe Gen 3.0 is already enabled."
        fi
        log "Updating system firmware and packages..." 
        sudo apt update && sudo apt full-upgrade -y
        sudo rpi-eeprom-update -a
        set_stage 1
        log "Rebooting... (Stage 1 will run after reboot)"
        sleep 2
        sudo reboot now
        ;;
    1)
        log "[Stage 1] Installing Hailo packages..."
        sudo apt update -y
        sudo apt install -y \
          dkms \
          build-essential \
          git wget curl pkg-config \
          python3 python3-venv python3-pip
        # Remove them
        sudo apt remove -y --purge $(dpkg -l | grep hailo | awk '{print $2}') 
        sudo apt autoremove -y
        sudo apt autoclean -y
        #sudo apt install -y dkms 2>&1 | tee -a "$LOG_FILE"
        sudo apt install -y hailo-all 2>&1 | tee -a "$LOG_FILE"
        sudo apt install -y hailort=4.20.0-1 python3-hailort=4.20.0-1 hailo-tappas-core=3.31.0+1-1 2>&1 | tee -a "$LOG_FILE"
        set_stage 2
        log "Rebooting... (Stage 2 will run after reboot)"
        sleep 2
        sudo reboot now
        ;;
    2)
        log "[Stage 2] Downloading and installing HailoRT 4.23.0 and PCIe driver..."
        cd "$HOME_DIR"
        if [ ! -f hailort_4.23.0_arm64.deb ]; then
            log "Downloading hailort_4.23.0_arm64.deb..."
            wget https://hailo-cdn.s3.amazonaws.com/software/hailort/4.23.0/hailort_4.23.0_arm64.deb 2>&1 | tee -a "$LOG_FILE"
        fi
        if [ ! -f hailort-pcie-driver_4.23.0_all.deb ]; then
            log "Downloading hailort-pcie-driver_4.23.0_all.deb..."
            wget https://hailo-cdn.s3.amazonaws.com/software/hailort/4.23.0/hailort-pcie-driver_4.23.0_all.deb 2>&1 | tee -a "$LOG_FILE"
        fi
        if [ ! -f hailort_4.23.0_arm64.deb ] || [ ! -f hailort-pcie-driver_4.23.0_all.deb ]; then
            log "ERROR: Failed to download required packages."
            exit 1
        fi
        sudo apt install -y rpicam-apps 2>&1 | tee -a "$LOG_FILE"

        log "Installing HailoRT packages..."
        sudo dpkg -i hailort_4.23.0_arm64.deb hailort-pcie-driver_4.23.0_all.deb 2>&1 | tee -a "$LOG_FILE"
        log "Installing base system dependencies..."
        sudo apt-get update 2>&1 | tee -a "$LOG_FILE"
        sudo apt-get install -y \
          python3 python3-venv python3-pip python3-dev python3-setuptools python3-virtualenv \
          build-essential cmake git wget curl pkg-config rsync \
          gcc-12 g++-12 gfortran \
          libopenblas-dev libatlas-base-dev \
          libopencv-dev python3-opencv \
          libjpeg-dev libpng-dev \
          libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libavdevice-dev \
          libv4l-dev libxvidcore-dev libx264-dev \
          libhdf5-dev libhdf5-serial-dev \
          libcap-dev libarchive-dev \
          libfreetype6 libcairo2-dev \
          libzmq3-dev \
          libcamera0 ffmpeg x11-utils \
          python3-gi python3-gi-cairo python-gi-dev \
          libgirepository1.0-dev \
          libgstreamer1.0-dev \
          libgstreamer-plugins-base1.0-dev \
          libgstreamer-plugins-bad1.0-dev \
          gstreamer1.0-tools \
          gstreamer1.0-plugins-base \
          gstreamer1.0-plugins-good \
          gstreamer1.0-plugins-bad \
          gstreamer1.0-plugins-ugly \
          gstreamer1.0-libav \
          gstreamer1.0-x \
          gstreamer1.0-alsa \
          gstreamer1.0-gl \
          gstreamer1.0-gtk3 \
          gstreamer1.0-pulseaudio \
          gir1.2-gtk-3.0 \
          gir1.2-gst-rtsp-server-1.0 gir1.2-gstreamer-1.0 \
          python3-paho-mqtt python3-websockets 2>&1 | tee -a "$LOG_FILE"

        log "Verifying HailoRT installation..."
        hailortcli --version 2>&1 | tee -a "$LOG_FILE" || true
        hailortcli scan 2>&1 | tee -a "$LOG_FILE" || true
        hailortcli fw-control identify 2>&1 | tee -a "$LOG_FILE" || true
        lsmod | grep hailo 2>&1 | tee -a "$LOG_FILE" || true
        log "--- Checking installed hailo packages ---"
        dpkg -l | grep hailo 2>&1 | tee -a "$LOG_FILE"

        log "Installing hailo-rpi5-examples..."
        cd "$HOME_DIR"
        if [ ! -d hailo-rpi5-examples ]; then
            git clone --branch 25.7.0 --depth 1 https://github.com/hailo-ai/hailo-rpi5-examples.git 2>&1 | tee -a "$LOG_FILE"
        fi
        sudo apt install -y hailo-tappas-core
        ./install.sh
        log "Cloning and building TAPPAS from official repository..."
        cd "$HOME_DIR"
        if [ ! -d tappas ]; then
            git clone --branch v5.2.0 --depth 1 https://github.com/hailo-ai/tappas.git 2>&1 | tee -a "$LOG_FILE"
        fi
        cd tappas
        mkdir -p hailort
        if [ ! -d hailort/sources ]; then
            git clone --branch v5.2.0 --depth 1 https://github.com/hailo-ai/hailort.git hailort/sources 2>&1 | tee -a "$LOG_FILE"
        fi
        # Clean up old GStreamer cache and libraries
        log "Cleaning up old GStreamer cache..."
        rm -rf ~/.cache/gstreamer-1.0/
        sudo rm -rf /root/.cache/gstreamer-1.0/
        if [ -f /usr/lib/$(uname -m)-linux-gnu/gstreamer-1.0/libgsthailotools.so ]; then
            sudo rm -rf /usr/lib/$(uname -m)-linux-gnu/gstreamer-1.0/libgsthailotools.so 
            log "Removing existing libgsthailotools.so..."
        fi
        log "Setting up TAPPAS for Raspbian Bookworm..."
        if [ ! -f /etc/lsb-release ]; then
            sudo tee /etc/lsb-release > /dev/null <<EOF
DISTRIB_ID=Raspbian
DISTRIB_RELEASE=12
DISTRIB_CODENAME=bookworm
DISTRIB_DESCRIPTION="Raspbian GNU/Linux 12 (bookworm)"
EOF
        fi
        log "Installing TAPPAS..."
        cd "$HOME_DIR/tappas"
        ./install.sh --skip-hailort 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f ~/tappas/tools/run_app/requirements_12.txt ]; then
            cp ~/tappas/tools/run_app/requirements_24_04.txt ~/tappas/tools/run_app/requirements_12.txt
            ./install.sh --skip-hailort 2>&1 | tee -a "$LOG_FILE"
        fi
        log "Verifying TAPPAS installation..."
        pkg-config --modversion hailo-tappas-core 2>&1 | tee -a "$LOG_FILE" || true
        pkg-config --variable=tappas_postproc_lib_dir hailo-tappas-core 2>&1 | tee -a "$LOG_FILE" || true
        # Set up GStreamer Hailo plugin libhailometa.so symlink
        ls -la /lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo*.so 2>&1 | tee -a "$LOG_FILE" || true
        sudo ln -sf /usr/lib/aarch64-linux-gnu/libgsthailometa.so.5.2.0 /usr/lib/aarch64-linux-gnu/libgsthailometa.so.3
        # Set GStreamer plugin paths to avoid conflicts
        GST_PLUGIN_PATH = ""
        GST_PLUGIN_SYSTEM_PATH = "/usr/lib/aarch64-linux-gnu/gstreamer-1.0"

        log "Cloning Hailo Model Zoo..."
        cd "$HOME_DIR"
        if [ ! -d hailo_model_zoo ]; then
            git clone --branch v5.2.0 --depth 1 https://github.com/hailo-ai/hailo_model_zoo.git 2>&1 | tee -a "$LOG_FILE"
        fi
        GST_PLUGIN_PATH=""
        GST_PLUGIN_SYSTEM_PATH="/usr/lib/aarch64-linux-gnu/gstreamer-1.0"
        log "Final verification..."
        gst-inspect-1.0 | grep hailo 2>&1 | tee -a "$LOG_FILE" || true
        set_stage 3
        log "Stage 2 completed successfully!"
        log "Setup is complete! Check the log for details."
        ;;
        
    3)
        log "Setup completed successfully!"
        remove_auto_resume
        rm -f "$STATE_FILE"
        log "All stages complete. Auto-resume has been disabled."
        echo ""
        echo "✓ Installation complete!"
        echo "  Full log: $LOG_FILE"
        ;;
        
    *)
        log "ERROR: Unknown stage: $STAGE"
        exit 1
        ;;
esac