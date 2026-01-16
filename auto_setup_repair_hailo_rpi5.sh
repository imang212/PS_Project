#!/usr/bin/env bash
# Auto setup and repair script for Hailo on Raspberry Pi 5 AI HAT
set -e

#USER_NAME=$(whoami)
#HOME_DIR=/home/$USER_NAME

#echo "[1/10] Fixing permissions in home directory..."
#sudo chown -R $USER_NAME:$USER_NAME $HOME_DIR

# Determine the correct config.txt location
CONFIG_FILE="/boot/config.txt"
if [ -f "/boot/firmware/config.txt" ]; then
    CONFIG_FILE="/boot/firmware/config.txt"
fi

echo "Checking PCIe Gen 3.0 setting in $CONFIG_FILE..."
if ! grep -q "^dtparam=pciex1_gen=3$" "$CONFIG_FILE"; then
    echo "Entry not found. Enabling PCIe Gen 3.0..."
    # FIX: Actually write the line to the file
    echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_FILE" > /dev/null
else
    echo "PCIe Gen 3.0 is already enabled."
fi

# Update and upgrade the system
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo reboot now

sudo apt install dkms
sudo apt install hailo-all
sudo apt install hailort=4.20.0-1 hailo-dkms=4.19.0-1 python3-hailort=4.20.0-1 hailo-tappas-core=3.31.0+1-1 
sudo reboot

echo "[2/10] Installing base system dependencies..."
sudo apt-get update && sudo apt-get install -y \
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
  gir1.2-gtk-3.0 



# Download and install HailoRT and PCIe driver packages
#sudo dpkg -i hailort_4.23.0_arm64.deb hailort-pcie-driver_4.23.0_all.deb

cd hailo-rpi5-examples
if [ -d venv_hailo_rpi_examples ]; then
    rm -rf venv_hailo_rpi_examples
fi
./install.sh

echo "[4/10] Verifying HailoRT installation..."
hailortcli --version || true
hailortcli scan || true
hailortcli fw-control identify || true
lsmod | grep hailo || true
echo "--- Checking installed hailo packages ---"
dpkg -l | grep hailo


echo "[5/10] Cloning and building TAPPAS from official repository..."
cd $HOME_DIR
if [ ! -d tappas ]; then
    git clone --branch v5.2.0 --depth 1 https://github.com/hailo-ai/tappas.git
fi
cd `tappas`
mkdir hailort
git clone --branch v5.2.0 --depth 1 https://github.com/hailo-ai/hailort.git hailort/sources


rm -rf ~/.cache/gstreamer-1.0/
sudo rm -rf /root/.cache/gstreamer-1.0/
if [ -f /usr/lib/$(uname -m)-linux-gnu/gstreamer-1.0/libgsthailotools.so ]; then
    sudo rm -rf /usr/lib/$(uname -m)-linux-gnu/gstreamer-1.0/libgsthailotools.so 
    echo "Removing existing libgsthailotools.so..."
fi

if [ ! -f /etc/lsb-release ]; then
    sudo tee /etc/lsb-release > /dev/null <<EOF
DISTRIB_ID=Raspbian
DISTRIB_RELEASE=12
DISTRIB_CODENAME=bookworm
DISTRIB_DESCRIPTION="Raspbian GNU/Linux 12 (bookworm)"
EOF
fi
cd ~/tappas/tools/run_app
cp requirements_24_04.txt requirements_12.txt
cd $HOME_DIR/tappas


./install.sh --skip-hailort

# Install GStreamer and its plugins
sudo apt-get install -y gstreamer1.0-*
# Install additional GStreamer plugins for RTSP support
sudo apt-get install -y gir1.2-gst-rtsp-server-1.0 gir1.2-gstreamer-1.0
# Install Python packages for MQTT and WebSockets
sudo apt install -y python3-paho-mqtt python3-websockets

#unset GST_PLUGIN_PATH
#unset LD_LIBRARY_PATH
#echo "[6/10] Fixing PKG_CONFIG_PATH..."
#echo 'export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH' >> $HOME_DIR/.bashrc
# export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH

echo "[7/10] Verifying TAPPAS installation..."
pkg-config --modversion hailo-tappas-core || true
pkg-config --variable=tappas_postproc_lib_dir hailo-tappas-core || true

ls -la /lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo*.so || true
sudo ln -s /usr/lib/aarch64-linux-gnu/libgsthailometa.so.5.2.0 /usr/lib/aarch64-linux-gnu/libgsthailometa.so.3

echo "[8/10] Installing hailo-rpi5-examples..."
cd $HOME_DIR
if [ ! -d hailo-rpi5-examples ]; then
    git clone --branch 25.7.0 --depth 1 https://github.com/hailo-ai/hailo-rpi5-examples.git
fi

cd $HOME_DIR
if [ ! -d hailo_model_zoo ]; then
    git clone --branch v5.2.0 --depth 1 https://github.com/hailo-ai/hailo_model_zoo.git
fi


echo "[9/10] Final verification..."
gst-inspect-1.0 | grep hailo || true


echo "Setup and repair completed successfully!"