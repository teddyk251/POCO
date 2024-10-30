# Installing CUDA 11

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda

echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
ls /usr/local/cuda-11.8/lib64/libcusparse.so.11



# If existing NVIDA drivers are installed, remove them to ensure compatability with the CUDA version
sudo apt-get purge nvidia*

sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install build-essential dkms

# Add the CUDA repository public GPG key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo apt-key add 7fa2af80.pub
rm 7fa2af80.pub

# Add the repository (replace 'ubuntu2004' with your Ubuntu version)
sudo bash -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo apt-get update
sudo apt-get install nvidia-driver-520
sudo reboot

# Resume installation after reboot
# Install Dependencies
conda config --set channel_priority disabled
conda env create -f environment.yml
conda activate poco
pip install -r requirements.txt


