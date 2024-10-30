# Run this script to download the ShapeNet dataset and move it to the POCO data directory
# RUN this script after running install.sh and cloning the POCO repository
git clone https://github.com/autonomousvision/occupancy_networks.git
cd occupancy_networks

# Download the ShapeNet dataset
bash scripts/download_data.sh


# Move downloaded data to POCO data directory
mkdir ../POCO/data
mv data/ShapeNet/ ../POCO/data/ShapeNet


# Move back to the POCO directory
cd ../POCO