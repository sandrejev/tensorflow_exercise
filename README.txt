# EMBL HPC cluster setup

module load Anaconda3
module load CUDA/8.0.44-GCC-5.4.0-2.26
source activate /g/scb/patil/andrejev/python36

# Download and link cuDNN/6
# wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
# tar -xzf cudnn-8.0-linux-x64-v6.0.tgz
# conda create --no-default-packages --copy -m -p /g/scb/patil/andrejev/python36 python tensorflow-gpu pandas
export LD_LIBRARY_PATH=/g/scb/patil/andrejev/python36/cudnn6/lib64/:$LD_LIBRARY_PATH