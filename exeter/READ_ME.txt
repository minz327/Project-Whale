# How to get sleap - ull instructions @ https://sleap.ai/installation.html
#Ensure that python is running ARM and not x86 (if using Apple Silicon)
#Check tensor flow is installed with ARM version
#Create new environment:
conda create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.4.1
# activate GUI
sleap-label
# open .slp file in GUI "First_general_model_CWR.v001.slp"


