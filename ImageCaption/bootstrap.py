import os
import subprocess
import sys

def pip_install_required(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def init():
    pip_install_required("kaggle")
    pip_install_required("rouge_score")
    pip_install_required("evaluate")
    pip_install_required("torchmetrics")
    platform = 'None'

    if 'google.colab' in str(get_ipython()):
      print('Running on CoLab')
      platform = 'CoLab'
    else:
      print('Not running on CoLab')
      platform = 'Custom'

    if platform == 'CoLab':
        os.system("cd  /content/drive/Shareddrives/projects_data/image_caption/")
        os.system("rm -r ~/.kaggle")
        os.system("mkdir ~/.kaggle")
        os.system("mv ./kaggle.json ~/.kaggle/")
        os.system("chmod 600 ~/.kaggle/kaggle.json")
        os.system("kaggle datasets list")
        
    elif platform == 'Custom':
        os.environ['KAGGLE_CONFIG_DIR'] = "~/.kaggle"
        os.system("rm -r ~/.kaggle")
        os.system("mkdir ~/.kaggle")
        os.system("cp ./kaggle.json ~/.kaggle/")
        os.system("chmod 600 ~/.kaggle/kaggle.json")
        os.system("kaggle datasets list")
        