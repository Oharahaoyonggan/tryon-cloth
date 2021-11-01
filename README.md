# tryon-cloth
try on cloth
Install Library Packages
The library packages needed to the system are PyTorch Library, CuPy Library, OpenCV Library, Scikit-image Library, Scipy Library, Pillow Library, Mediapipe Library, and Flask Library. These libraries can be download after a new conda environment is created in the anaconda terminal.
Step 1: conda create -n tryon python=3.7 
Step 2: conda activate tryon
Step 3: conda install pip
Step 4: follow table 4.1.2

CHAPTER 4: SYSTEM IMPLEMENTATION, TESTING AND RESULT
Library Packages
CuPy
Scipy Pillow Mediapipe
Flask
Download the system
Command
pip install cupy==6.0.0
pip install scipy==1.6.2
pip install pillow==8.3.1
pip install mediapipe==0.8.7 pip install flask==2.0.1
Table 4.1.2: commands to download the library packages
     PyTorch
   conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
    OpenCV
  pip install opencv- python==4.5.1
   Scikit-image
   pip install scikit- image==0.18.1
         User can get the code from github:
         Pip clone https://github.com/Oharahaoyonggan/tryon-cloth.git cd tryon-cloth



After that download the pretrain checkpoint and unzip the checkpoint and paste it in the checkpoint folder in tryon-cloth https://drive.google.com/file/d/1JGxhpblMuxYZefyxLbgd1PksPkOBfw89/view?usp=sharing
