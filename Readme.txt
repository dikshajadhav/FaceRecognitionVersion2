REQUIREMENTS OF FACE RECOGNITION SYSTEM
  
Hardware:
    Pansonic IP cam BL-C131/C111(Face Recognition)
    Logitech C170 Webcam (Face Data collection)

Python 3.6.x 

Libraries 
    opencv-contrib-python 
    numpy 
    scipy 
    sk-learn 
    Tensorflow==1.5 
    requests 
    autocrop
    imutils
    tkinter
    PIL

Files: 
    Facenet model file in path (model.pb)
    Haar-Cascade file (haarcascade_frontalface_alt) 
    Embedding file (Extracted_Dict_Infogen.pickle) 
    User name list (EachIndividual_Infogen.pickle) 
    Face recognition python file using IP cam (mainip.py)
    Face recognition python file using USB/Integratated cam (mainusb.py) 

Execution: 
    Run following files
    Using IP camera:
    - python mainip.py model18.pb
    OR
    Using IP camera:
    - python mainip.py model18.pb