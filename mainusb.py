import logging
from scipy import misc
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.metrics.pairwise import pairwise_distances #To find out Pairwise distance between x, y[] matrix
from tensorflow.python.platform import gfile # its file I/O wrapper without thread loacking
import numpy as np
import requests
import argparse
import time
import cv2
import csv
import re
from datetime import datetime
import sys
import pickle
import json
import imutils
from imutils.video import VideoStream
from tkinter import *
from tkinter import messagebox
import tkinter.font
from tkinter import font
import PIL
from PIL import Image,ImageTk


distance_treshold = 0.7

class IdData():
    """We Created the class to Keeps track of known identities and calculates id matches"""

    def __init__(self,sess, embeddings, images_placeholder, phase_train_placeholder):
        # print('Loading known identities: ', end='')
        logging.info('class called  %s' % str(datetime.now()))
        self.distance_treshold = 0.7
        # self.id_folder = id_folder
        #self.mtcnn = mtcnn
        self.id_names = []

        with (open("EachIndividual_Infogen.pickle", "rb")) as openfile:
            while True:
                try:
                    self.EachIndividualPersons=pickle.load(openfile)
                except EOFError as e:
                    logging.error(e)
                    break

        with (open("Extracted_Dict_Infogen.pickle", "rb")) as openfile:
            while True:
                try:
                    self.embeddings=pickle.load(openfile)
                except EOFError as e:
                    logging.error(e)
                    break
    
    def FindPersonNameBasedOnEmbeddingDistance(self, embs):
        matching_ids = []
        matching_distances = []
        try:
            distance_matrix = pairwise_distances(embs, self.embeddings)
            logging.info('distance_matrix = pairwise_distances(embs, self.embeddings) %s' % str(datetime.now()))
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                logging.info('min_index = np.argmin(distance_row) %s' % str(datetime.now())) #Returns the indices of the minimum values along an axis.
                if distance_row[min_index] < self.distance_treshold: # if min_index is less then the threshold i.e., 0.7
                    matching_ids.append(self.EachIndividualPersons[min_index]) # If true then append the name into maching_ids[] list 
                    matching_distances.append(distance_row[min_index])# and also append the distance associated with that image
                    logging.info('min_index = np.argmin(distance_row) %s' % str(datetime.now()))
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
            return matching_ids, matching_distances  # FindPersonNameBasedOnEmbeddingDistance returns the name and the distance of that perticular image
        except Exception as e:
            logging.error(e)                

def LoadFeatureExtractedModel(model): 
    model_exp = os.path.expanduser(model) #pass the facenet model path
    if (os.path.isfile(model_exp)):
        print('Loading model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:  # 'gfile' make it more efficient for some backing filesystems.
            graph_def = tf.GraphDef() #'GraphDef' is the class created by the protobuf liberary
            graph_def.ParseFromString(f.read()) 
            tf.import_graph_def(graph_def, name='') # To load the TF Graph
    else:
        raise ValueError('Specify model file, not directory!')
        logging.error(sys.exit("model.pb not found"))

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def Checkin():
    conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['CHECK-IN']
    with open('confirmation.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(conftxt)
    csvFile.close()
    jsdata = { 'id': emp_id,'name':emp_name ,'flag': 1 }  # FLAG 1:Checking 0:Checkout
    json_data = json.dumps(jsdata)
    print('json_data>>>',json_data) #  {"id": "2118", "name": "Ajinkya", "flag": 1}
    # resp = requests.post('http://localhost:8989/tasks/',
    #                      data=json.dumps(jsdata),
    #                      headers={'Content-Type':'application/json'})
    # user.destroy()
    user_msg = 'Hi '+emp_name+', '+'Emp.code: '+emp_id+'\nChecked-in Successfully'


def Telluser():
    MsgBox = messagebox.askquestion ('Check-in warning !','Are you sure you want to Check-in ?',icon = 'question')
    if MsgBox == 'yes':
        Checkin()
    else:
        messagebox.showinfo('Return','You will now return to the application screen')
    # messagebox.showinfo('Check-in Successfully',user_msg)
    # user.destroy()

def Checkout():
    conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['CHECK-OUT']
    with open('confirmation.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(conftxt)
    csvFile.close()
    jsdata = { 'id': emp_id,'name':emp_name ,'flag': 0 }  # FLAG 1:Checking 0:Checkout
    json_data = json.dumps(jsdata)
    print('json_data>>>',json_data) 
    # resp = requests.post('http://localhost:8989/tasks/',
    #                      data=json.dumps(jsdata),
    #                      headers={'Content-Type':'application/json'})
    # user.destroy()

def Askuser():
    MsgBox = messagebox.askquestion ('Check-Out warning !','Are you sure you want to Check-Out for the day ?',icon = 'warning')
    if MsgBox == 'yes':
       Checkout()
    else:
        messagebox.showinfo('Return','You will now return to the application screen')
        # user.destroy()

def Correct():
    conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['CORRECT']
    with open('confirmation.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(conftxt)
    csvFile.close()
    # user.destroy()

def Incorrect():
    conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['INCORRECT']
    with open('confirmation.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(conftxt)
    csvFile.close()

root = Tk()
# window = tkinter.Tk()
root.title('Face Recognition System')
root.geometry("920x530+50+50")
root.configure(background='white')

#Left-corner LOGO
load = Image.open(r"logo/infogen.png")
render = ImageTk.PhotoImage(load)
# image = image.subsample(4, 4) 
img = Label(root,image=render)
img.image = render
img.place(x=-5, y=-10)

banner='Face Recognition System'
tk01=Label(root,text=banner,fg = "green",bg='white',font = "Times 24 italic bold")
tk01.place(x=500, y=10)
tk01.pack(side=TOP)
# canvas0 = Canvas(root, width = 600, height = 50, bg='white' )  #, 
# canvas0.pack()
# canvas0.create_text(400,20,fill="green",font="Times 20 italic bold",justify=CENTER,text=banner)

# img = Image(file="infogen.png")      
# canvas0.create_image(100,10, anchor=NW, image=img) 

root.bind('q', lambda e: root.quit())
lmain = Label(root, bg='#ffffff')
lmain.place(x=0, y=45)
# lmain.pack(side=LEFT)

def main(args):
    global aligned
    #Haar-Cascade classifier file path
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')  #lbpcascade_frontalface_improved.xml
    
    # print("payload_size: {}".format(payload_size))
    with tf.Graph().as_default(): # getting the tensorflow graph() function as default mode
        with tf.Session() as sess: #Start tensorflow session

            logging.info('with tf.Session() as sess: %s' % str(datetime.now()))
            LoadFeatureExtractedModel(args.model) #IT loads the facenet 20170512-110547.pb pre-trained model
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Load anchor IDs
            id_data = IdData(sess, embeddings, images_placeholder, phase_train_placeholder)
            # cap=cv2.VideoCapture(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            vs = VideoStream(src=0, resolution=(640, 480),framerate=32).start()
            
            # timeCheck = time.time()
            while(True):
                # ret, frame = cap.read()
                
                # start=time.time()
                def show_frame():
                    try:
                        frame = vs.read()
                        # img_resp=requests.get('http://192.168.1.7/SnapshotJPEG?Resolution=640x480')
                        # img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
                        # frame = cv2.imdecode(img_arr,1)
                        # median = cv2.medianBlur(frame, 5)
                        
                        frame = cv2.flip(frame,180)
                    except:
                        print('Chech Network connection for IP cam')

                    # Add Logo to live video stream
                    # logo = cv2.imread('logo.png')
                    # rows,cols,channels = logo.shape
                    # logo_gray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
                    # # Create a mask of the logo and its inverse mask
                    # ret, mask = cv2.threshold(logo_gray, 200, 255, cv2.THRESH_BINARY_INV)
                    # # Now just extract the logo
                    # mask_inv = cv2.bitwise_not(mask)
                    # logo_fg = cv2.bitwise_and(logo,logo,mask = mask)
                    # # To put logo on top-left corner, create a Region of Interest (ROI)
                    # roi = frame[0:rows, 0:cols ] 
                    # # Now blackout the area of logo in ROI
                    # frm_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                    # # Next add the logo to each video frame
                    # dst = cv2.add(frm_bg,logo_fg)
                    # frame[0:rows, 0:cols ] = dst

                    #To display Date & time
                    cv2.putText(frame, dt.strftime("Date: %d/%m/%y Time: %H:%M:%S"),(10,470), font, 0.5, (255,118,0), 1)

                    faces = faceCascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
                    print('faces>>',faces)
                    face_patches=[]
                    # cv2.rectangle(frame, (261,174),(457,380),(0,0,0),1,cv2.LINE_4)
                    for (x,y,w,h) in faces:
                        print('bounding boox of faces',x,y,w,h)

                        cv2.rectangle(frame, (x-20,y-20), (x+w+20,y+h+20), (255,0,0), 2)
                        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,cv2.LINE_8)
                        #roi_gray = gray[y:y+h, x:x+w]
                        try:
                            roi_color = frame[y-10:y+h+15, x-10:x+w+15]
                            print('shapee roi',roi_color.shape)
                        except:
                            roi_color = frame[y:y+h, x:x+w]
                            print('shapee roi',roi_color.shape)
                        try:
                            aligned = misc.imresize(roi_color, (160, 160), interp='bilinear')
                        except:
                            print("aligned = misc.imresize(roi_color, (160, 160), interp='bilinear')")
                            pass
                        try:
                            prewhitened = prewhiten(aligned)
                            prewhitened = prewhiten(prewhitened)
                            face_patches.append(prewhitened)
                        except:
                            print('aligned')
                            pass

                    if len(face_patches) > 0:
                        face_patches = np.stack(face_patches) #we created sequence of array of Face images

                        feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                        print('##################################') # Passing this face arrays as Image Placeholder its  variablewhich will only assign the image information at later stage
                        embs = sess.run(embeddings, feed_dict=feed_dict) #Feed placeholders to the tensorflow graphs using feed_dictionary 

                        print('Matches in frame:',embs.shape)
                        matching_ids, matching_distances = id_data.FindPersonNameBasedOnEmbeddingDistance(embs) #Passing the image embeddings to 'id_data' which is class veriable
                        for (x,y,w,h),matching_id, dist in zip(faces,matching_ids, matching_distances):

                            try:
                                # print('<>-<>-<>',matching_id, dist)
                                #matching_id = re.sub('[_]', ' ', matching_id)
                                # canvas1 = Canvas(root, width = 320, height = 620, bg='#ffffff')  #,  bg='#ffffff' 
                                # canvas1.pack(side=RIGHT)

                                if matching_id != None and dist != None and dist <= 0.70:
                                    id=matching_id.split('_')
                                    match=id[0]+' '+id[1]
                                    cv2.rectangle(frame, (x-20,y-20), (x+w+20,y+h+20), (0,0,0), 2)#bounding box
                                    # cv2.rectangle(frame, (x-20,y-70), (x+w+22, y-22), (0,0,0), -1) # text background above the bounding box

                                    cv2.rectangle(frame, (x-20,y-18), (x+w+20, y+15), (0,0,0), -1) # text background inside the bounding box
                                    # cv2.putText(frame, match, (x-18,y-40), font, 0.5, (255,255,255), 1) # text above the bounding box

                                    cv2.putText(frame, match, (x-18,y+3), font, 0.5, (0,255,255), 1) # text inside bounding box
                                    # cv2.putText(frame, str(dist), (10,15), font, 0.5, (255,0,0), 1) # Display distance threshold

                                    font2 = cv2.FONT_HERSHEY_DUPLEX
                                    # if z == ord("u"):
                                    emp_id=str(id[0])
                                    emp_name=str(id[1])
                                    user_name=str(id[0] +' '+id[1])
                                    print('>>>>',emp_id,'||',emp_name,'<<<<')

                                    uid = StringVar()
                                    uid.set(emp_id)
                                    print('~~~~uid~~~~',uid.get())

                                    uname=StringVar()
                                    uname.set(emp_name)
                                    print('~~~~uname~~~~',uname.get())

                                    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                                    
                                    hi='Hi,'
                                    tk0=Label(root,text=hi,fg = "red",bg='white',font = "Times 18 bold")
                                    tk0.place(x=650, y=80)
                                    # tk0.after(1)

                                    tk1=Label(root,text='Name: ',fg = "red",bg='white',font = "Times 18 bold")
                                    tk1.place(x=650, y=110)

                                    # global tk2
                                    # tk2.delete("all")
                                    tk21=Label(root,text='                  ',fg = "white",bg="white",font = "Times 18 italic bold")
                                    tk21.place(x=720, y=110)

                                    tk2=Label(root,text=uname.get(),fg = "green",bg="white",font = "Times 18 italic bold")
                                    tk2.place(x=720, y=110)

                                    # global tk3
                                    # tk3.delete("all")
                                    tk3=Label(root,text='Emp.Id: ',fg = "red",bg='white',font = "Times 18 bold")
                                    tk3.place(x=650, y=140)

                                    tk41=Label(root, text='            ',fg = "white",bg='white',font = "Times 18 italic bold")
                                    tk41.place(x=740, y=140)

                                    tk4=Label(root, text=uid.get(),fg = "green",bg='white',font = "Times 18 italic bold")
                                    tk4.place(x=740, y=140)

                                    welcome="Welcome to Infogen-labs\n"
                                    user_text= welcome+'What you want to do ?'

                                    tk51=Label(root,text='                         \n                             ',fg = "white",bg='white',font = "Times 18 bold")
                                    tk51.place(x=650, y=180)

                                    tk5=Label(root,text=user_text,fg = "blue",bg='white',font = "Times 18 bold")
                                    tk5.place(x=650, y=180)

                                    def Checkin():
                                        conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['CHECK-IN']
                                        with open('confirmation.csv', 'a') as csvFile:
                                            writer = csv.writer(csvFile)
                                            writer.writerow(conftxt)
                                        csvFile.close()
                                        jsdata = { 'id': emp_id,'name':emp_name ,'flag': 1 }  # FLAG 1:Checking 0:Checkout
                                        json_data = json.dumps(jsdata)
                                        print('json_data>>>',json_data) #  {"id": "2118", "name": "Ajinkya", "flag": 1}
                                        # resp = requests.post('http://localhost:8989/tasks/',
                                        #                      data=json.dumps(jsdata),
                                        #                      headers={'Content-Type':'application/json'})
                                        # user.destroy()
                                        user_msg = 'Hi '+emp_name+', '+'Emp.Id: '+emp_id+'\nChecked-in Successfully'
                                        messagebox.showinfo('Check-in message',user_msg)


                                    def Telluser():
                                        MsgBox = messagebox.askquestion ('Check-in warning !','Are you sure you want to Check-in ?',icon = 'question')
                                        if MsgBox == 'yes':
                                            Checkin()
                                        else:
                                            messagebox.showinfo('Return','Press OK to continue')
                                        # messagebox.showinfo('Check-in Successfully',user_msg)
                                        # user.destroy()

                                    def Checkout():
                                        conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['CHECK-OUT']
                                        with open('confirmation.csv', 'a') as csvFile:
                                            writer = csv.writer(csvFile)
                                            writer.writerow(conftxt)
                                        csvFile.close()
                                        jsdata = { 'id': emp_id,'name':emp_name ,'flag': 0 }  # FLAG 1:Checking 0:Checkout
                                        json_data = json.dumps(jsdata)
                                        print('json_data>>>',json_data) 
                                        # resp = requests.post('http://localhost:8989/tasks/',
                                        #                      data=json.dumps(jsdata),
                                        #                      headers={'Content-Type':'application/json'})
                                        user_msg = 'Hi '+emp_name+', '+'Emp.Id: '+emp_id+'\nCheck-out Successfully'
                                        messagebox.showinfo('Check-out message',user_msg)

                                    def Askuser():
                                        MsgBox = messagebox.askquestion ('Check-Out warning !','Are you sure you want to Check-Out for the day ?',icon = 'warning')
                                        if MsgBox == 'yes':
                                           Checkout()
                                        else:
                                            messagebox.showinfo('Return','Press OK to continue')
                                            # user.destroy()

                                    def Correct():
                                        conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['CORRECT']
                                        with open('confirmation.csv', 'a') as csvFile:
                                            writer = csv.writer(csvFile)
                                            writer.writerow(conftxt)
                                        csvFile.close()
                                        # user.destroy()

                                    def Incorrect():
                                        conftxt=[emp_id]+[emp_name]+[dist,dt.strftime("%d/%m/%y %H:%M:%S")]+['INCORRECT']
                                        with open('confirmation.csv', 'a') as csvFile:
                                            writer = csv.writer(csvFile)
                                            writer.writerow(conftxt)
                                        csvFile.close()
                                        msg='Incorrect Response request registered\n Please report to AI team'
                                        messagebox.showinfo('Incorrect',msg)

                                    # def Refresh():
                                    #     # tk2.destroy()
                                    #     # tk4.destroy()
                                    #     tk2.update()
                                    #     tk4.update()

                                    #     tk2=Label(root,text=uname.get(),fg = "white",font = "Times 18 italic bold")
                                    #     tk2.place(x=720, y=110)

                                    #     tk4=Label(root, text=uid.get(),fg = "white",font = "Times 18 italic bold")
                                    #     tk4.place(x=740, y=140)

                                    button1 = Button(root, text='CHECK-IN',fg="green",command=Telluser,font=('Helvetica', '14'))
                                    button1.place(x=730,y=250)
                                    # canvas1.create_window(140, 230, window=button1)
                                    root.bind("a", lambda event: Telluser())

                                    button2 = Button(root, text='CHECK-OUT', fg="red",command=Askuser,font=('Helvetica', '14'))
                                    button2.place(x=720,y=300)
                                    # canvas1.create_window(140, 280, window=button2)
                                    root.bind("d", lambda event: Askuser())
                                    
                                    button3 = Button(root, text='INCORRECT', fg="blue",command=Incorrect,font=('Helvetica', '14'))
                                    button3.place(x=720,y=350)
                                    root.bind("i", lambda event: Incorrect())


                                        # welcome="Welcome to Infogen-labs\n"
                                        # tk10=Label(root,text=welcome,fg = "blue",font = "Times 18 bold")
                                        # tk10.place(x=650, y=180)
                                        # canvas1.after(1,DisplayWelcome)
                                        # button5 = Button(root, text='Refresh', fg="blue",command=Refresh,font=('Helvetica', '16'))
                                        # button5.place(x=730,y=400)
                                        # canvas1.create_window(130, 330, window=button4)
                                        # root.bind("r", lambda event: Run())


                                    # button4 = Button(root, text='Refresh', fg="black",command=Refresh,font=('Helvetica', '14'))
                                    # button4.place(x=735,y=400)
                                    # root.bind("r", lambda event: Refresh())
                                    disp4= "Check-in Press 'a'"
                                    tk7=Label(root,text=disp4,fg ="blue",bg='white',font = "Arial 12 italic")
                                    tk7.place(x=650, y=460)

                                    disp5= "Check-out Press 'd'"
                                    tk7=Label(root,text=disp5,fg ="blue",bg='white',font = "Arial 12 italic")
                                    tk7.place(x=650, y=480)

                                    disp2= "Incorrect recognition Press 'i' "
                                    tk6=Label(root,text=disp2,fg ="blue",bg='white',font = "Arial 12 italic")
                                    tk6.place(x=650, y=500)

                                    tk2.update()
                                    tk4.update()
                                    tk2.update_idletasks()
                                    tk4.update_idletasks()
                               
                                logging.info('Attendance registered !')
                            except Exception as e:
                                logging.info(e)
                            if matching_id is None:
                                matching_id = 'Unknown'
                                print("Couldn't find match.")

                                # hi='Hi,'
                                # tk0=Label(root,text=hi,fg = "red",bg='white',font = "Times 18 bold")
                                # tk0.place(x=650, y=80)
                                # # tk0.after(1)

                                
                                # tk1=Label(root,text='            ',fg = "red",bg='white',font = "Times 18 bold")
                                # tk1.place(x=650, y=110)

                                # # # global tk2
                                # # # tk2.delete("all")
                                # tk21=Label(root,text='                      ',fg = "white",bg="white",font = "Times 18 italic bold")
                                # tk21.place(x=720, y=110)

                                # # tk2=Label(root,text='Unknown',fg = "green",bg="white",font = "Times 18 italic bold")
                                # # tk2.place(x=720, y=110)

                                # # # global tk3
                                # # # tk3.delete("all")
                                # tk3=Label(root,text='              ',fg = "red",bg='white',font = "Times 18 bold")
                                # tk3.place(x=650, y=140)

                                # tk41=Label(root, text='                 ',fg = "white",bg='white',font = "Times 18 italic bold")
                                # tk41.place(x=740, y=140)

                                # # tk4=Label(root, text='Unknown',fg = "green",bg='white',font = "Times 18 italic bold")
                                # # tk4.place(x=740, y=140)

                                # welcome="Welcome to Infogen-labs\n"
                                # # user_text= welcome+'Match not found !'

                                # tk51=Label(root,text='                         \n                             ',fg = "white",bg='white',font = "Times 18 bold")
                                # tk51.place(x=650, y=180)

                                # tk5=Label(root,text=welcome,fg = "red",bg='white',font = "Times 18 bold")
                                # tk5.place(x=650, y=180)

                                # button1 = Button(root, text='CHECK-IN',fg="green",font=('Helvetica', '14'))
                                # button1.place(x=730,y=250)
                                # # canvas1.create_window(140, 230, window=button1)
                                # # root.bind("a", lambda event: Telluser())

                                # tk510=Label(root,text='                         ',fg = "white",bg='white',font = "Times 18 bold")
                                # tk510.place(x=720,y=300)
                                # canvas1.create_window(140, 280, window=button2)
                                # root.bind("d", lambda event: Askuser())
                                
                                # button3 = Button(root, text='INCORRECT', fg="blue",command=Incorrect,font=('Helvetica', '14'))
                                # button3.place(x=720,y=350)
                                # root.bind("i", lambda event: Incorrect())

                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    img = PIL.Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)
                    lmain.after(5, show_frame)

                show_frame()
                #GUI window icon
                root.iconbitmap(r'logo/igl.ico')
                # root.after(10,Displayname())
                root.mainloop()
                # cv2.imshow('Face Recognition',frame)
                # k=cv2.waitKey(1)
                # if k == ord("q"):
                    # break
                # print('Rate :>>>',1/(time.time() - start))
                # timeCheck = time.time()

            # cv2.destroyAllWindows()

if __name__ == '__main__':
    dt = datetime.now()
    now=dt.strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig( filename=r'Logs/FR_'+str(now)+'.log',level=logging.DEBUG,format = '%(asctime)s  %(levelname)-10s %(processName)s  %(name)s %(message)s %(lineno)d %(funcName)s')
    #logging.disabled = True
    logging.disable(logging.NOTSET) #to enable logging
    #logging.disable(sys.maxsize)   #to disable logging
    logging.info('parsing arguments')
    try:
        parser = argparse.ArgumentParser() #To access the arguments from the command line
        parser.add_argument('model', type=str, help='Path to facenet model', default=r'E:\\Ajinkya\\assets\\Infogen\\FR_Infogen_v3\\server\\Facenetmodel18\\model18.pb')
        # parser.add_argument('-t', '--threshold', type=float,help='Distance threshold defining an id match', default=0.7)# setting up the default threshod value 
    except Exception as e:
        logging.exception(e)

    main(parser.parse_args()) #parse_args() will typically be called with no arguments, and the 'ArgumentParser' will automatically determine the command-line arguments