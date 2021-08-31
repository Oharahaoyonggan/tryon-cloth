import os
import cv2
import numpy as np
import shutil
from PIL import Image
from flask import Flask 
from flask import request
from flask import render_template
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation

app =Flask(__name__)
UPLOAD_CLOTH = "dataset/test_clothes"
CLOTH_EDGE= "dataset/test_edge"
PERSON_IMG= "dataset/test_img"
CLOTH_NAME="CLOTH.jpg"
PERSON_NAME="PERSON.jpg"

def whiteBackGrounf(img):
    BG_COLOR = (192, 192, 192) # gray
    MASK_COLOR = (255, 255, 255) # white
    
    with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
        
        # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
        results = selfie_segmentation.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(img.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(img.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, fg_image, bg_image)
        
        results = selfie_segmentation.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        white=img.copy()
        white[:] = 255
        blurred_image =  cv2.GaussianBlur(img,(55,55),0)
        
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, img, white)
        return output_image
def clothMasking(Cimage_location):
    shutil.copyfile(Cimage_location, 'U-2-Net-master/images/CLOTH.jpg')
    os.system('python -W ignore u2net_test.py')
    os.replace('U-2-Net-master/resultsCLOTH.png','dataset/test_edge/CLOTH.jpg')
    

# @app.route('/')
# def index():
#     return render_template("index.html")


@app.route('/', methods=['POST',"GET"])
def upload_predict():
    if request.method=="POST":
        Pimage_file=request.files["image"] #get the person image from html
        Cimage_file=request.files["Cimage"] #get the cloth image from html
        
        if Pimage_file and Cimage_file:  #check is they upload both image or not
            #to save the person image
            Pimage_location= os.path.join(
                PERSON_IMG,
                PERSON_NAME
            )
            Pimage_file.save(Pimage_location)
            
            #to save the cloth image
            Cimage_location= os.path.join(
                UPLOAD_CLOTH,
                CLOTH_NAME
            )
            Cimage_file.save(Cimage_location)
            
            #generate the cloth edge image and save it
            clothMasking(Cimage_location)
            # img = cv2.imread(Cimage_location)
            # OLD_IMG = img.copy()
            # mask = np.zeros(img.shape[:2], np.uint8)
            # SIZE = (1, 65)
            # bgdModle = np.zeros(SIZE, np.float64)

            # fgdModle = np.zeros(SIZE, np.float64)
            # rect = (1, 1, img.shape[1], img.shape[0])
            # cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

            # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            # mask2[mask2 == 1] = 255
            # cv2.imwrite(os.path.join(
            #     CLOTH_EDGE,
            #     CLOTH_NAME
            # ), mask2)
            
            #change person background
            img=cv2.imread(Pimage_location)
            output_image=whiteBackGrounf(img)
            cv2.imwrite(Pimage_location,output_image)

            #change person size
            image = Image.open(Pimage_location)
            new_image = image.resize((192, 256))
            new_image.save(Pimage_location)
            #change cloth size
            image = Image.open(Cimage_location)
            new_image = image.resize((192, 256))
            new_image.save(Cimage_location)
            #change edge size
            image = Image.open(os.path.join(
                CLOTH_EDGE,
                CLOTH_NAME
            ))
            new_image = image.resize((192, 256))
            new_image.save(os.path.join(
                CLOTH_EDGE,
                CLOTH_NAME
            ))
            #write in demo.txt
            f=open("demo.txt","w")
            f.write(PERSON_NAME+" "+CLOTH_NAME)
            f.close()
            #run the virtual try on
            os.system('python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0 --nThreads 0')
            #return render image
            os.replace("results/demo/PFAFN/0.jpg", "static/0.jpg")
            return render_template("result.html", user_image = 'static/0.jpg')
            #return render_template("index.html")


    return render_template("index.html")



if __name__ =="__main__":
    app.run( debug=True)