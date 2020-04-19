from tkinter import *
import os
import cv2 #OpenCV - Open Computer Vision
import numpy as np
import faceDetectorFunctions as fr



root = Tk()
root.title("Live Facial Recognition")
root.geometry("400x400")

print(os.listdir('trainingimages'))


label = Label(root, text="Enter your name")
label.grid(row=0,column=0)

e = Entry(root)
e.grid(row=0,column=1)

def getImageData():
	if(len(e.get()) == 0):
		return
	
	name = e.get()
	newDir = str(len(os.listdir('trainingimages')))
	os.chdir('trainingimages')
	os.mkdir(newDir)
	os.chdir(newDir)

	#Train live image
	cap = cv2.VideoCapture(0)
	count = 0
	
	while True:
		ret, test_img = cap.read()
		print("video image captured")
		faces_detected, gray_img = fr.faceDetection(test_img)

		if not ret:
			print("No camera detected")
			continue
		elif len(faces_detected) < 1:
			print("No face detected")
			continue

		cv2.imwrite("frame%d.jpg" % count, test_img) # Save frame as JPG
		count += 1
		resized_img = cv2.resize(test_img, (1280,720))
		cv2.imshow('Reading Face', resized_img)

		if cv2.waitKey(10) == ord('q'):
			break
		
	
	cap.release()
	cv2.destroyAllWindows()
	os.chdir('../..')
	with open ("labels.csv", mode="a", encoding="utf-8") as labelsFile:
		labelsFile.write(newDir+","+name+'\n')

	faces, faceID = fr.labels_for_training_data("trainingimages")
	face_recognizer = fr.train_classifier(faces,faceID)
	face_recognizer.save('trainingData.yml')
	print("All images fully trained!")
	

def testRecognition():
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.read('trainingData.yml')

	name = {}

	with open("labels.csv", encoding="utf-8") as labelsData:
		while True:
			line = labelsData.readline()
			if not line:
				break

			faceTag,label = line.split(',')
			name[faceTag] = label

	print(name)

	cap = cv2.VideoCapture(0)
	while True:
		ret, test_img = cap.read()
		faces_detected,gray_img = fr.faceDetection(test_img)

		for (x,y,w,h) in faces_detected:
			cv2.rectangle(test_img, (x,y), (x+w,y+h),(255,0,0),thickness=7)

		resized_img = cv2.resize(test_img, (1280,720))
		cv2.imshow('face detection', resized_img)
		cv2.waitKey(10)

		for face in faces_detected:
			(x,y,w,h) = face
			roi_gray = gray_img[y:y+w, x:x+h]
			label, confidence = face_recognizer.predict(roi_gray)
			print("confidence:", confidence)
			print("label:", label)
			fr.draw_rect(test_img, face)
			predicted_name = name[str(label)].upper()
			if confidence < 80:
				fr.put_text(test_img, predicted_name, x, y)


		resized_img = cv2.resize(test_img, (1280,720))
		cv2.imshow('face recognition', resized_img)
		if cv2.waitKey(10) == ord('q'):
			break


	cap.release()
	cv2.destroyAllWindows()


trainImageBtn = Button(root, text="Add New Image Data", 
	command=getImageData)
trainImageBtn.grid(row=0,column=2)

testImageBtn = Button(root, text="Live Image Recognition", 
	command=testRecognition)
testImageBtn.grid(row=2,column=0)



root.resizable(False,False)
root.mainloop()