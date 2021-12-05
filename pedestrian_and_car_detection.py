import cv2
video=cv2.VideoCapture('Pedestrians Compilation.mp4')
'''
video=cv2.VideoCapture('Cars Compilation.mp4')
'''
car_file='cars_xml.xml'
car_tracker=cv2.CascadeClassifier(car_file)
pedestrian_file='pedestrian_xml.xml'
pedestrian_tracker=cv2.CascadeClassifier(pedestrian_file)
while True:
    (read_successful, frame) =video.read()
    if read_successful:
        grayscaled_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    cars=car_tracker.detectMultiScale(grayscaled_frame)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+2), (x+w,y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    
    pedestrians=pedestrian_tracker.detectMultiScale(grayscaled_frame)
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 2)

    cv2.imshow('Pedestrian and Car detector', frame)
    cv2.waitKey(1)
video.release()