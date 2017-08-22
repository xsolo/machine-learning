import cv2

class VideoCamera(object):
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('/Users/danila/workspace//opencv//data/haarcascades/haarcascade_frontalface_default.xml')

        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.fishface = cv2.face.createFisherFaceRecognizer() 
        self.fishface.load("fishface.model")
        self.emo = {10:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        font = cv2.FONT_HERSHEY_SIMPLEX


        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            e, conf = self.fishface.predict(cv2.resize(gray, (48, 48)) )
            cv2.putText(frame,self.emo.get(e) + " {:.1f}".format(conf),(x,y+h), font, 1,(255,255,255),1)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()