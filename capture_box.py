import cv2
import os

def detect(folder_path):
    face_cascade = cv2.CascadeClassifier(r'C:\Users\Nil\VscodeProjects\python\Computer-Vision-Crash-Course-\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(r'C:\Users\Nil\VscodeProjects\python\Computer-Vision-Crash-Course-\haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    i = 0  # Counter for saving images

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))

            #for (ex, ey, ew, eh) in eyes:
                #cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Save the detected face region with a descriptive filename:
            face_filename = f"detected_face_{i}.jpg"
            face_path = os.path.join(folder_path, face_filename)
            cv2.imwrite(face_path, roi_gray)
            i += 1

            # Display the frame with detected faces and eye rectangles:
            cv2.imshow("camera", frame)

        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the folder path where you want to save detected faces:
    folder_path = r'C:\Users\Nil\VscodeProjects\python\Computer-Vision-Crash-Course-\data\captured'  # Replace with your actual folder path
    detect(folder_path)
