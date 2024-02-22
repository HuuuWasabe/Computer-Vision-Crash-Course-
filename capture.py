import cv2
import os
import time

# Set the duration (in seconds) and the number of images to capture
duration = 10
num_images = 100

# Create a new directory to store the images
output_directory = "captured_images"
os.makedirs(output_directory, exist_ok=True)

# Open the camera
cap = cv2.VideoCapture(0)

# Set the frame width and height
cap.set(3, 640)
cap.set(4, 480)

# Start capturing images
start_time = time.time()

for i in range(num_images):
    # Read a frame from the camera
    ret, frame = cap.read()

    # Save the frame as an image file
    image_filename = os.path.join(output_directory, f"image_{i + 1}.jpg")
    cv2.imwrite(image_filename, frame)

    # Display the captured image (optional)
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(1)

    # Check if the specified duration has elapsed
    if time.time() - start_time > duration:
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"{num_images} images captured in {duration} seconds. Images are saved in '{output_directory}'.")
