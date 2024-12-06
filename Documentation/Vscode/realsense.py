import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Données
x_data = np.array([340, 280, 250, 115, 78, 67])
y_data = np.array([14, 17, 20, 40, 60, 72])
x_vitesse = np.array([72,20])
y_vitesse = np.array([20,0])

# Ajuster la fonction polynomiale
params, covariance = curve_fit(lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e, x_data, y_data)
params2, covariance2 = curve_fit(lambda x, a, b: a * x + b, x_vitesse, y_vitesse)

# Générer des points pour la courbe ajustée
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = (lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e)(x_fit, *params)
x_fit2 = np.linspace(min(x_vitesse), max(x_vitesse), 100)
y_fit2 = (lambda x, a, b: a * x + b)(x_fit2, *params2)

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)

# Start streaming
pipeline.start(config)

stop_cascade = cv2.CascadeClassifier(r'C:\Users\victo\OneDrive\Bureau\PING\Autonomous_Car_Equipe_1\RPI3\Detection_Panneau_STOP\Stop_classificateur.xml')

# Initialize VideoWriter for saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose a different codec if needed
output_video = cv2.VideoWriter('C:/Users/victo/OneDrive/Bureau/PING/Autonomous_Car_Equipe_1/RPI3/Detection_Panneau_STOP/output_video.avi', fourcc, 20.0, (640, 360))

def stop_sign_detection(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    panneaux = stop_cascade.detectMultiScale(img, scaleFactor=1.01, minNeighbors=10)
    detected_signs = []

    for (x, y, w, h) in panneaux:
        detected_signs.append({'x': x, 'y': y, 'width': w, 'height': h})

    return detected_signs

try:
    while True:
        start_time = cv2.getTickCount()  # Record the start time
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        # Convert color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        color_image2 = color_image[0:280, 400:]
        #color_image = cv2.resize(color_image,(320,240))
        detected_signs = stop_sign_detection(color_image2)

        for sign in detected_signs:
            x, y, w, h = sign['x'], sign['y'], sign['width'], sign['height']
            cv2.rectangle(color_image2, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            distance = (lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e)(w, *params)
            speed = (lambda x, a, b: a * x + b)(distance, *params2)
            print(f"Detected stop sign at ({x}, {y}), size: {w} x {h}, distance: {distance} cm, speed: {speed} km/h")
        #images = np.hstack((color_image, color_image2))
        # Write the frame with detected signs to the output video
        output_video.write(color_image)
        # Display the resulting frame 
        # images = np.hstack((color_image,depth_colormap))
        cv2.imshow('RealSense Camera', color_image2)
        end_time = cv2.getTickCount()  # Record the end time
        elapsed_time = (end_time - start_time) / cv2.getTickFrequency()  # Calculate elapsed time in seconds
        fps = 1.0 / elapsed_time  # Calculate frames per second
        print(f"FPS: {fps}")

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    # Release the VideoWriter
    output_video.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
