import cv2
import numpy as np
import pytesseract
import pyzbar.pyzbar as pyzbar

# Load the image
image = cv2.imread('/home/jakapong/example_code/test3.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to read image.")
    exit(1)

# Copy of the original image for final annotation
output_image = image.copy()

# Step 1: Use OCR to detect numbers
# Preprocess image for OCR
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dilated_image = cv2.dilate(thresh_image, np.ones((2, 2), np.uint8), iterations=1)

# Use Tesseract OCR to extract numbers
ocr_config = "--psm 6"  # Assume a single uniform block of text
ocr_result = pytesseract.image_to_string(dilated_image, config=ocr_config)

# Extract and print numbers from the detected text
numbers = [int(s) for s in ocr_result.split() if s.isdigit()]
print(f"Detected numbers: {numbers}")

# Step 2: Detect QR code
decoded_objects = pyzbar.decode(image)
qr_data = None
qr_code_area = None  # Variable to store QR code area

for obj in decoded_objects:
    qr_data = obj.data.decode("utf-8")
    print(f"Detected QR code: {qr_data}")
    # Draw a rectangle around the QR code
    points = obj.polygon
    if len(points) == 4:
        qr_code_area = cv2.boundingRect(np.array(points))  # Get bounding box of QR code
        cv2.polylines(output_image, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

# Step 3: Detect the blue rectangle
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize detected circles, colors, and positions
detected_circles = []
detected_colors = []

# Step 4 & 5: Detect circles and colors inside the valid blue rectangle
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if h > 50 and w > 50:  # Adjust based on the expected size of the blue rectangle
        if qr_code_area:
            # Exclude areas overlapping with QR code
            qr_x, qr_y, qr_w, qr_h = qr_code_area
            if not (x < qr_x + qr_w and x + w > qr_x and y < qr_y + qr_h and y + h > qr_y):
                roi = image[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Detect circles using HoughCircles
                circles = cv2.HoughCircles(
                    gray_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=40,  # Minimum distance between circle centers
                    param1=50,
                    param2=30,   # Adjusted to detect circles more reliably
                    minRadius=10,
                    maxRadius=50
                )

                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    circles = sorted(circles, key=lambda c: c[1])  # Sort by vertical position
                    circles = circles[:3]  # Only keep the top 3 circles

                    for (cx, cy, r) in circles:
                        # Ensure the circle is inside the blue rectangle
                        if y + cy > y and y + cy < y + h and x + cx > x and x + cx < x + w:
                            # Get the color inside the circle
                            circle_mask = np.zeros_like(roi)
                            cv2.circle(circle_mask, (cx, cy), r, (255, 255, 255), -1)
                            mean_color = cv2.mean(roi, mask=circle_mask[:, :, 0])

                            color_name = "Unknown"
                            if mean_color[2] > 150 and mean_color[1] < 100:  # Red
                                color_name = "Red"
                            elif mean_color[0] > 150 and mean_color[1] > 150 and mean_color[2] > 150:  # White
                                color_name = "White"

                            detected_circles.append((x + cx, y + cy, r, color_name))

                            # Draw circle and color name on output image
                            cv2.circle(output_image, (x + cx, y + cy), r, (0, 255, 0), 2)
                            cv2.putText(output_image, color_name, (x + cx - 20, y + cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Step 6: Align numbers to circles based on x-axis
if len(detected_circles) == 3 and len(numbers) == 3:
    detected_circles = sorted(detected_circles, key=lambda c: c[0])  # Sort circles by x-axis position
    numbers = sorted(numbers)  # Sort numbers to match with sorted circles
    for i, (circle_x, circle_y, radius, color) in enumerate(detected_circles):
        number = numbers[i]
        # Annotate the number with the circle
        cv2.putText(output_image, f"{number}", (circle_x, circle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"Circle {i+1} (Color: {color}) has number: {number}")

# Annotate QR code on the output image
if qr_data:
    cv2.putText(output_image, f"QR: {qr_data}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the final annotated image
cv2.imshow("Annotated Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
