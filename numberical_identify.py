import cv2
import numpy as np
import pytesseract

# Load the input image
image = cv2.imread('/home/drobot/Documents/hemanth/amazon ps/WhatsApp Image 2024-09-13 at 5.24.28 PM.jpeg')
orig = image.copy()
(H, W) = image.shape[:2]

# Set the new width and height for resizing to 320x320, as required by EAST
newW, newH = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# Resize the image for EAST
image = cv2.resize(image, (newW, newH))

# Load the pre-trained EAST text detector
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# Define the two output layer names for the EAST detector model
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",  # The output probability score map
    "feature_fusion/concat_3"         # The output geometry map
]

# Construct a blob from the image and pass it through the network
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# Function to decode predictions (bounding boxes) from the EAST output
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    return (rects, confidences)

# Decode the predictions with a minimum confidence threshold of 0.5
rects, confidences = decode_predictions(scores, geometry, min_confidence=0.5)

# Apply non-maxima suppression to remove overlapping boxes
boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

# Loop over the bounding boxes and crop text regions
for i in range(len(boxes)):
    (startX, startY, endX, endY) = rects[boxes[i][0]]

    # Scale the bounding boxes back to the original image size
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Extract the ROI from the original image
    roi = orig[startY:endY, startX:endX]

    # Use OCR to extract text from the ROI
    text = pytesseract.image_to_string(roi, config='--psm 7 digits')

    # Check if the extracted text contains numerical data
    if text.isdigit():
        print(f"Extracted Text: {text}")
        # Optionally save the cropped image
        cv2.imwrite(f'cropped_region_{startX}_{startY}.png', roi)

# Display the output image with bounding boxes (optional)
for i in range(len(boxes)):
    (startX, startY, endX, endY) = rects[boxes[i][0]]
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
