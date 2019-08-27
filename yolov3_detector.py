# Author: Ritika Gupta
# Email: ritikagupta1998@gmail.com

# USAGE
# python yolov3_detector.py  --image media/test.jpg --yolo config/

# Import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from deep_text import detect_text

# In this block of code, we take all the external input and add them to our parser object
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True, help="Path of the input image")
parser.add_argument("-y", "--yolo", required = True, help="Directory containing YOLOv3 weights trained on number plate recogniniton")
parser.add_argument("-c", "--confidence", required = False, type=float, default=0.5, help="Threshold for confidence")
parser.add_argument("-t", "--threshold", required = False, type=float, default=0.3, help="Non-maximum supression threshold")

# These parsers are for DeepText and currently we do not require to input them from terminal.
# We have kept it default (same as the original) for this competition
parser.add_argument('--image_folder', default = 'yolov3_detected_cropped/', required=False, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model',default = 'config/deep_text_weights_ResNetTPS.pth', required=False, help="path to saved_model to evaluation")

# Data processing arguments 
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')

# DeepText model architecture aruguments
parser.add_argument('--Transformation', type=str, default= 'TPS', required=False, help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str,default = 'ResNet', required=False, help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default = 'BiLSTM', required=False, help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str,default = 'Attn', required=False, help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
#W We convert Namespace to Dictionary using vars
args = parser.parse_args()

# Here we load the classes. In this case, we only have to detect number plate and thus have only one label in plate.names file
labelsPath = os.path.sep.join([args.yolo, "plate.names"])
LABELS = open(labelsPath).read().strip().split("\n")

color = (0, 255, 0)

# In the following two lines, we load neural networks weight trained on vehicle plate dataset and model architecture
weightsPath = os.path.sep.join([args.yolo, "yolov3_plate_detector.weights"])
configPath = os.path.sep.join([args.yolo, "yolov3_plate.cfg"])

# In this line we use readFromDarknet method applied on cv2.dnn to create the network from weights and configuration file
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Here we oad the input image from the parsed arguments and extract its height and width
image = cv2.imread(args.image)
(H, W) = image.shape[:2]

# In this case, we get the names of ouput layers of the built network
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Int the following codes, we convert out image to a blob, resize, rescale and swap color channels
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Now we set input blob to the network for forward pass and calculate its outputs
net.setInput(blob)
layerOutputs = net.forward(ln)


# We initialize three lists for detected bounding boxes, confidences, and class IDs, respectively
boxes = []
confidences = []
classIDs = []

# We loop over each of the layer outputs
for output in layerOutputs:
	# Once we get the ouput, we loop over each of the detections in the outpur
	for detection in output:
		# Here we extract the class ID which is only 0 in this case as we have only one class 'number plate' and confidence of the detected object
		
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# Next, we filter out weak predictions if its probability is < 0.5
		if confidence > args.confidence:
			# Here we get the output of the bounding boxes coordinates
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# We convert the bounding box coordinates to xmin, ymin, and xmax, ymax
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# Append the boxes coordinates, confidences, and classIDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# Here we store only strong detection by applying Non-maxima supression
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.threshold)

# Name of the file to save the cropped image
croppedName = 'yolov3_detected_cropped/testImageCroppedROI'
# ensure at least one detection exists
if len(idxs) > 0:
	# Here we loop over the indexes preserved after NMS threshold
	for i in idxs.flatten():
		# Now we extract the bounding box coordinates in X, Y, W and H format
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# Now we derive the only those regions that is detected by our model i.e. the number plate
		roi = image[y:y+h, x:x+w]
		cv2.imwrite('{}.png'.format(croppedName), roi)

		# Next we pass this cropped image to the DeepText instance for character recognition on the number plate
		textOutput = detect_text(args)
		print(textOutput)

		# Here we draw the rectangle, the class name and the recognized text
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
		cv2.putText(image, textOutput.upper(), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0,0,255), 2)

# Finally we save the image to the current directory
cv2.imwrite("detected_plate_with_text.png", image)
