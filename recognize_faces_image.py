import random
from pdb import Pdb

import face_recognition
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] Loading encodings..")
data = pickle.loads(open(args["encodings"], "rb").read())

image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("[INFO] Recognizing faces..")
boxes = face_recognition.face_locations(rgb, model="detection method")
encodings = face_recognition.face_encodings(rgb, boxes)
names = []

for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"
    # find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
    if True in matches:
        # we need to determine the indexes of where these True  values are in matches, we construct a simple list of matchedIdxs
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        # determine the recognized face with the largest number of votes (note: in the event of an unlikely tie Python will select first entry in the dictionary)
        name = max(counts, key=counts.get)
    names.append(name)
     # (Pdb) matchedIdxs
     # (Pdb) counts

for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
# show the output image
cv2.imshow("Image", image)
k = cv2.waitKey(0)

if k == ord('s'):  # wait for 's' key to save and exit
    a = random.randint(0, 99)
    cv2.imwrite(f"Sample_{a}.jpg", image)
    cv2.destroyAllWindows()
