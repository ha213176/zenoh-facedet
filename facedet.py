import argparse
import time
import cv2
import zenoh
import numpy as np
import json

parser = argparse.ArgumentParser(
    prog='zdisplay',
    description='zenoh video display example')
parser.add_argument('-m', '--mode', type=str, choices=['peer', 'client'],
                    help='The zenoh session mode.')
parser.add_argument('-e', '--connect', type=str, metavar='ENDPOINT', action='append',
                    help='zenoh endpoints to listen on.')
parser.add_argument('-l', '--listen', type=str, metavar='ENDPOINT', action='append',
                    help='zenoh endpoints to listen on.')
parser.add_argument('-d', '--delay', type=float, default=0.05,
                    help='delay between each frame in seconds')
parser.add_argument('-k', '--key', type=str, default='demo/zcam',
                    help='key expression')
parser.add_argument('-c', '--config', type=str, metavar='FILE',
                    help='A zenoh configuration file.')

args = parser.parse_args()

conf = zenoh.Config.from_file(args.config) if args.config is not None else zenoh.Config()


if args.mode is not None:
    conf.insert_json5(zenoh.config.MODE_KEY, json.dumps(args.mode))
if args.connect is not None:
    conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(args.connect))
if args.listen is not None:
    conf.insert_json5(zenoh.config.LISTEN_KEY, json.dumps(args.listen))

cams = {}

def frames_listener(sample):
    npImage = np.frombuffer(bytes(sample.value.payload), dtype=np.uint8)
    matImage = cv2.imdecode(npImage, 1)
    cams[sample.key_expr] = matImage


print('[INFO] Open zenoh session...')
zenoh.init_logger()
z = zenoh.open(conf)

sub = z.declare_subscriber(args.key, frames_listener)

# load AI model
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (500,  375))


while True:
    for cam in list(cams):
        img = cams[cam]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        print("facedet done!")
        out.write(img)
        # cv2.imwrite("test.png", img)
        # cv2.imshow(str(cam), img)
        
    key = cv2.waitKey(1) & 0xFF
    time.sleep(0.05)
