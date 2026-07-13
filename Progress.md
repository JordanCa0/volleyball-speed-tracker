### V1 - Player Detector 
- First started just by creating the detector to detect the player / ball 
- Can scan film, live footage, or even stream from the phone camera 

Method: 

`detect.py ` is a frame-by-frame video pipeline 
- Reads frames from a video file or camera and runs each through a pretrained YOLOv8 neural network to decipher people and balls, it then draws a laveled box on the frame, and saves it. 
- Currenly no tracking or speed math yet. 

1. The model and classes 
- script uses yolov11n.py which is a YOLOv11 "nano" model that is pretrained on the COCO dataset. It has 80 everyday object categories. Although it was never trained specifially on volleyball it knows class 0 ("person", relabeled as "player") and class 32 ("sports ball", relabelled as "ball")

2. CLI Arguments
`--source` accepts either a file path or camera index
`--model` accepts the neural network model used for classification 
`--conf` sets the minimum confidence (default 25%) 
`--save` writes an annotated MP4
`--no-display` skips the live window

3. The main loop
- `cap.read()` grabs the next frame as a NumPy array (loop ends when the file runs out or the camera fails)
- `model.predict(frame, classes = [PERSON, SPORTS_BALL], conf=args.conf)` runs one forward pass of the NN. The classes filter tells Ultralyrics to discord the other categories, and `conf= drops` weak detections
- `draw detections` loops over each detected box. Every box carries a class id, a confidence, and `xyxy` corner coordinate, it then draws a rectangle and a bar above the box with the label and confidence as a percentage 

### V1.1 - Improvements on Ball Detection 
- There were some issues with ball detection when testing. Some fixes were a function to press b during running allowing the user to draw a box around the ball and then the program uses hsv colour detection to track the ball. Also, added a motion mask option so that static objects don't get mistaken for the ball. 