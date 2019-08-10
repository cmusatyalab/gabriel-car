# AAA Handoff Document for Developers
Information to ease development of this repo as well as future Gabriel cognitive assistants. You will still need to dig through the code—this just serves as a high level summary that may make reading the code easier. Details on how to run AAA are not here, instead in README.md at the root of the repo.

### Gabriel System Overview
Gabriel is a platform that handles communication between a smart device client and a cloudlet server. The client will stream raw data (in AAA's case, a camera feed) to the server for processing. The reason for having two devices is because the server has higher processing power and can use more resource intensive methods (ML-based object detection) in real-time.

The general workflow can be summarized as:
1. Client sends its camera stream to the server, frame-by-frame.
2. Server processes frame to determine user progress. In AAA, this is detecting objects of interest using ML-based computer vision.
3. Server sends back an appropriate response based on processing.
4. Client plays response to the user, in the form of audio-visual feedback. In AAA, this is a combination of image, video, speech, and drawn graphics over the camera feed.

The client is a fairly simple Android application that simply streams its camera feed and can read responses from the server (via HTTP requests). We made extensive changes to the client for additional features not found in other cognitive assistants, such as drawing boxes around detected objects on the client side and sending session IDs.

The server is relatively much more complex. The server component is made up of 3 or more services each with their own IP address: the control server, the ucomm server, the proxy server, and any additional processing services.

Quite simply, the control and ucomm servers handle the connection process of sending and receiving messages. Note that they are not reading and crafting messages, these services are merely the messenger. For our purposes (and more likely than not, yours), we did not have to change anything in the control and ucomm servers, nor did we have to distinguish the two services. The control and ucomm servers are boilerplate code for all Gabriel cognitive assistants.

The proxy server is what actually processes the camera feed i.e. reads incoming messages and generates the appropriate response. More likely than not, all of your changes to the server will reside here. The proxy server is what distinguishes the Gabriel cognitive assistants by their tasks (LEGO, model car, RibLoc).

Within the proxy server are calls to a separate processing services. In our case, this was the object detection classifier service exported from [TPOD](https://github.com/cmusatyalab/tpod) and a video resource server where we hosted all our guidance videos.

### Summary of Relevant Files
These are the main files you'd update to introduce changes to AAA or build a new cognitive assistant, versus boilerplate code for setting up the control and ucomm server.
##### Client
`legacy_client/app/src/main/res/layout/activity_main.xml`: Layout of UI (size of camera feed, size of guidance images, etc.)
`legacy_client/app/src/main/java/edu/cmu/cs/gabrielclient/GabrielClientActivity.java`: Handles reading a message from the server (reading speech, playing video, drawing bounding boxes client-side)

##### Server
`start_demo.sh`: Starts the Gabriel control server, Gabriel ucomm server, and video resource server. Probably won't have to edit this except for changing configurations detailed in `README.md`
`car.py`: Highest level wrapper for the proxy server. You run this file to start the proxy
`car_task.py`: Contains everything from receiving the frame to generating the appropriate response, including running object detection on the frame. Bulk of the code is here, found in `Task.get_instruction()`
`object_detection.py`: Various functions that handle the sending of the raw frame to the TPOD classifier service, as well as some processing of its results e.g. handling overlapping bounding boxes with the same label. This also handles the spinning up of the TPOD services, when using `car.py`
`car_stream.py`: Debug proxy server to just run a camera feed and show object detections. You need to spin up the TPOD classifier service yourself

### Object Detection via TPOD
TPOD is an all-encompassing tool for object detection. It handles labeling of training videos, training the model, and exporting a Docker container classifier. Running the container will start up a service that you communicate with via HTTP requests. You send images and it'll respond with detections in the form of labeled bounding boxes.

Details of how to use TPOD are in its [repo]( https://github.com/cmusatyalab/tpod).

One key detail specific to AAA is that we actually use multiple different containers each that detect different things. This is markedly different from other cognitive assistants, which have one big classifier for everything. We could not do that because of a TPOD bug that limited each classifier to ~50 videos.

During use, AAA will start up the corresponding classifier service based on what you want to detect. We could not spin them all up at the same time because of limitations with Docker and our machine. Implementation details are in `object_detection.py`

All our trained classifiers (Docker containers) are on the machine at cloudlet011.elijah.cs.cmu.edu i.e. you cannot run AAA on any other machine without training your own classifiers (which you shouldn't do for AAA).
