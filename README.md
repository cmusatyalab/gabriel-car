# AutoAssemblyAssistant (AAA)
#### Wearable Cognitive Assistant for Model Car Assembly
* [Overview Video](https://www.youtube.com/watch?v=4bHLLkaQ5V4)
* [Poster](docs/poster.pdf)
* [Technical Report](https://docs.google.com/document/d/19uSh-1quI0Lwa_Gkl5fhDjU3kvZnpxR0g08xO-aqA4k/edit?usp=sharing)
* [Demo Video](https://www.youtube.com/watch?v=OS9efSw-fM8)
* [Handoff Doc for Developers](handoff.md)

AAA is a wearable cognitive assistant which guides a user through the steps of building a model car, prompting just-in-time alerts and corrections when a mistake is made. AAA is built off the [Gabriel platform](https://github.com/cmusatyalab/gabriel) and makes use of the [TPOD object detection tool](https://github.com/cmusatyalab/tpod).

##### You are unlikely to be able to run AAA without working with the [CMU Satya Lab](https://github.com/cmusatyalab).
##### To run AAA, you will need to setup:
* A cloudlet server running Gabriel
* Trained TPOD classifiers
* Our guidance images and videos
* A client on an Android device (smartphone/smart glasses)

#### Configuring AAA
1. Setup a clean install of Python 2.7 (probably a virtual environment)
2. `pip install -r requirements.txt`
3. Download [Gabriel](https://github.com/cmusatyalab/gabriel) and `cd <gabriel repo>/server/`
4. `python setup.py install`
5. Download this repo and `cd <this repo>`
6. Edit `start_demo.sh` with your server's parameters
7. Open the repo in Android Studio, build the client, and deploy it to your Android device

#### How to run AAA (after configuration)
1. Navigate to this repo:
`cd <root of repo>`
2. Start the Gabriel control server, Gabriel ucomm server, and video resource server:
`sudo ./start_demo.sh`
3. Start the AAA Proxy server:
`python car.py`
4. Connect the Android client to the Gabriel control server (use the control server's IP)

#### Future Work
We've started some work on a state-based model where changes to the car are tracked instead of having steps. The demo we have covers inserting the green and gold washers into the frame. It can detect which orientation the frame is in and thus, which hole a washer is put in or removed regardless of the orientation of the frame. The demo is located under the `state_based` branch.
