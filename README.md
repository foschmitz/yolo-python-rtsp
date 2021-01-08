# Object Detection with Yolo, OpenCV and Python via Real Time Streaming Protocol (RTSP)

Object detection using deep learning with Yolo, OpenCV and Python via Real Time Streaming Protocol (`RTSP`)

Recognized objects are stored in date seperated in folders per class for further training or face recognition.

OpenCV `dnn` module supports running inference on pre-trained deep learning models from popular frameworks like Caffe, Torch and TensorFlow.

When it comes to object detection, popular detection frameworks are
 * YOLO
 * SSD
 * Faster R-CNN

 Support for running YOLO/DarkNet has been added to OpenCV dnn module recently.

 ## Dependencies
  * opencv
  * numpy
  * imageio-ffmpeg

`pip install numpy opencv-python imageio-ffmpeg`

**Note: Python 2.x is not supported**

 ## YOLO (You Only Look Once)

 Download the pre-trained YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights) or for tiny weights for slower machines [link](https://pjreddie.com/media/files/yolov3-tiny.weights) and place it in the current directory or you can directly download to the current directory in terminal using

 `$ wget https://pjreddie.com/media/files/yolov3.weights`

 `$ wget https://pjreddie.com/media/files/yolov3-tiny.weights`

 Provided all the files are in the current directory, below command will apply object detection on the input video `commuters.mp4`.

 `$ python yolo_opencv.py --input sampledata/commuters.mp4 --config cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights --classes cfg/yolov3.txt`

 For RTSP simply put the RTSP URL as --input

  `$ python yolo_opencv.py --input rtsp://xxxxx:1935/live/sys.stream --framestart 100 --framelimit 100 --config cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights --classes cfg/yolov3.txt`

 **Arguments**

 | parameter | type    | description                                      |
 | --------- | ------- | ------------------------------------------------ |
 | `input`     | String  | /path/to/input/stream |
 | `outputfile`  | String | /path/to/outputfile |
 | `outputdir` | String  | /path/to/outputdir  |
 | `framestart` | Int  | start detecting at frame x (int) |
 | `framelimit` | Int  | stop after x (int) frames and save the video in case of streams. 0 no limit |
 | `config` | String  | /path/to/config/file  |
 | `weights` | String  | /path/to/weights/file  |
 | `classes`  | String | /path/to/classes/file |
 | `invertcolor` | Boolean  | in case of BGR streams |
 | `fpsthrottle` | Int  | in case of slower machines to keep up with a stream  |

 ### sample output :
 ![](object-detection.png)

Checkout the object detection implementation available in [cvlib](http:cvlib.net) which enables detecting common objects in the context through a single function call `detect_common_objects()`.

 ## Credits
 This project is based on [Arun Ponnusamy's Object Detection OpenCV](https://github.com/arunponnusamy/object-detection-opencv)

 Sample video footage from [Videvo - Free Stock Video Footage](https://www.videvo.net/video/people-crossing-road-in-hong-kong-cbd/8162/)
