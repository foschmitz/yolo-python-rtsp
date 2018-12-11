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
  
`pip install numpy opencv-python`

**Note: Python 2.x is not supported**

 ## YOLO (You Only Look Once)
 
 Download the pre-trained YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights) or for tiny weights for slower machines [link](https://pjreddie.com/media/files/yolov3-tiny.weights) and place it in the current directory or you can directly download to the current directory in terminal using
 
 `$ wget https://pjreddie.com/media/files/yolov3.weights`
 
 `$ wget https://pjreddie.com/media/files/yolov3-tiny.weights`
 
 Provided all the files are in the current directory, below command will apply object detection on the input image `dog.jpg`.
 
 `$ python yolo_opencv.py --input sampledata/commuters.mp4 --config cfg/yolov3.cfg --weights yolov3-tiny.weights --classes cfg/yolov3.txt`
 
 For RTSP simply put the RTSP URL as --input
 
  `$ python yolo_opencv.py --input rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov --config cfg/yolov3.cfg --weights yolov3-tiny.weights --classes cfg/yolov3.txt`

  (stream courtesy of [Wowza Demo RTSP](https://www.wowza.com/demo/rtsp) 

 **Command format** 
 
 _$ python yolo_opencv.py --input /path/to/input/stream --outputfile /path/to/outputfile --outputdir /path/to/outputdir --framestart 0 (start detecting at frame x (int))  --framelimit 0 (stop after x (int) frames and save the video in case of streams. 0 no limit) --config /path/to/config/file --weights /path/to/weights/file --classes /path/to/classes/file_
 
 ### sample output :
 ![](object-detection.png)
 
Checkout the object detection implementation available in [cvlib](http:cvlib.net) which enables detecting common objects in the context through a single function call `detect_common_objects()`.
 
 ## Credits
 This project is based on [Arun Ponnusamy's Object Detection OpenCV](https://github.com/arunponnusamy/object-detection-opencv)
 
 Sample video footage from [Videvo - Free Stock Video Footage](https://www.videvo.net/video/people-crossing-road-in-hong-kong-cbd/8162/)

 Sample RTSP stream courtesy of [Wowza Demo RTSP](https://www.wowza.com/demo/rtsp) 