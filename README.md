# YOLO Object Detection via RTSP / Video / Webcam

Object detection using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) on video files and RTSP streams.

Detected objects are cropped and saved in date-separated folders per class — useful for further training or recognition tasks.

## Setup

```bash
pip install -r requirements.txt
```

The YOLO model weights (e.g. `yolo11n.pt`) are downloaded automatically on first run.

## Usage

**Local video:**
```bash
python yolo_opencv.py --input sampledata/commuters.mp4
```

**RTSP stream:**
```bash
python yolo_opencv.py --input rtsp://user:pass@ip:port/stream --framelimit 100
```

**Directory of videos:**
```bash
python yolo_opencv.py --input /path/to/videos/
```

## Arguments

| parameter | type | default | description |
|-----------|------|---------|-------------|
| `--input` | String | `sampledata` | path to video file, directory, or stream URL |
| `--outputfile` | String | `output.mp4` | output video filename |
| `--outputdir` | String | `output` | folder for cropped detections |
| `--model` | String | `yolo11n.pt` | YOLO model (`yolo11n.pt`, `yolo11s.pt`, `yolov8n.pt`, etc.) |
| `--framestart` | Int | `0` | start detecting at frame N |
| `--framelimit` | Int | `0` | stop after N frames (0 = all) |
| `--confidence` | Float | `0.5` | detection confidence threshold |
| `--padding` | Float | `0.15` | padding around crops (0.15 = 15%) |
| `--fpsthrottle` | Int | `1` | process every Nth frame (for slow machines) |

### Sample output:
![](object-detection.png)

## Credits
Based on [Arun Ponnusamy's Object Detection OpenCV](https://github.com/arunponnusamy/object-detection-opencv).

Sample video from [Videvo](https://www.videvo.net/video/people-crossing-road-in-hong-kong-cbd/8162/).
