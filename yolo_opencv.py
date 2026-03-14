#############################################
# Object detection via RTSP - YOLO - OpenCV
# Author : Frank Schmitz   (Dec 11, 2018)
# Website : https://www.github.com/zishor
############################################

import os
import sys
import cv2
import argparse
import datetime
import imageio_ffmpeg as imageio
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser(description='YOLO object detection on video files and RTSP streams')
    ap.add_argument('-i', '--input', required=False,
                    help='path to input video or stream URL', default='sampledata')
    ap.add_argument('-o', '--outputfile', required=False,
                    help='filename for output video', default='output.mp4')
    ap.add_argument('-od', '--outputdir', required=False,
                    help='path to output folder for cropped detections', default='output')
    ap.add_argument('-m', '--model', required=False,
                    help='YOLO model to use (e.g. yolo11n.pt, yolov8s.pt)', default='yolo11n.pt')
    ap.add_argument('-fs', '--framestart', required=False,
                    help='start frame', type=int, default=0)
    ap.add_argument('-fl', '--framelimit', required=False,
                    help='number of frames to process (0 = all)', type=int, default=0)
    ap.add_argument('-conf', '--confidence', required=False,
                    help='confidence threshold (0.0 - 1.0)', type=float, default=0.5)
    ap.add_argument('-pad', '--padding', required=False,
                    help='padding around crop as fraction (e.g. 0.2 = 20%%)', type=float, default=0.15)
    ap.add_argument('-fpt', '--fpsthrottle', required=False,
                    help='skip every N frames to keep up with stream (1 = no throttle)',
                    type=int, default=1)
    return ap.parse_args()


def save_crop(image, label, confidence, x1, y1, x2, y2, outputdir, padding):
    """Save a padded crop of the detected object."""
    h, w = image.shape[:2]

    # Add padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    now = datetime.datetime.now()
    dirname = os.path.join(outputdir, label, now.strftime('%Y-%m-%d'))
    os.makedirs(dirname, exist_ok=True)

    filename = label + '_' + now.strftime('%Y-%m-%d_%H_%M_%S_%f') + '_conf' + "{:.2f}".format(confidence) + '.jpg'
    print('Saving crop: ' + filename)
    roi = image[y1:y2, x1:x2]
    if roi.size > 0:
        cv2.imwrite(os.path.join(dirname, filename), roi)


def detect(frame, model, confidence, outputdir, padding):
    """Run YOLO detection, save crops, draw annotations, return annotated frame."""
    results = model(frame, conf=confidence, verbose=False)[0]

    # Save crops before drawing
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = results.names[int(box.cls[0])]
        save_crop(frame, label, conf, x1, y1, x2, y2, outputdir, padding)

    # Draw annotations
    annotated = results.plot()
    return annotated


def processvideo(file, model, outputfile, outputdir, confidence, padding, framestart=0, framelimit=0):
    cap = cv2.VideoCapture(file)

    writer = imageio.write_frames(outputfile, (int(cap.get(3)), int(cap.get(4))))
    writer.send(None)
    frame_counter = 0
    while cap.isOpened():
        frame_counter += 1
        if framelimit > 0 and frame_counter > framestart + framelimit:
            break
        ret, frame = cap.read()
        if ret:
            if frame_counter < framestart:
                print('Skipping frame ' + str(frame_counter))
                continue
            print('Detecting objects in frame ' + str(frame_counter))
            if frame is not None:
                image = detect(frame, model, confidence, outputdir, padding)
                writer.send(image)
            else:
                print('Frame error in frame ' + str(frame_counter))
        else:
            break
    cap.release()
    writer.close()


def is_stream_url(input_path):
    stream_prefixes = ('rtsp://', 'rtsps://', 'rtmp://', 'http://', 'https://')
    return any(input_path.startswith(p) for p in stream_prefixes)


def main():
    args = parse_args()

    model = YOLO(args.model)

    if is_stream_url(args.input):
        cap = cv2.VideoCapture(args.input)
        if args.framelimit > 0:
            writer = imageio.write_frames(args.outputfile, (int(cap.get(3)), int(cap.get(4))))
            writer.send(None)
        frame_counter = 0
        try:
            while True:
                if args.framelimit > 0 and frame_counter > args.framestart + args.framelimit:
                    break

                if frame_counter % args.fpsthrottle == 0:
                    ret, frame = cap.read()
                    if ret and frame_counter >= args.framestart:
                        print('Detecting objects in frame ' + str(frame_counter))
                        frame = detect(frame, model, args.confidence, args.outputdir, args.padding)
                        if args.framelimit > 0:
                            writer.send(frame)
                    else:
                        print('Skipping frame ' + str(frame_counter))
                else:
                    cap.grab()
                    print('FPS throttling. Skipping frame ' + str(frame_counter))
                frame_counter += 1
        except KeyboardInterrupt:
            print('\nInterrupted by user')
        finally:
            cap.release()
            if args.framelimit > 0:
                writer.close()

    else:
        if os.path.isdir(args.input):
            video_files = []
            for dirpath, dirnames, filenames in os.walk(args.input):
                for filename in [f for f in filenames if f.endswith(".mp4")]:
                    video_files.append(os.path.join(dirpath, filename))
            for idx, video_path in enumerate(video_files):
                base, ext = os.path.splitext(args.outputfile)
                if len(video_files) > 1:
                    outputfile = base + '_' + str(idx) + ext
                else:
                    outputfile = args.outputfile
                print('Processing video ' + video_path)
                processvideo(video_path, model, outputfile, args.outputdir,
                             args.confidence, args.padding, args.framestart, args.framelimit)
        else:
            processvideo(args.input, model, args.outputfile, args.outputdir,
                         args.confidence, args.padding, args.framestart, args.framelimit)


if __name__ == '__main__':
    main()
