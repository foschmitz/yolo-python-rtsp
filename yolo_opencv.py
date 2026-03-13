#############################################
# Object detection via RTSP - YOLO - OpenCV
# Author : Frank Schmitz   (Dec 11, 2018)
# Website : https://www.github.com/zishor
############################################

import os
import os.path
import sys
import cv2
import argparse
import numpy as np
import imageio_ffmpeg as imageio
import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=False,
                    help='path to input image', default='sampledata')
    ap.add_argument('-o', '--outputfile', required=False,
                    help='filename for output video', default='output.mp4')
    ap.add_argument('-od', '--outputdir', required=False,
                    help='path to output folder', default='output')
    ap.add_argument('-fs', '--framestart', required=False,
                    help='start frame', type=int, default=0)
    ap.add_argument('-fl', '--framelimit', required=False,
                    help='number of frames to process (0 = all)', type=int, default=0)
    ap.add_argument('-c', '--config', required=False,
                    help='path to yolo config file', default='cfg/yolov3-tiny.cfg')
    ap.add_argument('-w', '--weights', required=False,
                    help='path to yolo pre-trained weights', default='yolov3-tiny.weights')
    ap.add_argument('-cl', '--classes', required=False,
                    help='path to text file containing class names', default='cfg/yolov3.txt')
    ap.add_argument('-ic', '--invertcolor', required=False,
                    help='invert RGB 2 BGR', type=str2bool, default=False)
    ap.add_argument('-fpt', '--fpsthrottle', required=False,
                    help='skips (int) x frames in order to catch up with the stream for slow machines 1 = no throttle',
                    type=int, default=1)
    return ap.parse_args()


def save_bounded_image(image, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, outputdir, invertcolor):
    label = str(classes[class_id])
    now = datetime.datetime.now()
    dirname = os.path.join(outputdir, label, now.strftime('%Y-%m-%d'))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = label + '_' + now.strftime('%Y-%m-%d_%H_%M_%S_%f') + '_conf' + "{:.2f}".format(confidence) + '.jpg'
    print('Saving bounding box:' + filename)
    roi = image[y:y_plus_h, x:x_plus_w]
    if roi.any():
        if not invertcolor:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dirname, filename), roi)


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, colors):
    label = str(classes[class_id])
    color = colors[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 3)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)


def detect(image, net, output_layer_names, classes, colors, outputdir, invertcolor):

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(output_layer_names)

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = max(0, center_x - w / 2)
                y = max(0, center_y - h / 2)
                w = min(w, Width - x)
                h = min(h, Height - y)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Save crops from original before drawing annotations
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        save_bounded_image(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h),
                           classes, outputdir, invertcolor)

    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h),
                        classes, colors)

    if invertcolor:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def processvideo(file, net, output_layer_names, classes, colors, outputfile, outputdir, invertcolor,
                 framestart=0, framelimit=0):
    cap = cv2.VideoCapture(file)

    writer = imageio.write_frames(outputfile, (int(cap.get(3)), int(cap.get(4))))
    writer.send(None)
    frame_counter = 0
    while(cap.isOpened()):
        frame_counter = frame_counter + 1
        if framelimit > 0 and frame_counter > framestart + framelimit:
            break
        ret, frame = cap.read()
        if ret == True:
            if frame_counter < framestart:
                print('Skipping frame ' + str(frame_counter))
                continue
            print('Detecting objects in frame ' + str(frame_counter))
            if frame is not None:
                image = detect(frame, net, output_layer_names, classes, colors, outputdir, invertcolor)
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

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if not os.path.isfile(args.weights):
        print('Error: weights file not found: ' + args.weights)
        print('Download weights with: wget https://pjreddie.com/media/files/yolov3-tiny.weights')
        sys.exit(1)
    if not os.path.isfile(args.config):
        print('Error: config file not found: ' + args.config)
        sys.exit(1)

    net = cv2.dnn.readNet(args.weights, args.config)

    # Compute output layer names once
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    if is_stream_url(args.input):

        cap = cv2.VideoCapture(args.input)
        if args.framelimit > 0:
            writer = imageio.write_frames(args.outputfile, (int(cap.get(3)), int(cap.get(4))))
            writer.send(None)
        frame_counter = 0
        try:
            while(True):
                if args.framelimit > 0 and frame_counter > args.framestart + args.framelimit:
                    break

                if frame_counter % args.fpsthrottle == 0:
                    ret, frame = cap.read()
                    if ret and frame_counter >= args.framestart:
                        print('Detecting objects in frame ' + str(frame_counter))
                        frame = detect(frame, net, output_layer_names, classes, colors,
                                       args.outputdir, args.invertcolor)
                        if args.framelimit > 0:
                            writer.send(frame)
                    else:
                        print('Skipping frame ' + str(frame_counter))
                else:
                    # Read and discard frame to drain the buffer
                    cap.grab()
                    print('FPS throttling. Skipping frame ' + str(frame_counter))
                frame_counter = frame_counter + 1
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
                processvideo(video_path, net, output_layer_names, classes, colors,
                             outputfile, args.outputdir, args.invertcolor,
                             args.framestart, args.framelimit)
        else:
            processvideo(args.input, net, output_layer_names, classes, colors,
                         args.outputfile, args.outputdir, args.invertcolor,
                         args.framestart, args.framelimit)


if __name__ == '__main__':
    main()
