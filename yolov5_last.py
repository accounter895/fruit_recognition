import os
import cv2
import sys
import argparse
import time
import numpy as np
from flask import Flask, Response, render_template
from py_utils.coco_utils import COCO_test_helper

# 实现了用flask框架把模型输出的数据流输出到浏览器上 在执行了这个文件后，只要访问http://<跑模型主机的IP地址>:5000这个地址，就会在浏览器上显示检测结果
# 模型输入为图片，模型输出为检测框，检测框的类别和置信度

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The following two params are for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
           "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
           "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

app = Flask(__name__)

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold and only keep specified classes.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    # Define the allowed class indices for "banana", "apple", "orange"
    allowed_classes = [0, 46, 47, 49]  # Corresponding to "banana", "apple", "orange"

    # Filter classes to only include allowed classes
    valid_indices = np.isin(classes, allowed_classes)
    boxes = boxes[valid_indices]
    classes = classes[valid_indices]
    scores = scores[valid_indices]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:, :2, :, :] * 2 - 0.5
    box_wh = pow(position[:, 2:4, :, :] * 2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2  # bottom right y

    return xyxy


def post_process(input_data, anchors):
    print("Model output shapes:")
    for idx, output in enumerate(input_data):
        print(f"Output {idx} shape: {output.shape}")

    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    input_data = [_in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:, :4, :, :], anchors[i]))
        scores.append(input_data[i][:, 4:5, :, :])
        classes_conf.append(input_data[i][:, 5:, :, :])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold and allowed classes
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # Debug: Print the unique class indices found after filtering
    if classes is not None:
        print(f"Unique class indices found: {np.unique(classes)}")

    # nms
    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        if cl >= len(CLASSES):
            print(f"Warning: Class index {cl} is out of range. Skipping this detection.")
            continue
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform


def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def generate_frames(model, platform, anchors, co_helper):
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头，可以更改
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 设置摄像头帧率
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 初始化帧率计算变量
    start_time = time.time()
    frame_count = 0

    while True:
        ret, img_src = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 处理图像
        pad_color = (0, 0, 0)
        img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.
        else:
            input_data = img

        outputs = model.run([np.expand_dims(input_data, 0)])
        boxes, classes, scores = post_process(outputs, anchors)

        if boxes is not None:
            draw(img_src, co_helper.get_real_box(boxes), scores, classes)

        # 计算帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
            cv2.putText(img_src, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 调整图像大小
        img_src = cv2.resize(img_src, (img_src.shape[1] // 1, img_src.shape[0] // 1))  # 修改这里

        # 显示在本地显示屏上
        cv2.imshow('YOLOv5 Detection', img_src)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', img_src)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()    
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(model, platform, anchors, co_helper),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required=True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')

    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    parser.add_argument('--anchors', type=str, default='../model/anchors_yolov5.txt', help='target to anchor file, only yolov5, yolov7 need this param')

    args = parser.parse_args()

    # load anchor
    with open(args.anchors, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3, -1, 2).tolist()
    print("use anchors from '{}', which is {}".format(args.anchors, anchors))

    # init model
    model, platform = setup_model(args)

    # init COCO_test_helper
    co_helper = COCO_test_helper(enable_letter_box=True)

    # Create and run the Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
