#
# ----------------------------------------------------------------------------------------------------------------------
from core.sort import *
import imutils
import time
import cv2


# ----------------------------------------------------------------------------------------------------------------------
# 函数的初始化以及参数的初始化
# line = [(0, 200), (400, 200)] --- 表示所画线的位置，参数是两个坐标
# ----------------------------------------------------------------------------------------------------------------------

tracker = Sort()
memory = {}
# line = [(0, 450), (1280, 450)] --- test_1
# line = [(0, 450), (1280, 450)] --- test_2
# line = [(0, 450), (1920, 130)] --- video-02
line = [(0, 450), (1920, 130)]
counter = 0
counter_up = 0
counter_down = 0
writer = None


# ----------------------------------------------------------------------------------------------------------------------
# 下面两个函数用于得到方框和线的关系 --- 类似于碰撞检测
# ----------------------------------------------------------------------------------------------------------------------

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
#return ccw(A,C,D) !=ccw(B,C,D) and ccw(A,B,C) !=ccw（A,B,D）
#


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# --------------------------------------------------------------------------------------------------------

"""
# load the COCO class labels our YOLO model was trained on
# 加载存放可以识别物体的名称，例如car,people,truck...，将得到的名称存放在LABELS中
# LABELS ---> 类别
#
# initialize a list of colors to represent each possible class label
# 一张图片可以有上述多个可以被识别的物体，使用框圈起来的时候，要使用不一样的颜色
# np.random.seed() ---> 设置随即种子
#
# derive the paths to the YOLO weights and model configuration
# 加载权重和相应的配置数据
#
# load our YOLO object detector trained on COCO dataset (80 classes)
# 加载好数据之后，开始利用上述数据恢复yolo训练的时候神经网络
# 神经网络 ---> net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# 获取YOLO输出层的名称
# ln = net.getLayerNames() ---> 得到神经网络层的名称 ---> ['conv_0', 'bn_0', 'relu_0', 'conv_1', 'bn_1', 'relu_1', 'conv_2', 'bn_2', 'relu_2'...]
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] ---> ['yolo_82', 'yolo_94', 'yolo_106']

"""
# ----------------------------------------------------------------------------------------------------------------------


labelsPath = "./yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

weightsPath = "./yolo-coco/yolov3.weights"
configPath = "./yolo-coco/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ----------------------------------------------------------------------------------------------------------------------
"""
# cv2.VideoCapture() --- 里面参数如果地址，打开视频文件 --- 里面参数是0/1，打开摄像头
# 当参数是0的时候，打开计算机的内置摄像头，当参数为1的时候打开计算机的外置摄像头
# (W, H) = (None, None) --- 视频的宽度和高度，初始化视频编写器（writer）和帧尺寸

"""
# ----------------------------------------------------------------------------------------------------------------------

vs = cv2.VideoCapture('./input/video-02.mp4')
(W, H) = (None, None)

# ----------------------------------------------------------------------------------------------------------------------
"""
# try to determine the total number of frames in the video file
# 打开一个指向视频文件的文件指针，循环读取帧 --- 尝试确定视频文件中的总帧数（total），以便估计整个视频的处理时间;
# CV_CAP_PROP_FRAME_COUNT --- 视频的帧数
# 这里使用是处理视频的时候固定的过程，不必过度的纠结其使用 ---
#					   if imutils.is_cv2():
#					   	  prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
#					   else:
#					   	  prop = cv2.CAP_PROP_FRAME_COUNT
#
# vs.get(prop) --- cv2.VideoCapture.get(prop) --- 得到视频的总帧数 
# print("[INFO] {} total frames in video".format(total)) --- 输出视频的帧数

"""
# ----------------------------------------------------------------------------------------------------------------------
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

while True:
    # ----------------------------------------------------------------------------------------------------------------------

    """
    # cv2.VideoCapture.read() ---> 读取视频，在while中循环读取视频的frame
    # vs.read() ---> 得到两个参数，其中ret是布尔值，如果读取帧是正确的则返回True，
    # 如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
    # 第一个参数为False的时候，if not grabbed --- True --- 循环结束，

    """
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]


    """
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    # 将图片构建成一个blob，设置图片尺寸，然后执行一次，YOLO前馈网络计算，最终获取边界框和相应概率
    # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False) -------
    # 这个函数是用来读取图片的接口，其中参数很重要，会直接影响到模型的检测效果，前面几个参数与模型训练的时候对图片
    # 进行预处理有关系。其中最后一个参数是blob = cv2.dnn.blobFromImage()，swapRB，是选择是否交换R与B颜色通道，
    # 一般用opencv读取caffe的模型就需要将这个参数设置为false， 读取tensorflow的模型， 则默认选择True即可，读取
    # dnn(暗网)时，即yolo网络，swapRB设置为True，需要交换R与B颜色通道。
    # layerOutputs = net.forward(ln) ---> YOLO前馈网络计算，最终获取边界框和相应概率
    # 
    """

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    """
    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    # boxes ---> 用于存放识别物体的框的一些信息 ---> 框的左上角横坐标x和纵坐标y以及框的高h和宽w
    # confidences ---> 表示识别是某种物体的可信度
    # classIDs ---> 表示识别物体归属于哪一类 ---> ['person', 'bicycle', 'car', 'motorbike'....]

    """
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            """
            # extract the class ID and confidence (i.e., probability) of the current object detection
            # scores = detection[5:] ---> [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  							   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #    						   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  							   0. 0. 0. 0. 0. 0. 0. 0.]
            #							  ....
            # classID = np.argmax(scores) ---> 得到物体的类别ID
            # confidence = scores[classID] ---> 得到是某种物体的置信度

            """
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            """
            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            # 只保留置信度大于某值的边界框 ---> 这里设置的是50%的置信度，也就是可能性大于百分之五十就可以显示出来
            # 如果图片的质量较差或者视频的质量较差， 那么可以将置信度的边界值调整低一些，这样也许可以识别更多的物体

            """

            if confidence > 0.3:
                """
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                # 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是边界框的中心坐标以及边界框的宽度和高度
                # np.array ---> 生成一个数组 
                # box.astype("int") ---> 使用 astype("int") 对上述 array 进行强制类型转换
                # centerX ---> 框的中心点横坐标， centerY ---> 框的中心点纵坐标
                # width ---> 框的宽度， height ---> 框的高度

                """
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                """
                # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                # x = int(centerX - (width / 2))  ---> 计算边界框的左上角的横坐标
                # y = int(centerY - (height / 2)) ---> 计算边界框的左上角的纵坐标

                """

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                """
                # update our list of bounding box coordinates, confidences,and class IDs
                # boxes.append([x, y, int(width), int(height)]) ---> 将边框的信息添加到列表boxes
                # confidences.append(float(confidence)) ---> 将识别出是某种物体的置信度添加到列表confidences
                # classIDs.append(classID) ---> 将识别物体归属于哪一类的信息添加到列表classIDs

                """

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    """
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes ensure at least one detection exists
    # 使用非极大值抑制方法抑制弱、重叠边界框 ---> 对于检测出结果的优化处理
    # 应用非最大值抑制可以抑制明显重叠的边界框，只保留最自信（confidence）的边界框，NMS还确保我们没有任何冗余或无关的边界框。
    # 利用OpenCV内置的NMS DNN模块实现即可实现非最大值抑制 ，所需要的参数是边界 框、 置信度、以及置信度阈值和NMS阈值。
    # cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"]) --- 第一个参数是存放边界框的列表
    # 第二个参数是存放置信度的列表，第三个参数是自己设置的置信度，第四个参数是关于threshold（阈值）

    """
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    """
    # if len(idxs) > 0 ---> 假设存在至少一个检测结果，就循环用非最大值抑制确定idx ---> for i in idxs.flatten()。
    # 然后，我们使用随机类颜色在图像上绘制边界框和文本。最后，显示结果图像，直到用户按下键盘上的任意键。--- cv2.waitKey(0)
    # dets = [] --- 用于存放方框的五个信息 --- 左上角横坐标，纵坐标，右下角横坐标，纵坐标，检测到物体的置信度
    #
    """

    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            """
            #
            # 因为yolo可以识别的物体有很多，这里因为只是识别人，所以加了一个条件语句 if LABELS[classIDs[i]] == "car":
            # (x, y) --- 方框的左上角坐标，(w, h) --- 方框的宽和高
            # dets.append --- 将五个信息加入到列表之中 --- 用于跟踪
            #
            """

            if LABELS[classIDs[i]] == "car" or LABELS[classIDs[i]] == "bus" or LABELS[classIDs[i]] == "truck":
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x + w, y + h, confidences[i]])
    """
    #
    # np.set_printoptions --- 改变数据类型 --- 整数类型转为浮点数类型，而且规定小数点后三位
    # np.asarray --- 结构数据转化为ndarray
    # 
    """
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    """
    #
    # 这里设置条件语句的原因 --- 有时候yolo跟踪目标会出现检测不到的现象，但是依旧是传入数据，导致卡尔曼滤波出现错误
    # tracks = tracker.update(dets) --- 将数据传入跟踪器
    #
    """
    if np.size(dets) == 0:
        continue
    else:
        tracks = tracker.update(dets)

# ----------------------------------------------------------------------------------------------------------------------

    """
    #
    # 数据传入跟踪器之后得到的返回值的处理
    # tracks --- 得到的返回值 --- 返回值有五个信息，和之前一样，左上角横坐标，纵坐标，右下角横坐标，纵坐标，检测到物体的置信度
    # boxes --- 存放tracks的前四个返回值
    # indexIDs --- 存放tracks的第五个返回值 --- 物体的置信度
    # memory = {} --- 用于存放boxes最后一个数据的信息
    # previous = memory.copy() --- 用于存放之前的信息，
    #
    """

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    """
    #
    # if len(boxes) > 0: --- 如果boxes存有数据的话 --- 执行下面的代码
    # (x, y) = (int(box[0]), int(box[1])) --- 左上角坐标
    # (w, h) = (int(box[2]), int(box[3])) --- 方框的宽和高
    # color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]] --- 对颜色的设定，可以设置为一个颜色 --- [0,0,255]
    # cv2.rectangle(frame, (x, y), (w, h), color, 2) --- 绘制方框
    # 第一个参数是当前帧，第二个参数是左上角坐标，第三个参数是方框的宽和高，第四个参数是颜色，第五个参数是边框的宽度
    #
    """

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            """
            #
            # if indexIDs[i] in previous: --- 这里处理的是当前数据和现在数据之间的关联
            # previous_box --- 之前识别的方框
            # (x2, y2)，(w2, h2) --- 之前方框的数据 --- 左上角坐标，宽和高
            # p0 --- 当前方框的中心点
            # p1 --- 之前方框的中心点
            # cv2.line(frame, p0, p1, color, 3) --- 将两个点用直线连接起来
            #
            """

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                cv2.line(frame, p0, p1, color, 3)

                """
                #
                # 做碰撞检测 --- 两点的连线穿过设置的基准线 --- 行人的数目+1
                # if y2 < y: --- 判断方向 --- 行人是上还是下 --- 如果行人时左右行进的时候，这里面设置为x2和x之间的关系
                #
                """
                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
                    if y2 < y:
                        counter_down += 1
                    else:
                        counter_up += 1

            i += 1
    """
    #
    # cv2.line(frame, line[0], line[1], (0, 255, 0), 3) --- 绘制基准线，根据自己的实际的参数，选择
    # cv2.putText --- 左上角计数数据的显示
    # str(counter) --- counter是整数类型的数据，必须要转为字符型的数据才可以显示
    # cv2.putText --- 参数 --- 当前帧，显示内容，显示位置，字体类型，字体大小，字体颜色，字体胖瘦（哈哈哈）
    #
    """
    cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
    cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
    cv2.putText(frame, str(counter_up), (130, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 0), 3)
    cv2.putText(frame, str(counter_down), (230, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

    """
    # if writer is None: --- 没有设置视频编码器信息，执行下面的代码
    # cv2.VideoWriter_fourcc(*"mp4v") --- 设置编码格式
    # cv2.VideoWriter --- 视频编码信息的设置
    # writer.write(frame) --- 将视频流的帧经过处理后写入视频
    #
    """

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("./output/output.mp4",
                                 fourcc,
                                 30,
                                 (frame.shape[1], frame.shape[0]),
                                 True)
    writer.write(frame)

    """
    #
    # cv2.imshow --- 显示当前帧
    # if cv2.waitKey(1) & 0xFF == ord('q'):break --- 按下Q的时候程序退出
    #
    """
    frame = imutils.resize(frame, width=1080)
    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


writer.release()