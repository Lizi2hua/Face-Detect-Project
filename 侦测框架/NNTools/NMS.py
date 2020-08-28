import numpy as np
# 舍不得删的废案
# def NMS(boxes,thresh=0.2):
#        # The input data shoulde be like:[[Cofidence,top_left_x,top_left_y,botton_right_x,botton_right_y],[],[]]
#        boxes=np.array(boxes,dtype=float)
#        #进行降序排序
#        boxes=boxes[np.argsort(boxes[:,0])[::-1]]
#        # print(boxes)
#        #将boxes第一个元素与剩下元素计算Iou，大于阈值则删除，小于则保留
#        for i in range(len(boxes)):
#               j=i+1
#               # print("i=", i)
#               # print("boxed len",len(boxes))
#               if i >=len(boxes):
#                      break
#               compare_base_box=boxes[i][1:]
#               # print(max_confidence_box)
#               for item in boxes[j:]:
#                      # print(j)
#                      # print(item)
#                      item=item[1:]
#                      _,_,IOU=iou(item,compare_base_box)
#                      # print(IOU)
#                      #因为np.delele删除后的boxes维度会缩小一行， 所以删除时，遍历指针j相当于+1
#                      if IOU>=thresh:
#                             boxes=np.delete(boxes,j,0)
#                             # print('delte!')
#                      else:j+=1
#                      # print("________")
#               # print(boxes)
#        return boxes
# import numpy as np
# # IoU in Rectangle Condition
# def iou(box1,boxes,is_Min=False):
#     """
#     :param box1: 输入的框，形式为 [top_left_x,top_left_y,botton_right_x,botton_right_y]
#     :param boxes: 输入的框，形式为 [[top_left_x,top_left_y,botton_right_x,botton_right_y],[]]
#     :param is_Min: 是否用最小面积计算IoU
#     :return: 交并比
#     """
#     #计算交集面积
#     inter_top_left_x=np.maximum(box1[0],boxes[:,0])
#     inter_top_left_y=np.maximum(box1[1],boxes[:,1])
#     inter_botton_right_x=np.minimum(box1[2],boxes[:,2])
#     inter_botton_right_y=np.minimum(box1[3],boxes[:,3])
#     # 判断是否满足形成交集的条件:右下角的x值大于左上角的x值，右下角的y值大于左上角的y值（即右下角的坐标应该在右下角，、
#     # 左上角的坐标应该在左上角）
#     # w=右下角的x值-左上角的x值，h=右下角的y值-左上角的y值
#     w=np.maximum(0,inter_botton_right_x-inter_top_left_x)
#     h=np.maximum(0,inter_botton_right_y-inter_top_left_y)
#     inter_area=w*h
#     #计算并集面积 union_area=box1_area+box2_area-inter_area
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     if is_Min:
#         IOU=np.true_divide(inter_area,np.minimum(box1_area,boxes_area))
#     else:
#         IOU=np.true_divide(inter_area,(box1_area+boxes_area-inter_area))
#     return IOU
#
# def NMS(boxes,thresh=0.2,is_Min=False):
#        """
#        :param boxes: 输入
#        :param thresh: 超参，阈值，用于判断是否舍弃数据
#        :param is_Min: IoU函数的参数
#        :return: NMS后的数据
#        """
#        #如果网络没有输出建议框,box.shape=(0,)
#        if boxes.shape[0]==0:
#               return np.array([])
#        buffer=[]
#        # boxes_sorted=boxes[np.argsort(boxes[:,0])[::-1]]#降序
#        boxes_sorted=boxes[(-boxes[:,0]).argsort()]
#        # print(boxes_sorted.shape)
#        # print(boxes_sorted)
#        # while boxes_sorted.shape[0]>1:
#        while boxes_sorted.shape[0] >=1:
#            #取第一个框
#               a_box=boxes_sorted[0]
#
#            #取剩下的框
#               b_boxes=boxes_sorted[1:]
#            #第一个框必定是保留的
#               buffer.append(a_box)
#            #比较IoU，大于阈值的框去掉，用bool索引
#               IOU=iou(a_box,b_boxes,is_Min)
#               mask=np.where(IOU<thresh)
#
#            #用bool索引得到得array来替代上一次迭代保留的框的array数据
#               boxes_sorted=b_boxes[mask]
#
#
#        # if boxes_sorted.shape[0]>0:
#        #        buffer.append(boxes_sorted[0])#boxes_sorted=[[]]的形状
#
#        return np.stack(buffer)
#        # return buffer

import numpy as np

#重叠率
def iou(box, boxes, is_Min = False): #1st框，一堆框，inMin(IOU有两种：一个除以最小值，一个除以并集)
    #计算面积：[x1,y1,x2,y3]
    box_area = (box[2] - box[0]) * (box[3] - box[1]) #原始框的面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  #数组代替循环

    #找交集：
    xx1 = np.maximum(box[0], boxes[:, 0]) #横坐标，左上角最大值
    yy1 = np.maximum(box[1], boxes[:, 1]) #纵坐标，左上角最大值
    xx2 = np.minimum(box[2], boxes[:, 2]) #横坐标，右下角最小值
    yy2 = np.minimum(box[3], boxes[:, 3]) #纵坐标，右小角最小值

    # 判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    #交集的面积
    inter = w * h  #对应位置元素相乘
    if is_Min: #若果为False
        ovr = np.true_divide(inter, np.minimum(box_area, area)) #最小面积的IOU：O网络用
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))  #并集的IOU：P和R网络用；交集/并集

    return ovr


#非极大值抑制
#思路：首先根据对置信度进行排序，找出最大值框与每个框做IOU比较，再讲保留下来的框再进行循环比较，知道符合条件，保留其框
def NMS(boxes, thresh=0.3, is_Min = False):
    #框的长度为0时(防止程序有缺陷报错)
    if boxes.shape[0] == 0:
        return np.array([])

    #框的长度不为0时
    #根据置信度排序：[x1,y1,x2,y2,C]
    _boxes = boxes[(-boxes[:, 0]).argsort()] # #根据置信度“由大到小”，默认有小到大（加符号可反向排序）
    #创建空列表，存放保留剩余的框
    r_boxes = []
    # 用1st个框，与其余的框进行比较，当长度小于等于1时停止（比len(_boxes)-1次）
    while _boxes.shape[0] > 1: #shape[0]等价于shape(0),代表0轴上框的个数（维数）
        #取出第1个框
        a_box = _boxes[0]
        #取出剩余的框
        b_boxes = _boxes[1:]

        #将1st个框加入列表
        r_boxes.append(a_box) ##每循环一次往，添加一个框
        # print(iou(a_box, b_boxes))

        #比较IOU，将符合阈值条件的的框保留下来
        index = np.where(iou(a_box, b_boxes,is_Min) < thresh) #将阈值小于0.3的建议框保留下来，返回保留框的索引
        _boxes = b_boxes[index] #循环控制条件；取出阈值小于0.3的建议框

    if _boxes.shape[0] > 0: ##最后一次，结果只用1st个符合或只有一个符合，若框的个数大于1；★此处_boxes调用的是whilex循环里的，此判断条件放在循环里和外都可以（只有在函数类外才可产生局部作用于）
        r_boxes.append(_boxes[0]) #将此框添加到列表中
    #stack组装为矩阵：:将列表中的数据在0轴上堆叠（行方向）
    return np.stack(r_boxes)

#
#
#
#
# boxes=np.array([[0.97,133,80,225,265],
#        [0.89,157,69,261,238],
#        [0.85,148,132,238,282],
#        [0.70,105,112,195,303],
#        [0.69,88,50,187,193],
#        [0.70,316,209,378,312],
#        [0.50,298,173,348,340],
#        [0.90,490,67,563,251],
#        [0.70,446,46,526,181],
#        [0.79,533,41,622,175],
#        [0.85,429,87,619,216]])
# box=NMS(boxes,thresh=0.1)
# print(box)
#
