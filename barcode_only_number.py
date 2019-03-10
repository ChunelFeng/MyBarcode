#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 22:28
# @Author  : Chunel
# @File    : barcode_only_number.py
# @Software: PyCharm
# @Brief   : 当你看到这段代码的时候，我可能已经不在海康了
#            以下是我自己在家里时候，写的一个一维码识别模型，包括训练和预测两个部分
#            源码分享给你，希望你今后工作顺利，早点学好python
#
#            使用的时候，首先打开训练模式： TRAIN_MODE = True
#            然后直接从__main__函数处，运行本py文件
#            预计普通工作机，所有默认参数不变，大概3个小时会训练完成。
#            生成的模型信息，会保存在同级的model文件夹里(一共四个文件，大小总共1G左右)
#            训练过程中，有大量的IO操作，不建议用自己笔记本跑
#            训练完毕后，进入预测模式： TRAIN_MODE = False
#            将model文件夹下生成的文件名，以demo中的形式，写在BCD_MODEL_NAME处
#            再次从__main__函数处，运行本py文件，就可以看到预测结果
#            条码长度和宽高信息，都是根据第三方库生成的结果写死的，不要随便修改
#            我自己笔记本上用的是python3.6.3版本，工作机上用的是anaconda3.5版本。应该3以上的版本，都能正常运行这个demo
#            24-29行用到的第三方库，均是必须要用的，自己装一下就可以了
#            看懂的话，可以在部门python学习小组分享的时候，讲给组里其他同事听

import os
import time
import random

# 以下库需要自行安装
import tensorflow as tf
import numpy as np
from PIL import Image
from barcode.writer import ImageWriter    # 对应的库名称为：pybarcode
from barcode.codex import Code39
import matplotlib.pyplot as plt    # 当预测的时候，想要显示随机生成的条码图片，会用到这个库。详见代码中注释


TRAIN_MODE = True           # 是否为训练模式
ACCURACY_RATE = 0.95        # 训练精度，预测准确度超过95%的时候，停止训练并且保存当前模型

BCD_IMAGE_WIDTH = 360       # 条码图片宽度(图片为第三方库生成，6位条码对应的图像宽高分别为：360，280)
BCD_IMAGE_HEIGHT = 280      # 条码图片长度
BCD_TYPE_NUM = 10           # 组成条码的内容种类，一共10种不同的情况 [0-9，纯数字]
BCD_LEN = 6                 # 条码长度，如条码为：099028，长度为：6
# 最后保存的模型的名字，放在同级的model文件夹下。1900是我自己跑的数据，你跑的时候，应该会是其他的数字，把数字替换掉就可以了
BCD_MODEL_NAME = r"./model/crack_bcd.model-1900"


def convert2gray(img):
    # 将图像变为灰度图
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
    # 将条码信息转变成向量形式
    # 比如，6位的条码900928，则变成60维的数组
    # 其中，前[0-9]位中的第9位为1，其他为0。[10-19]位中第10位为1，其他为0。以此类推
    vector = np.zeros(BCD_TYPE_NUM * BCD_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, ch in enumerate(text):
        idx = i * BCD_TYPE_NUM + char2pos(ch)
        vector[idx] = 1
    return vector


def generate_bcd_image_and_info():
    # 随机生成一个条码和对应的图片信息
    bcd_info = ''.join(str(random.choice(range(BCD_TYPE_NUM))) for _ in range(BCD_LEN))
    image_writer = ImageWriter()
    picture = Code39(bcd_info, writer=image_writer, add_checksum=False)
    image_path = picture.save(bcd_info)    # 以条码信息保存图片

    bcd_image = Image.open(image_path)
    bcd_image = np.array(bcd_image)
    os.remove(image_path)    # 处理完了，删除图片

    return bcd_info, bcd_image


def crack_bcd_cnn(w_alpha=0.01, b_alpha=0.1):
    # 构建cnn神经网络
    x = tf.reshape(X, shape=[-1, BCD_IMAGE_HEIGHT, BCD_IMAGE_WIDTH, 1])

    # 第一层CNN
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))    # 3，3是卷积核大小，1是输入维度，32是输出维度
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding="SAME"), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 以2*2的窗口做pooling
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # 第二层CNN
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))  # 3，3还是卷积核，32是上一层的输出维度，当作这一层的输入维度。64是这一层的输出维度
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding="SAME"), b_c2))    # 卷积之后的结果，过relu函数
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")    # 对2*2的窗口做pooling
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 第三层CNN
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding="SAME"), b_c3))    # 卷积之后的结果，走relu函数
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")    # 对2*2的窗口做pooling
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 将pooling后的数据，转成1024维的向量，[a*b*c]指的是前面的特征图pool之后的结果
    # 其中，35=BCD_IMAGE_HEIGHT/(2***)，280除以2的三次方，表示被以2为单位pooling三次， 45=BCD_IMAGE_WIDTH/(2***)
    w_d = tf.Variable(w_alpha*tf.random_normal([35*45*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])  # 将conv3进行转换，变换成35*45*64的形式
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))  # 用变换好的东西，*w+b
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, BCD_TYPE_NUM*BCD_LEN]))  # 将1024维向量，转成60维的值
    b_out = tf.Variable(b_alpha*tf.random_normal([BCD_TYPE_NUM*BCD_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def get_bcd_batch(batch_size=64):
    # 一次性获取一个batch(批量的数据)，用于训练模型，和训练时
    batch_x = np.zeros([batch_size, BCD_IMAGE_WIDTH*BCD_IMAGE_HEIGHT])
    batch_y = np.zeros([batch_size, BCD_LEN*BCD_TYPE_NUM])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = generate_bcd_image_and_info()
            if image.shape == (BCD_IMAGE_HEIGHT, BCD_IMAGE_WIDTH, 3):
                return text, image

    # 分别将获取batch_size张图片，和对应条码信息(60维的数组)，放到batch_x和batch_y中去
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255    # 将图像信息的每个像素点值，转到0-1之间的值
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def bcd_train():
    # 开始训练流程
    output = crack_bcd_cnn()

    # 设置训练参数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, BCD_LEN, BCD_TYPE_NUM])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, BCD_LEN, BCD_TYPE_NUM]), 2)
    correct_predict = tf.equal(max_idx_p, max_idx_l)    # 比较模型跑出来的值，和给定的值是否一致
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))    # 计算精度的函数

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            # 持续的训练，每批次送入64张图片(一个batch大小为64)
            batch_x, batch_y = get_bcd_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)    # step是训练的批次，loss_值的大小，表示预测结果和真实结果之间的差距

            if step % 100 == 0:
                # 每次训练是100的倍数的时候，计算一下当前模型的精确度
                batch_x_test, batch_y_test = get_bcd_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0})
                print("*** step={}, acc={}", step, acc)

                # 如果当前模型精确度，大于设定的ACCURACY_RATE，则停止训练，并保存当前模型，放置在model文件夹下，训练流程结束
                if acc > ACCURACY_RATE:
                    print("save model")
                    saver.save(sess, r"./model/crack_bcd.model", global_step=step)    # 指定保存的模型的名字
                    break
            step += 1


def crack_bcd(bcd_image):
    # 根据训练出来的模型，输入的bcd_image信息，并返回条码信息结果
    output = crack_bcd_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, BCD_MODEL_NAME)    # 加载训练模型
        predict = tf.argmax(tf.reshape(output, [-1, BCD_LEN, BCD_TYPE_NUM]), 2)
        test_list = sess.run(predict, feed_dict={X: [bcd_image], keep_prob: 1})
        text = test_list[0].tolist()
        return text    # 返回预测的条码信息


def bcd_predict():
    # 开始预测流程
    text, image = generate_bcd_image_and_info()

    # 以下代码，打开注释，可以看随机生成的条码图像。注意，plt.show()是阻塞函数
    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.text(0.1, 0.9, text, ha="center", va="center", transform=ax.transAxes)
    # plt.imshow(image)
    # plt.show()

    image = convert2gray(image)
    image = image.flatten() / 255

    predict_info = crack_bcd(image)
    print("real : {}, predict : {}".format(text, predict_info))


if __name__ == "__main__":
    if not os.path.exists("./model"):
        os.mkdir("model")    # 先创建一个model文件夹，用来保存训练模型

    # 入口函数，先定义几个tf的通用参数
    # X代表输入(图像信息)，Y代表输出(图像中的条码值信息)。整体思路 Y = w*X + b
    # 在训练的时候，根据大量的数据，求取w和b的值
    # 预测的时候，根据求得的w和b，传入X，获取Y值
    X = tf.placeholder(tf.float32, [None, BCD_IMAGE_HEIGHT*BCD_IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, BCD_LEN*BCD_TYPE_NUM])
    keep_prob = tf.placeholder(tf.float32)

    if TRAIN_MODE:
        start = time.time()
        bcd_train()    # 训练流程
        print(time.time() - start)    # 打印训练时间。我在公司的工作机上，精确度为95%的情况下，大概跑了三个小时
    else:
        bcd_predict()    # 预测流程
