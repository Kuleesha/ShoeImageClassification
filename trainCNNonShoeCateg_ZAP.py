# -*- coding: utf-8 -*-
"""
@author: Kuleesha
"""

#train category for  shoes UTK50K
import os
import tensorflow as tf
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import matplotlib.pyplot as plt
from viewCNN import viewCNNparameters
# from trainCNNAllShoeGrp import cnn

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.misc import imread, imresize


##############Train the CNN##########################


class trainCNNZap:

    def trainCNNModel(self, sess=None):

        tr, trl, ts, tsl = self.manageData()
        self.getTrainingGraph()
        self.trainModel(tr, trl, ts, tsl, sess)


    def manageData(self):
        #Read images and presprocess them
        images = []
        idx = 0
        labels = []
        for r, d, files in os.walk("/media/Models/UTZAP50k/SampleData"):
            for fileN in files:
                fName = r+"/"+fileN
                image = io.imread(fName)
                if len(image.shape)==3:
                    image_resized = resize(image, (40, 40))
                    image_resized = image_resized.astype('float32')
                    images.append(image_resized)
                    labels.append(idx)
            if len(files)>0:
                idx = idx+1
        #    images_reverse.append(image_resized[:, ::-1])

        # for i in range(50):
        #     io.imshow(images[i])
        #     io.show()

        # generate hot encoded categories
        saleOneHotEncode = self.dense_to_one_hot(np.array(labels))
        saleOneHotEncode = saleOneHotEncode.astype('float32')


        #extract test data
        imageData_test = []
        lblData_test = []
        randomIds = np.random.randint(9000, size=100)
        for n in range(100):
            imageData_test.append(images[randomIds[n]])
            lblData_test.append(saleOneHotEncode[randomIds[n]])
            del images[randomIds[n]]
        #    del images_reverse[randomIds[n]]
            saleOneHotEncode = np.delete(saleOneHotEncode, (randomIds[n]), axis=0)

        return images, saleOneHotEncode, imageData_test, lblData_test

        # Convert class labels from scalars to one-hot vectors
    def dense_to_one_hot(self, labels_dense):
        num_classes = 10
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        index_one = (index_offset + labels_dense.ravel()).astype(int)
        labels_one_hot.flat[index_one] = 1

        return labels_one_hot

    #saleOneHotEncode = np.concatenate([saleOneHotEncode,saleOneHotEncode])
    #images = images + images_reverse


    def getTrainingGraph(self):

        with tf.name_scope('input'):
            # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
            self.X = tf.placeholder(tf.float32, [None, 40, 40, 3])
            # correct answers will go here
            self.Y_ = tf.placeholder(tf.float32, [None, 10])
            # variable learning rate
            lr = tf.placeholder(tf.float32)

        # three convolutional layers with their channel counts, and a
        # fully connected layer (tha last layer has 10 softmax neurons)
        #J = 8
        K = 8  # first convolutional layer output depth
        L = 12  # second convolutional layer output depth
        M = 20  # third convolutional layer
        N = 200
          # fully connected layer


        with tf.name_scope('conv1'):
            with tf.name_scope('weights'):
                self.W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
            with tf.name_scope('biases'):
                self.B1 = tf.Variable(tf.ones([K])/10)
            with tf.name_scope("activation") as scope:
                stride = 1  # output is 28x28
                self.Y1 = tf.nn.relu(tf.nn.conv2d(self.X, self.W1, strides=[1, stride, stride, 1], padding='SAME') + self.B1)

        ## pool1
        #P1 = tf.nn.max_pool(Y1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

        with tf.name_scope('conv2'):
            with tf.name_scope('weights'):
                self.W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
            with tf.name_scope('biases'):
                self.B2 = tf.Variable(tf.ones([L])/10)
            with tf.name_scope("activation") as scope:
                stride = 2  # output is 14x14
                self.Y2 = tf.nn.relu(tf.nn.conv2d(self.Y1, self.W2, strides=[1, stride, stride, 1], padding='SAME') + self.B2)

        with tf.name_scope('conv3'):
            with tf.name_scope('weights'):
                self.W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
            with tf.name_scope('biases'):
                self.B3 = tf.Variable(tf.ones([M])/10)
            with tf.name_scope("activation") as scope:
                stride = 2  # output is 7x7
                self.Y3 = tf.nn.relu(tf.nn.conv2d(self.Y2, self.W3, strides=[1, stride, stride, 1], padding='SAME') + self.B3)

        ## pool1
        #R2 = tf.nn.max_pool(Y3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')

        # reshape the output from the third convolution for the fully connected layer

        with tf.name_scope('FClayer1'):
            with tf.name_scope('weights'):
                self.W4 = tf.Variable(tf.truncated_normal([10 * 10 * M, N], stddev=0.1))
            with tf.name_scope('biases'):
                self.B4 = tf.Variable(tf.ones([N])/10)
            with tf.name_scope("activation") as scope:
                self.YY = tf.reshape(self.Y3, shape=[-1, 10 * 10 * M])
                self.Y4 = tf.nn.relu(tf.matmul(self.YY, self.W4) + self.B4)

        with tf.name_scope('FClayer2'):
            with tf.name_scope('weights'):
                self.W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
            with tf.name_scope('biases'):
                self.B5 = tf.Variable(tf.ones([10])/10)
            with tf.name_scope("activation") as scope:
                self.YY4 = tf.nn.dropout(self.Y4, 0.5)
                self.Ylogits = tf.matmul(self.YY4, self.W5) + self.B5
                self.Y = tf.nn.softmax(self.Ylogits)


        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y_)
            with tf.name_scope('total'):
                self.cross_entropy = tf.reduce_mean(self.cross_entropy)*100
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        # accuracy of the trained model, between 0 (worst) and 1 (best)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        # training step, the learning rate is a placeholder
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(0.01).minimize(self.cross_entropy)

        self.saver = tf.train.Saver()



    def trainModel(self, images, saleOneHotEncode, imageData_test, lblData_test, sess):

        init = tf.initialize_all_variables()
        sess.run(init)

        #visualization of the activations and filters
        activation0 = np.full((40,40),255, dtype='uint8')
        image=np.full((40,40,3),255, dtype='uint8')
        imageLst = []
        imageLst.append(image)
        for p in range(35):
            imageLst.append(activation0)

        plotCNNParams = viewCNNparameters(0,0,imageLst)


        #train model in batches of 100
        for i in range(1000):

            # generate batch of 100 randomly
            randomIds = np.random.randint(len(images), size=100)
            imageData = []
            lblData = []
            for p in range(100):
                imageData.append(images[randomIds[p]])
                lblData.append(saleOneHotEncode[randomIds[p]])
            lblData = np.reshape(lblData, (100, 10))

            #training
            _, ac, c, w1, w2, y1, y2, y3, y = sess.run([self.train_step, self.accuracy, self.cross_entropy, self.W1, self.W2, self.Y1, self.Y2, self.Y3, self.Y], {self.X: imageData, self.Y_: lblData})
            print(str(i) + "accuracy" +str(ac) + "Y" +str(y[0])+ "Y_"+str(lblData[0])+ " loss: " + str(c) ) #+ "w max: " + str(np.max(w1))+ "w min: " + str(np.min(w1)))

            #validation on test data
            if i%10==0:
                a, c = sess.run([self.accuracy, self.cross_entropy], {self.X: imageData_test, self.Y_: lblData_test})
                print(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))


            #visualization of the activations and filters
            imageLst = []
            imageLst.append(np.uint8(imageData[0]*255))

            activation =[]
            activation2 =[]
            for j in range(8):
                act1 = y1[0, :, : , j]
                act1 = np.reshape(act1, (40, 40))
                x_min = np.min(act1)
                x_max = np.max(act1)
                kernel1 = ((act1 - x_min)*255) / (x_max - x_min)
                activation.append(np.uint8(kernel1))
                imageLst.append(np.uint8(kernel1))

            for j in range(9):
                act2 = y2[0, :, : , j]
                act2 = np.reshape(act2, (20, 20))
                x_min = np.min(act2)
                x_max = np.max(act2)
                kernel1 = ((act2 - x_min)*255) / (x_max - x_min)
                activation2.append(np.uint8(kernel1))
                imageLst.append(np.uint8(kernel1))

            for j in range(9):
                act2 = y3[0, :, : , j]
                act2 = np.reshape(act2, (10, 10))
                x_min = np.min(act2)
                x_max = np.max(act2)
                kernel1 = ((act2 - x_min)*255) / (x_max - x_min)
                activation2.append(np.uint8(kernel1))
                imageLst.append(np.uint8(kernel1))

            for j in range(8):
                B = w1[:, :, : , j]
                x_min = np.min(B)
                x_max = np.max(B)
                kernel1 = ((B - x_min)*255) / (x_max - x_min)
                imageLst.append(kernel1)

            imageLst.append(activation0)
            plotCNNParams.clearPlot()
            plotCNNParams.updateImgLst(imageLst)
            plotCNNParams.plotFigure()

        self.saver.save(sess, '/home/savedModel/Zappo4/ZAP_model.ckpt')


    def predictShoeCategory(self, imgs=None, sess=None):

    ##################### predict class for new image#########################

        ckpt = tf.train.get_checkpoint_state('/home/savedModel/Zappo')
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        feed_dict = {self.X : imgs}
        features, predictions = sess.run([self.Y4, self.Y], feed_dict)
        print(predictions)

        return predictions

    def getShoeCategoryFeature(self, imgs=None, sess=None):

    ##################### predict class for new image#########################

        ckpt = tf.train.get_checkpoint_state('/home/savedModel/Zappo')
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        feed_dict = {self.X : imgs}
        features, predictions = sess.run([self.Y4, self.Y], feed_dict)
        print(predictions)

        return features

    def findMatchingShoes(self, sess=None):
        # Read images and presprocess them
        images = []
        images_ori = []
        for r, d, files in os.walk("/media/Models/CNN_Retail/Retail_image_test/SideTiltedView"):
            for fileN in files:
                fName = r + "/" + fileN
                image = io.imread(fName)
                images_ori.append(image)
                image_resized = resize(image, (40, 40))
                image_resized = image_resized.astype('float32')
                images.append(image_resized[:, ::-1])

        ###only subcate
        cnnZap = trainCNNZap()
        cnnZap.getTrainingGraph()
        feat = cnnZap.predictShoeCategory(images, sess)

        ###all shoe groups
        # cnnZap = cnn()
        # feat = cnnZap.getImageFeatures(images)

        dd = pdist(feat, 'euclidean')

        ss = squareform(dd)

        ###test for image 1
        testIdx = 9   #1

        distMax0 = np.array(ss[:, testIdx])

        distMax0[testIdx] = 100000

        minIdx1 = np.argmin(distMax0)
        distMax0[minIdx1] = 100000
        minIdx2 = np.argmin(distMax0)
        distMax0[minIdx2] = 100000
        minIdx3 = np.argmin(distMax0)
        distMax0[minIdx3] = 100000

        ####view images

        fig = plt.figure()

        g1 = fig.add_subplot(1, 4, 1)
        g1.grid(False)
        g1.imshow(images_ori[testIdx])
        g2 = fig.add_subplot(1, 4, 2)
        g2.grid(False)
        g2.imshow(images_ori[minIdx1])
        g3 = fig.add_subplot(1, 4, 3)
        g3.grid(False)
        g3.imshow(images_ori[minIdx2])
        g4 = fig.add_subplot(1, 4, 4)
        g4.grid(False)
        g4.imshow(images_ori[minIdx3])

        plt.show()
        plt.pause(100)



