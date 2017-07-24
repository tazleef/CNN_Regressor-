#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from keras.utils import np_utils, generic_utils
from scipy import misc

##================== PREPARE DATA ============================================##
sess = tf.InteractiveSession()
# X_train, y_train, X_val, y_val, X_test, y_test = \
#                                 tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
#
# nb_classes=10;

csv = np.loadtxt(open("D:\LEIDEN\Applicator\csv_gt.csv", "rb"), delimiter=",")
csv=csv/20;

y_train = csv[0:8000]#np.argmax(np.zeros([8000,1], dtype= np.int), axis=-1)
y_val = csv[8000:9000]#np.argmax(np.zeros([1000,1], dtype= np.int), axis=-1)
y_test = csv[9000:10000]#np.argmax(np.zeros([1000,1], dtype= np.int), axis=-1)
#

# nb_classes=0
# y_train = np_utils.to_categorical(y_train, nb_classes)
# y_val = np_utils.to_categorical(y_val, nb_classes)
# y_test = np_utils.to_categorical(y_test, nb_classes)


# y_train = to_categorical(y_train)
# y_valid = to_categorical(y_valid)
# y_test = to_categorical(y_test)



#original image
X_train = np.empty([8000,64,64,1], dtype= np.float)

for num in range(0, 8000):
    # image= misc.imread('D:\LEIDEN\Applicator\model_64.jpg')

    image= misc.imread('D:\LEIDEN\Applicator\OnlyRotAug/'+ str(num+1) + '.jpg')
    image = misc.imresize(image, [64, 64])
    # model= misc.imread('D:/LEIDEN/Applicator/model_28.jpg')
    # model=misc.imresize(model,[100,92])

    X_train[num,:,:,0]=image / 255

X_test = np.empty([1000,64,64,1], dtype= np.float)

for num in range(0, 1000):
    # image= misc.imread('D:\LEIDEN\Applicator\model_64.jpg')

    image= misc.imread('D:\LEIDEN\Applicator\OnlyRotAug/'+ str(num+8001) + '.jpg')
    image = misc.imresize(image, [64, 64])
    # model= misc.imread('D:/LEIDEN/Applicator/model_28.jpg')
    # model=misc.imresize(model,[100,92])

    X_test[num,:,:,0]=image/255


X_val = np.empty([1000,64,64,1], dtype= np.float)

for num in range(0, 1000):
    # image= misc.imread('D:\LEIDEN\Applicator\model_64.jpg')

    image= misc.imread('D:\LEIDEN\Applicator\OnlyRotAug/'+ str(num+9001) + '.jpg')
    image = misc.imresize(image, [64, 64])
    # model= misc.imread('D:/LEIDEN/Applicator/model_28.jpg')
    # model=misc.imresize(model,[100,92])

    X_val[num,:,:,0]=image/255

# y_test[:]=0
# y_val[:]=0
# y_train[:]=0












#
# def pad_distort_im_fn(x):
#     """ Zero pads an image to 40x40, and distort it.
#
#     Examples
#     ---------
#     x = pad_distort_im_fn(X_train[0])
#     print(x, x.shape, x.max())
#     tl.vis.save_image(x, '_xd.png')
#     tl.vis.save_image(X_train[0], '_x.png')
#     """
#     b = np.zeros((40, 40, 1))
#     o = int((40-28)/2)
#     b[o:o+28, o:o+28] = x
#     x = b
#     x = tl.prepro.rotation(x, rg=30, is_random=True, fill_mode='constant')
#     # x = tl.prepro.shear(x, 0.05, is_random=True, fill_mode='constant')
#     x = tl.prepro.shift(x, wrg=0.25, hrg=0.25, is_random=True, fill_mode='constant')
#     # x = tl.prepro.zoom(x, zoom_range=[0.95, 1.05], is_random=True, fill_mode='constant')
#     return x
#
# def pad_distort_ims_fn(X):
#     """ Zero pads images to 40x40, and distort them. """
#     X_40 = []
#     for X_a, _ in tl.iterate.minibatches(X, X, 50, shuffle=False):
#         X_40.extend(tl.prepro.threading_data(X_a, fn=pad_distort_im_fn))
#     X_40 = np.asarray(X_40)
#     return X_40

# create dataset with size of 40x40 with distortion
# X_train_40 = pad_distort_ims_fn(X_train)
# X_val_40 = pad_distort_ims_fn(X_val)
# X_test_40 = pad_distort_ims_fn(X_test)

# tl.vis.save_images(X_test[0:32], [4, 16], '_imgs_original.png')
# tl.vis.save_images(X_test_40[0:32], [4, 16], '_imgs_distorted.png')

##================== DEFINE MODEL ============================================##
batch_size = 64
x = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[batch_size, ], name='y_')

def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        nin = InputLayer(x, name='in')
        ## 1. Localisation network
        # use MLP as the localisation net
        # nt = FlattenLayer(nin, name='tf')
        # nt = DenseLayer(nt, n_units=20, act=tf.nn.tanh, name='td1')
        # nt = DropoutLayer(nt, 0.8, True, is_train, name='tdrop')
        # # you can also use CNN instead for MLP as the localisation net
        nt = Conv2d(nin, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc1')
        #32x32x16
        nt = Conv2d(nt, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')
        # 16x16x16
        nt = Conv2d(nt, 32, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc3')
        # 8X8X32
        nt = Conv2d(nt, 32, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc4')
        # 4X4X32
        nt = Conv2d(nt, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc5')
        nt =  tl.layers.PoolLayer(nt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool = tf.nn.max_pool, name ='pool_layer1')
        # 2X2X32
        nt = Conv2d(nt, 64, (2, 2), (1, 1), act=tf.nn.relu, padding='SAME', name='tc6')
        nt = tl.layers.PoolLayer(nt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                                 name='pool_layer2')

        # 1X1X64
        nt = Conv2d(nt, 1, (1, 1), (1, 1),act=None, padding='SAME', name='tc7')
        # 1X1X1
        n=nt
        # n = FlattenLayer(nt, name='f')



        #
        # nt = Conv2d(nt, 8, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')
        # # 2. Spatial transformer module (sampler)
        # n = SpatialTransformer2dAffineLayer(nin, nt, out_size=[40, 40], name='ST')
        # s = n
        # ## 3. Classifier
        # n = Conv2d(n, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='c1')
        # n = Conv2d(n, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='c2')
        # n = FlattenLayer(n, name='f')
        # n = DenseLayer(n, n_units=1024, act=tf.nn.relu, name='d1')
        # n = DenseLayer(n, n_units=1, act=tf.identity, name='do')
    ## 4. Cost function and Accuracy
        y = n.outputs
        cost = tl.cost.mean_squared_error(y, y_)
        #correct_prediction = tf.equal(y, y_)
        #acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        subs=tf.subtract(y,y_)
        acc=tf.divide(subs,y)
        acc1 = tf.reduce_mean(tf.cast(acc, tf.float32))

        return n, cost, acc1, y

net_train,  cost, _, y_temp = model(x, is_train=True, reuse=False)
net_test,  cost_test, acc ,y_temp_test= model(x, is_train=False, reuse=True)

##================== DEFINE TRAIN OPS ========================================##
n_epoch = 500
learning_rate = 0.00001
print_freq = 10

train_params = tl.layers.get_variables_with_name('STN', train_only=True, printable=True)
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-05, use_locking=False).minimize(cost, var_list=train_params)

##================== TRAINING ================================================##
tl.layers.initialize_global_variables(sess)
net_train.print_params()
net_train.print_layers()

for epoch in range(n_epoch):
    start_time = time.time()
    ## you can use continuous data augmentation
    # for X_train_a, y_train_a in tl.iterate.minibatches(
    #                             X_train, y_train, batch_size, shuffle=True):
    #     X_train_a = tl.prepro.threading_data(X_train_a, fn=pad_distort_im_fn)
    #     sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})
    ## or use pre-distorted images (faster)
    for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=False):
           err, ac ,y_temp_value= sess.run([cost_test, acc, y_temp], feed_dict={x: X_train_a, y_: y_train_a})
           train_loss = train_loss+ err; train_acc = train_acc+ ac; n_batch =n_batch + 1
        #print("Y: %f" % y_temp_value)
        print("   train loss: %f" % (train_loss/ n_batch))
        # print("   train acc: %f" % (train_acc/ n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(
                                    X_val, y_val, batch_size, shuffle=False):
             err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
             val_loss += err; val_acc += ac; n_batch += 1
        print("   val loss: %f" % (val_loss/ n_batch))
        # print("   val acc: %f" % (val_acc/ n_batch))

        # net_train.print_params()
        # net_test.print_params()
        # net_trans.print_params()
        # print('save images')
        # trans_imgs = sess.run(net_trans.outputs, {x: X_test_40[0:64]})
        # tl.vis.save_images(trans_imgs[0:32], [4, 16], '_imgs_distorted_after_stn_%s.png' % epoch)

##================== EVALUATION ==============================================##
print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(
                            X_test, y_test, batch_size, shuffle=False):
    err, ac = sess.run([ cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
    test_loss += err; test_acc += ac; n_batch += 1

print("   test loss: %f" % (test_loss/n_batch))
print("   test acc: %f" % (test_acc/n_batch))
