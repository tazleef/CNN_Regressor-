
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from keras.utils import np_utils, generic_utils
from scipy import misc
from tensorlayer.layers import set_keep

##================== PREPARE DATA ============================================##

LOGDIR = 'D:\LEIDEN\CNN_Regressor\CNN_Regressor-/LOG/'
#tf.reset_default_graph()
sess = tf.InteractiveSession()




csv = np.loadtxt(open("D:\LEIDEN\Applicator\csv_gt2.csv", "rb"), delimiter=",")
#csv[:,0]=csv[:,0]/180;

y_train = csv[0:8000]#np.argmax(np.zeros([8000,1], dtype= np.int), axis=-1)
y_val = csv[8000:9000]#np.argmax(np.zeros([1000,1], dtype= np.int), axis=-1)
y_test = csv[9000:10000]#np.argmax(np.zeros([1000,1], dtype= np.int), axis=-1)


#original image
X_train = np.empty([8000,64,64,1], dtype= np.float)
for num in range(0, 8000):
    image= misc.imread('D:\LEIDEN\Applicator\OnlyRotAug2/'+ str(num+1) + '.jpg')
    image = misc.imresize(image, [64, 64])
    X_train[num,:,:,0]=image / 255


X_val = np.empty([1000,64,64,1], dtype= np.float)
for num in range(0, 1000):
    image= misc.imread('D:\LEIDEN\Applicator\OnlyRotAug2/'+ str(num+8001) + '.jpg')
    image = misc.imresize(image, [64, 64])
    X_val[num,:,:,0]=image/255



X_test = np.empty([1000,64,64,1], dtype= np.float)
for num in range(0, 1000):
    image= misc.imread('D:\LEIDEN\Applicator\OnlyRotAug2/'+ str(num+9001) + '.jpg')
    image = misc.imresize(image, [64, 64])
    X_test[num,:,:,0]=image/255





##================== DEFINE MODEL ============================================##
batch_size = 200
x = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[batch_size, 3], name='y_')

def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        nt = InputLayer(x, name='in')

        nt = Conv2d(nt, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc1')
        nt = Conv2d(nt, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')

        #32x32x16
        nt = Conv2d(nt, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc3')
        # 16x16x16
        nt = Conv2d(nt, 32, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc4')
        # 8X8X32
        nt = Conv2d(nt, 32, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc5')
        # 4X4X32
        nt = Conv2d(nt, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='VALID', name='tc6')
        # nt =  tl.layers.PoolLayer(nt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool = tf.nn.max_pool, name ='pool_layer1')
        # 2X2X32
        nt = Conv2d(nt, 64, (2, 2), (1, 1), act=tf.nn.relu, padding='VALID', name='tc7')
        # nt = tl.layers.PoolLayer(nt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
        #                          name='pool_layer2')
        nt = Conv2d(nt, 32, (1, 1), (1, 1), act=tf.nn.relu, padding='VALID', name='tc8')
        # 1X1X64
        nt = Conv2d(nt, 3, (1, 1), (1, 1),act=None, padding='SAME', name='tc9')
        # 1X1X1

        nt = FlattenLayer(nt, name='f')
        # n = DenseLayer(n, n_units=3, act=None, name='do')
    ## 4. Cost function and Accuracy
        y = nt.outputs
        cost = tl.cost.mean_squared_error(y, y_, 'cost')
        tf.summary.scalar('cost',cost)






        loss1 = tf.losses.mean_squared_error(y[:,0], y_[:,0])
        tf.summary.scalar('loss1', loss1)

        loss2 = tf.losses.mean_squared_error(y[:,1], y_[:,1])
        tf.summary.scalar('loss2', loss2)

        loss3 = tf.losses.mean_squared_error(y[:,2], y_[:,2])
        tf.summary.scalar('loss3', loss3)









        #correct_prediction = tf.equal(y, y_)
        #acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # subs=tf.subtract(y,y_)
        # acc1=tf.divide(subs,y)
        # #acc1 = tf.reduce_mean(tf.cast(acc, tf.float32))
        # params = nt.all_params


        return nt, cost,  y, loss1, loss2 , loss3

net_train,  cost, y_temp,ls1,ls2,ls3 = model(x, is_train=True, reuse=False)
net_test,  cost_test ,y_temp_test, lsz1,lsz2,lsz3= model(x, is_train=False, reuse=True)

##================== DEFINE TRAIN OPS ========================================##

n_epoch = 1000
learning_rate = 0.001
print_freq = 10

train_params = tl.layers.get_variables_with_name('STN', train_only=True, printable=True)
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-05, use_locking=False).minimize(cost, var_list=train_params)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(LOGDIR + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(LOGDIR + '/test')
tf.global_variables_initializer().run()




##================== TRAINING ================================================##
tl.layers.initialize_global_variables(sess)
net_train.print_params()
net_train.print_layers()
train=1;

if(train ==1):
    for epoch in range(n_epoch):
        start_time = time.time()

    #    tl.files.load_ckpt(sess=sess, mode_name='model.ckpt',var_list=model.all_params, save_dir='model', printable=True)

        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
            sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_loss1,train_loss2,train_loss3, n_batch = 0, 0, 0,0,0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=False):
               sum, err,y_temp_value, lss1,lss2,lss3= sess.run([merged, cost, y_temp, ls1,ls2,ls3], feed_dict={x: X_train_a, y_: y_train_a})
               train_writer.add_summary(sum, epoch)


               # accu1 = 1 - np.mean(abs(ac[:, 0]));
               # accu2 = 1 - np.mean(abs(ac[:, 1]));
               # accu3 = 1 - np.mean(abs(ac[:, 2]));


               train_loss = train_loss+ err; n_batch =n_batch + 1; train_loss1 += lss1;train_loss2 += lss2;train_loss3 += lss3;
            #print("Y: %f" % y_temp_value)


            # test_writer = tf.summary.FileWriter(LOGDIR + '/test')
            # tf.global_variables_initializer().run() #Otherwise you encounter this error : Attempting to use uninitialized value conv2d/kerne
            # sess.run(tf.global_variables_initializer())
            #summ = tf.summary.merge_all()
            #train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
            # val_writer = tf.summary.FileWriter(LOGDIR + '/val')
            print("   train loss: %f" % (train_loss/ n_batch))

            print("   rot loss: %f" % (train_loss1 / n_batch))
            print("   trans1 loss: %f" % (train_loss2 / n_batch))
            print("   trans2 loss: %f" % (train_loss3 / n_batch))

            #train_writer.add_summary(train_loss/n_batch, epoch)
            # print("   train acc1: %f" % (train_acc1/ n_batch))
            # print("   train acc2: %f" % (train_acc2/ n_batch))
            # print("   train acc3: %f" % (train_acc3 / n_batch))

            val_loss, val_loss1, val_loss2, val_loss3, n_batch = 0, 0, 0,0,0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=False):
                 summary, err, lss1,lss2,lss3 = sess.run([merged, cost_test,lsz1,lsz2,lsz3], feed_dict={x: X_train_a, y_: y_train_a})
                 test_writer.add_summary(summary, epoch)
                 # accu1 = 1 - np.mean(abs(ac[:, 0]));
                 # accu2 = 1 - np.mean(abs(ac[:, 1]));
                 # accu3 = 1 - np.mean(abs(ac[:, 2]));



                 val_loss += err; val_loss1 +=lss1; val_loss2 += lss2; val_loss3 += lss3;


                 n_batch += 1

            # val_writer.add_summary(val_loss/n_batch, epoch)
            print("   val loss: %f" % (val_loss/ n_batch))
            # print("   val acc1: %f" % (val_acc1/ n_batch))
            # print("   val acc2: %f" % (val_acc2 / n_batch))
            # print("   val acc3: %f" % (val_acc3 / n_batch))
            #

            print("   rot loss: %f" % (val_loss1 / n_batch))
            print("   trans1 loss: %f" % (val_loss2 / n_batch))
            print("   trans2 loss: %f" % (val_loss3 / n_batch))

            #tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=net.all_params, save_dir='model', printable=True)
            # saver = tf.train.Saver()
            # save_path = saver.save(sess, "D:\LEIDEN/non-rigid\Spatial-Transformer-Nets-master/model.ckpt")
            # #tl.files.save_npz(model.all_params, name='model.npz')
            # net_train.print_params()
            # net_test.print_params()
            # net_trans.print_params()
            # print('save images')
            # trans_imgs = sess.run(net_trans.outputs, {x: X_test_40[0:64]})
            # tl.vis.save_images(trans_imgs[0:32], [4, 16], '_imgs_distorted_after_stn_%s.png' % epoch)
            # saver = tf.train.Saver()
            # save_path = saver.save(sess, "./model.ckpt")
            # print("Model saved in file: %s" % save_path)
            #

#         # You can also save the parameters into .npz file.

# You can only save one parameter as follow.
# tl.files.save_npz([network.all_params[0]] , name='model.npz')
# Then, restore the parameters as follow.
# load_params = tl.files.load_npz(path='', name='model.npz')
# tl.files.assign_params(sess, load_params, network)

# In the end, close TensorFlow session.
#sess.close()

##================== EVALUATION ==============================================##
#tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', var_list=model.all_params, save_dir='model', is_latest=False, printable=True)
#tl.files.load_ckpt(sess=sess, var_list=net.all_params, save_dir='model', printable=True)

if(train==1):
    tl.files.save_npz(net_train.all_params, name='model_tensorboard.npz')

# npz=np.load('model.npz')
#
# params = []
# for val in sorted( npz.items() ):
#     print("  Loading %s" % str(val[1].shape))
#     params.append(val[1])
#
# tl.files.assign_params(sess, params, train_op)
#

# restore model from .npz
if(train==0):
    load_params = tl.files.load_npz(path='', name='model_tensorboard.npz')
    tl.files.assign_params(sess, load_params, net_test)



print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(
                            X_test, y_test, batch_size, shuffle=False):
    err,  y_temp_test = sess.run([ cost_test,  y_temp], feed_dict={x: X_test_a, y_: y_test_a})
    test_loss += err; #test_acc += ac;
    n_batch += 1


#test_writer.add_summary(s, i)
print("   test loss: %f" % (test_loss/n_batch))
#print("   test acc: %f" % (test_acc/n_batch))
