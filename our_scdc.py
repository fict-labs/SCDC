
# our HongDuc Deep Clustering (HDC)

from time import time
import numpy as np
import keras.backend as K
from keras.layers  import Layer, InputSpec
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics
import os
import tensorflow as tf
from keras import layers
from sklearn.metrics import silhouette_score, silhouette_samples
import csv
import pandas as pd

from my_data import load_data, save_BoVW, dataset_name, use_our_loss, dim_z, ae_weight_name, cluster_weight_name, kmeans_trials, run_mode, load_data, log_train, weight_params,draw_tSNE

KL = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
from datetime import datetime

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def write_to_file(file_name, data):    
    file1 = open(file_name, "a") 
    file1.write('\n' + data)
    file1.close()

def autoencoder_dense(input_dim=784, pre_training=True, act='relu', init='glorot_uniform', mse_loss_wt=0.15):
    
    # input
    x = Input(shape=(input_dim,), name='input')
    h = x

    h1 = Dense(500, activation=act, kernel_initializer=init, name='encoder_1')(h)        
    h2 = Dense(500, activation=act, kernel_initializer=init, name='encoder_2')(h1)    
    h3 = Dense(2000, activation=act, kernel_initializer=init, name='encoder_3')(h2)    
    h = h3
    
    # hidden layer    
    h = Dense(dim_z, kernel_initializer=init, name='our_hidden_layer')(h)  # hidden layer
    
    y = h        
    y3 = Dense(2000, activation=act, kernel_initializer=init, name='decoder_3')(y)
    y2 = Dense(500, activation=act, kernel_initializer=init, name='decoder_2')(y3)
    y1 = Dense(500, activation=act, kernel_initializer=init, name='decoder_1')(y2)
    y = y1  

    # output    
    y = Dense(input_dim, kernel_initializer=init, name='ae_output')(y)   
    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')    

class CenterLoss(Layer):
    """
    Code adapted from: https://github.com/XifengGuo/DEC-keras/blob/master/DEC.py
    """

    def __init__(self, n_clusters,  center_loss_wt=0.15,  weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(CenterLoss, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.center_loss_wt = center_loss_wt
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        self.P = None

    def set_P(self, p):
        self.P = p
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    
    def pairwise_sqd_distance(self, X, batch_size):

        tiled = tf.tile(tf.expand_dims(X, axis=1), tf.stack([1, batch_size, 1]))
        tiled_trans = tf.transpose(tiled, perm=[1,0,2])
        diffs = tiled - tiled_trans
        sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)

        return sqd_dist_mat

    def make_t_students(self, z, batch_size, alpha=1.0, show=0):

        sqd_dist_mat = self.pairwise_sqd_distance(z, batch_size)
        if show == 1:
            print('+ len(sqd_dist_mat): ', sqd_dist_mat, len(sqd_dist_mat), len(sqd_dist_mat[0]))  #batch_size x batch_size 

        q = 1.0 / (1.0 + sqd_dist_mat / alpha)
        q **= (alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        q = tf.clip_by_value(q, 1e-10, 1.0)
        
        return q
    
    @staticmethod
    def target_distribution(q):
        #weight = q ** 2 / q.sum(0)
        #return (weight.T / weight.sum(1)).T

        weight = q ** 2 / K.sum(q, axis=0) 
        return K.transpose( K.transpose(weight) / K.sum(weight, axis=1))

    def compute_center_loss(self, z, q):
        #inputs: z (embedded points), q: soft assignments, 
        #output:  spare center loss

        p = self.target_distribution(q) 
        
        d = K.sum((K.square(K.expand_dims(z, axis=1) - self.clusters)), axis=2)                    
        D = layers.Multiply()([p, d]) 
         
        SCL = K.mean(D)         
        return SCL*self.center_loss_wt

    
    def call(self, z_inputs, x_inputs, training=False):
        """ return q_ij computed based on student t-distribution
        """

        d = K.sum((K.square(K.expand_dims(z_inputs, axis=1) - self.clusters)), axis=2)    

        q = 1.0 / (1.0 + (d / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))      
        
        if training == False:  # inference mode            
            print(' + training phase:', training, ', center_wt:', self.center_loss_wt)  
        else: # training mode  
            print(' + training phase:', training, ', center_wt:', self.center_loss_wt)  

            if use_our_loss == 1: #compute center weight
                p = self.target_distribution(q)
                D = layers.Multiply()([p, d]) 
                SCL = K.mean(D)         
                d_center = SCL*self.center_loss_wt                
                self.add_loss(d_center)
        return q
   
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(CenterLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepCluster(object):
    def __init__(self, input_dim, pre_training=True, n_clusters=10, init='glorot_uniform', reg_wt = 0.01, center_loss_wt = 0.25, id=0):

        super(DeepCluster, self).__init__()

        print("----------------------init--------------------------")

        self.id = id                
        self.input_dim = input_dim          
        self.center_loss_wt = center_loss_wt
        self.reg_wt = reg_wt
        self.n_clusters = n_clusters
        self.pre_training = pre_training        

        self.autoencoder, self.encoder = autoencoder_dense(input_dim=self.input_dim, pre_training=self.pre_training, init=init)
        
        z_embed = self.encoder.get_layer(name='our_hidden_layer').output
        cluster_layer = CenterLoss(n_clusters=self.n_clusters, center_loss_wt=self.center_loss_wt, name='cluster_layer')(z_embed, self.encoder.input)                
        self.model = Model(inputs=self.autoencoder.input, outputs=[cluster_layer, self.autoencoder.output], name = 'cluster_model'+str(id))                
    
    def compile(self, optimizer='sgd', loss='kld'):

        if use_our_loss == 1:
            self.model.compile(optimizer=optimizer, loss={'cluster_layer': 'kld', 'ae_output': 'mse'}, loss_weights=[1.0, self.reg_wt]) 
        elif use_our_loss == -1: #DEC            
            self.model.compile(optimizer=optimizer, loss={'cluster_layer': 'kld'})
        elif use_our_loss == -2: #IDEC
            self.model.compile(optimizer=optimizer, loss={'cluster_layer': 'kld', 'ae_output': 'mse'}, loss_weights=[0.1, 1])        
            

        print('====== clustering model ========')
        self.model.summary()   
    
    
    def pretrain(self, x, y=None, optimizer='adam', epochs=150, batch_size=256, save_dir='results/temp', id_trial=0, save_model = 0):
        print('...Pretraining...')   

        self.autoencoder.compile(optimizer=optimizer,  loss='mse')
        self.autoencoder.summary()
        file_test = "results_ae.txt"
        
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y, encoder, autoencoder):
                    self.x = x
                    self.y = y                    
                    self.encoder = encoder
                    self.autoencoder = autoencoder
                    self.acc_best = 0
                    self.acc_last = 0
                    super(PrintACC, self).__init__()        

                def on_epoch_end(self, epoch, logs=None):                    
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return                         
                    
                    features = self.encoder.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, random_state = 5)
                    y_pred = km.fit_predict(features)
                    acc = metrics.acc(self.y, y_pred)
                    self.acc_last = acc
                    
                    if acc > self.acc_best:
                        self.acc_best = acc        
                
                    print(' '*8 + '| acc: %.4f, [acc_best: %.4f], nmi: %.4f|' % (acc, self.acc_best, metrics.nmi(self.y, y_pred)))

                    if(save_model >= 0):
                        file1 = open(file_test,"a")                                 
                        file1.write('\n   + epoch: %d, acc: %.4f, [acc_best: %.4f], nmi: %.4f,' % (epoch, acc, self.acc_best, metrics.nmi(self.y, y_pred)))
                        file1.close()

            cb_print = PrintACC(x, y, self.encoder, self.autoencoder)
            cb.append(cb_print)
        
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")       
        file1 = open(file_test,"a")
        ss = '\n   + id: ' + str(id_trial) + ', acc_last = %.4f, [acc_best = %.4f], epochs = %d, bs = %d' % (cb_print.acc_last, cb_print.acc_best, epochs, batch_size)+ ', time: ' + dt_string
        if(save_model <= 1):
            ss = '\n   ==> id: ' + str(id_trial) + ', acc_last = %.4f, [acc_best = %.4f], epochs = %d, bs = %d' % (cb_print.acc_last, cb_print.acc_best, epochs, batch_size)+ ', time: ' + dt_string + '\n'
        file1.write(ss)
        file1.close()
       
        if(run_mode >= 4):
            self.autoencoder.save_weights(save_dir + '/' + ae_weight_name)         
        self.pretrained = True
        return cb_print.acc_last
        
    
    def load_weights(self, weights):  
        self.model.load_weights(weights) 

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
        
    def cluster_finetuning(self, x, y=None, maxiter=9000, batch_size=128, update_interval=100, save_dir='./results/temp', ae_acc=0, id_trial=0, save_model = 0):

        print('+ Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)
      
        # Step 1: initialize cluster centers using k-means
        t1 = time()                
        kmeans = KMeans(n_clusters=self.n_clusters, n_init = kmeans_trials, random_state = 7)  
        features = self.encoder.predict(x)
        y_pred = kmeans.fit_predict(features)
        self.model.get_layer(name='cluster_layer').set_weights([kmeans.cluster_centers_])
       
        # Step 2: deep clustering
        # logging file       
        if log_train == 1:
            logfile = open(save_dir + '/cluster_log.csv', 'w')
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'loss'])
            logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        acc_best = 0.0
        ite_best = 0
        acc_last = 0.0 
        nmi_best = 0.0   

        for ite in range(int(maxiter)):            
            
            if (run_mode == 5 or run_mode == 6 or run_mode == 3) and use_our_loss == 1:
                if ite == 0 or  ite == 2000 or ite == maxiter-1:
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init = kmeans_trials, random_state = 7) 
                    features = self.encoder.predict(x)               
                    draw_tSNE(features, y, 'plot_t' + str(ite) + '.pdf')
             
            if (run_mode == 6 or run_mode == 3) and ite == maxiter-1:
                kmeans = KMeans(n_clusters=self.n_clusters, n_init = kmeans_trials, random_state = 7)  
                features = self.encoder.predict(x)
                kmeans.fit_predict(features)                
                save_BoVW(features, y, kmeans.cluster_centers_, id=id_trial) 
                break

            if ite % update_interval == 0:
                                    
                q, _ = self.model.predict(x, verbose=0)  
                p = self.target_distribution(q)                                  

                y_pred = q.argmax(1) # evaluate the clustering performance     
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    #loss = np.round(loss['loss'], 5)
                    if log_train == 1:
                        logdict = dict(iter=ite, acc=acc, nmi=nmi, loss=loss)                    
                        logwriter.writerow(logdict)

                    acc_last = acc                  
                    if acc > acc_best:
                        acc_best = acc
                        ite_best = ite
                        nmi_best = nmi
                        
                        if save_model == -3:
                            self.model.save_weights(save_dir + '/' + cluster_weight_name)
                    
                    print(' +===> iter %d: acc = %.4f, [acc_best = %.4f (ae_acc = %.4f), ite_best = %d, center_wt = %.4f, reg_wt = %.4f], nmi = %.4f' % (ite, acc, acc_best, ae_acc, ite_best, self.center_loss_wt, self.reg_wt,  nmi_best), ', our_loss=', use_our_loss)                    

            # train on batch      
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]     
            
            if use_our_loss ==-1:
                loss = self.model.train_on_batch(x=x[idx], y = p[idx]) 
            else:
                loss = self.model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
             
            if ite % 45 == 0:
                print('  + loss: ', loss)

            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
            if index == 0:
                np.random.shuffle(index_array)            

        if log_train == 1:
            logfile.close()

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        file_test = "results_clustering.txt"
        file1 = open(file_test,"a") 
        file1.write('\n   + id: ' + str(id_trial) + ', acc_last = %.4f, [acc_best = %.4f, (ae_acc = %.4f), nmi_best = %.4f, iter_best = %d],  max_iter = %d, bs = %d' % (acc_last, acc_best, ae_acc, nmi_best, ite_best, maxiter, batch_size) + ', time: ' + dt_string)
        file1.close()

        return acc_last


def start_training(x, y, reg_loss_wt=0.01, center_loss_wt = 0.2, id_trial=0, save_model = 0):

    K.clear_session()  #remove all the current models
    input_dim = x.shape[-1]     
    bs = 64  
    
    pretrain_epochs = 30   #pre-training
    clustering_iters = 5000 
    update_interval = 50

    n_clusters = len(np.unique(y))  #reuters10k dataset contains only 4 centroids
    print('+ n_clusters = ', n_clusters)

    pretrain_optimizer = 'adam'  
    clustering_optimizer = SGD(0.01, 0.9)
    init = 'glorot_uniform'    
    
    if dataset_name == 'fmnist':        
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
        update_interval = 100  
        clustering_iters = 5000
        pretrain_epochs = 100        
        clustering_optimizer = SGD(0.01, 0.9)
        pretrain_optimizer = SGD(lr=0.1, momentum=0.9)                                   

    elif dataset_name == 'reuters10k':
        update_interval = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform') 
        pretrain_epochs = 40
        pretrain_optimizer = SGD(lr=0.25, momentum=0.9)  
        clustering_optimizer =  SGD(0.001, 0.9)
        clustering_iters = 5000  
               
    elif dataset_name == 'stl10':
        update_interval = 50       
        pretrain_epochs = 30
        clustering_optimizer = SGD(0.01, 0.9)
        pretrain_optimizer = SGD(lr=1.0, momentum=0.9) 

    elif dataset_name == 'cifar10':        
        update_interval = 80     
        pretrain_epochs = 60
        clustering_optimizer = SGD(0.1, 0.9) 
        pretrain_optimizer = SGD(lr=1.0, momentum=0.9)     
    
    if(use_our_loss < 0): # DEC and IDEC
        #DEC setting: https://github.com/XifengGuo/DEC-keras/blob/master/DEC.py
        init = 'glorot_uniform'
        pretrain_optimizer = 'adam'        
        if dataset_name == 'fmnist':      
            update_interval = 140
            pretrain_epochs = 300              
            clustering_iters = 5000
            init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')            
            clustering_optimizer = SGD(0.01, 0.9)                      
            pretrain_optimizer = SGD(lr=1, momentum=0.9)

        elif dataset_name == 'reuters10k':                        
            init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')                         
            clustering_iters = 5000
            update_interval = 30
            pretrain_epochs = 50
            clustering_optimizer = SGD(0.01, 0.9)                      
            pretrain_optimizer = SGD(lr=1, momentum=0.9)
        
        elif dataset_name == 'stl10':
            update_interval = 30       
            pretrain_epochs = 10     
            clustering_optimizer = SGD(0.01, 0.9)   

        elif dataset_name == 'cifar10':
            update_interval = 80     
            pretrain_epochs = 60
            clustering_optimizer = SGD(0.01, 0.9)
            pretrain_optimizer = SGD(lr=1.0, momentum=0.9)     
        
               
    if save_model <= 1:
        pre_training = True
    else:
        pre_training = False

    dc_model = DeepCluster(input_dim = input_dim, pre_training = pre_training, n_clusters=n_clusters, init=init, reg_wt = reg_loss_wt, center_loss_wt = center_loss_wt) 
    
    ae_acc = 0
    if save_model <= 1:
        ae_acc = dc_model.pretrain(x=x, y=y, optimizer=pretrain_optimizer, epochs=pretrain_epochs, batch_size=bs, save_dir=save_dir, id_trial=id_trial, save_model=save_model)
    else: 
        dc_model.autoencoder.load_weights(save_dir + '/' + ae_weight_name)
        #test the loaded autoencoder model
        kmeans = KMeans(n_clusters=n_clusters, n_init=kmeans_trials, random_state = 3)
        y_pred = kmeans.fit_predict(dc_model.encoder.predict(x))
        if y is not None:
            ae_acc = np.round(metrics.acc(y, y_pred), 5)
            print('------------------------------------------------------------')
            print('     + loading autoencoder model: done!')
            print('     + test model: acc =  %.4f' % (ae_acc))
            print('------------------------------------------------------------')            
    
    if save_model >= 2:        
        dc_model.compile(optimizer=clustering_optimizer)        
        ae_acc = dc_model.cluster_finetuning(x, y=y, maxiter=clustering_iters, batch_size=bs, update_interval=update_interval, save_dir=save_dir, ae_acc=ae_acc, id_trial=id_trial, save_model=save_model)
            
    return ae_acc

def run_experiments(xx, yy, save_dir):
    
    write_to_file("results_ae.txt", '----------------------- run_experiments: dataset = [' + dataset_name + '] -----------------------')
    write_to_file("results_clustering.txt", '------------------------------ run_experiments: dataset = [' + dataset_name + '] ------------------------------')

    N = 10
    if run_mode >= 5:
        N = 1
    w_reg, w_center = weight_params[dataset_name]
    print('++++ weight_params[' + dataset_name + '] = ', w_center, w_center)

    c_method = 'SCDC'
    if use_our_loss == -1:
        c_method = 'DEC'
    elif use_our_loss == -2:
        c_method = 'IDEC'
    elif use_our_loss == -3:
        c_method = 'Kmeans'
        n_clusters = len(np.unique(yy))  
        kmeans = KMeans(n_clusters=n_clusters, n_init = kmeans_trials, random_state = 7)  #seed = constant: to make the initialization deterministics                        
        kmeans.fit_predict(xx)                
        save_BoVW(xx, yy, kmeans.cluster_centers_) 
        return

    file_name = "results_report.txt"
    write_to_file(file_name, '-------------- run: %d trials of model, dataset = [' % (N) + dataset_name + ', w_reg, w_center=' + str(weight_params[dataset_name]) + ', method = ' + c_method + ', use_our_loss=' + str(use_our_loss) + '] --------------')
    
    acc1_sum = 0
    acc2_sum = 0
    nmi1_sum = 0
    nmi2_sum = 0        
    ds_acc2 = []
    ds_nmi2 = []       
    for i in range(N):
        #1. pre-training the author autoencoder
        start_training(x = xx, y = yy, reg_loss_wt = w_reg, center_loss_wt = w_center, id_trial=i, save_model=1)

        #2. fine-tuning the cluster model
        start_training(x = xx, y = yy, reg_loss_wt = w_reg, center_loss_wt = w_center, id_trial=i, save_model=3)

        #read the results
        data = pd.read_csv(save_dir + '/cluster_log.csv')
        ds_acc = data[['acc']]
        ds_nmi = data[['nmi']]
        ds_acc = np.asarray(ds_acc, dtype=float)[0:,0]
        ds_nmi = np.asarray(ds_nmi, dtype=float)[0:,0]

        print(ds_acc)
        print(ds_nmi)

        acc_ae = ds_acc[0]
        acc_cluster = ds_acc[-1]
        nmi_ae = ds_nmi[0]
        nmi_cluster = ds_nmi[-1]
        print('+ accuracy: ',acc_ae, acc_cluster)
        print('+ nmi: ', nmi_ae, nmi_cluster)

        acc1_sum += acc_ae
        acc2_sum += acc_cluster

        nmi1_sum += nmi_ae
        nmi2_sum += nmi_cluster

        ds_acc2.append(acc_cluster)
        ds_nmi2.append(nmi_cluster)
      
        ss = '   + id_trial: %d, AE Accuracy: %.1f, Cluster Accuracy: %.1f, AE-based NMI: %.1f, Cluster-based NMI: %.1f' % (i, acc_ae*100, acc_cluster*100, nmi_ae*100, nmi_cluster*100)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        write_to_file(file_name, ss + ', ' + dt_string)        

    acc1_sum /= N
    acc2_sum /= N
    nmi1_sum /= N
    nmi2_sum /= N

    ds_acc2.sort() 
    ds_nmi2.sort() 
    print(ds_acc2)
    print(ds_nmi2)
    med_acc = ds_acc2[int(len(ds_acc2)/2)]
    med_nmi = ds_nmi2[int(len(ds_nmi2)/2)]

    ss1 = '   ==> Mean Accuracy: AE model = %.1f, Clustering model = %.1f' % (acc1_sum*100, acc2_sum*100)
    ss2 = '   ==> Mean NMI: AE model = %.1f, Clustering model = %.1f' % (nmi1_sum*100, nmi2_sum*100)
    ss3 = '   ==> Median Accuracy = %.1f, Median NMI = %.1f\n' % (med_acc*100, med_nmi*100)    

    write_to_file(file_name, ss1)
    write_to_file(file_name, ss2)
    write_to_file(file_name, ss3)
   
    
if __name__ == "__main__":

    save_dir = 'results/' + dataset_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    xx, yy = load_data(dataset_name)
    print('+ load_data: ', xx.shape ,yy.shape)
    
    if run_mode >= 4: #run 10 experiments to compute the mean performance
        run_experiments(xx, yy, save_dir)