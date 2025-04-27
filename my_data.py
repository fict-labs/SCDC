import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


dataset_name = 'stl10'  #['fmnist', 'reuters10k', 'stl10', 'cifar10']
use_our_loss = 1   # 1: SCDC, -1: DEC, -2: IDEC, -3: K-means
run_mode = 4 

weight_params = {'reuters10k': [0.0001, 0.2], 'cifar10': [0.01, 0.01], 'fmnist':[0.01, 0.1], 'stl10': [0.001, 0.2]}

#----------- fixed parameters -------------#
dim_z = 10
ae_weight_name = 'ae_weight_' + str(use_our_loss) + '.h5'
cluster_weight_name = 'cluster_weight_' + str(use_our_loss) + '.h5'
kmeans_trials = 20
log_train = 1 #0, 1: write loss values to a log file

import os

def compute_mean_patch(selected_patches):
    """
    Compute the mean image patch from a list of image patches.    
    Args:
        selected_patches (list of numpy arrays): List of cropped image patches (H, W, C)        
    Returns:
        mean_patch (numpy array): Mean image patch
    """
    if len(selected_patches) == 0:
        print("No patches available for computing mean.")
        return None

    # Convert to float and stack
    patch_stack = np.stack(selected_patches[0:5]).astype(np.float32)

    # Compute mean across patches
    mean_patch = np.mean(patch_stack, axis=0)

    return mean_patch.astype(np.uint8)  # Convert back to uint8 for visualization

def get_path_name(id=0):
    img_path = 'codewords/' + dataset_name + '/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    method = 'SCDC'
    if use_our_loss == -1:
        method = 'DEC'
    elif use_our_loss == -2:
        method = 'IDEC'
    elif use_our_loss == -3:
        method = 'Kmeans'
    
    img_path += method + '_id' + str(id)

    return img_path  

def closest_k_nodes(node, nodes, k):

    nodes = np.asarray(nodes)    
    node = np.asarray(node)     

    dist_2 = np.sum((nodes - node)**2, axis=1)

    #https://medium.com/@yurybelousov/the-beauty-of-python-or-how-to-get-indices-of-n-maximum-values-in-the-array-d362385794ef
    #Use np.argpartition. It does not sort the entire array. It only guarantees that the kth element is in sorted position and all smaller elements will be moved before it. Thus the first k elements will be the k-smallest elements butnote that these may not be in sorted order.

    #idx = heapq.nsmallest(k, range(len(dist_2)), dist_2.__getitem__)
    #idx = heapq.nsmallest(k, range(len(dist_2)), dist_2.take)
    
    idx = np.argsort(dist_2)[:k]  #sort entire the array
    #idx = np.argpartition(dist_2, k)[:k] #return k smalerst items but the order of k items is not correct
    id = np.argmin(dist_2)

    #note that there are many tests that id != idx[0] = the reason is that if there are many equal distances 
    # => the indexes of these items may be in a random order
    # the kmeans does not have this problem because the distances are computed on original space, so less equal distances

    return idx, id  

def show_images(codeword, answer_list, k_label):
        
    img_path = get_path_name()
    img_path += '_codeword' + str(k_label)

    if dataset_name == 'fmnist':
        x, y, = load_data(dataset_name)

    elif dataset_name == 'stl10':    
        data_path = '../data/stl'
        x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)    
        x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)   
        
        x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
        x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
        x = np.concatenate((x1, x2))       

    elif dataset_name == 'cifar10':  # => DON'T USE as the split are not correct as the original time
        from keras.datasets import cifar10
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        x = np.concatenate((train_x, test_x))       
    else:        
        return
    
    selected_list = []        
    for i in range(0, len(answer_list)):
        img = x[answer_list[i]]
        selected_list.append(img)        
    
    codeword_img = compute_mean_patch(selected_list)
    plt.imsave(img_path + '.png', codeword_img)    
    selected_list.append(codeword_img)

    return selected_list

def show_voronoi(X, codewords, labels, id=0):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from sklearn.manifold import TSNE
    plt.clf()  # Clears the current figure

    img_path = get_path_name(id)
    num_clusters = len(codewords)

    X, X_test, labels, y_test = train_test_split(X, labels, train_size=4000, random_state=11) 

    # Combine data points and centroids for t-SNE
    X_combined = np.vstack((X, codewords))

    # Apply t-SNE once on the combined dataset
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', random_state=13).fit_transform(X_combined)

    # Separate transformed data points and centroids
    X = X_embedded[:-num_clusters]  # Data points
    codewords = X_embedded[-num_clusters:]  # Centroids

    fig, ax = plt.subplots(figsize=(8, 6))        
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.75, s=12, label="Data Points", edgecolors='none')     
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.savefig(img_path+'_tse.pdf', format="pdf", bbox_inches="tight")

    vor = Voronoi(codewords)        
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=1.5)        
    plt.scatter(codewords[:, 0], codewords[:, 1], c='red', marker='X', s=120, label="Codewords")        
    plt.legend(loc="upper left")

    plt.savefig(img_path+'_voronoi.pdf', format="pdf", bbox_inches="tight")
    

def show_heatmap(X, codewords, labels, id=0):
    
    plt.clf()  # Clears the current figure
    img_path = get_path_name(id)
    num_clusters = len(codewords)
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(codewords, cmap="coolwarm", annot=False, xticklabels=[f"Feature {i+1}" for i in range(X.shape[1])],
                yticklabels=[f"Codeword {i+1}" for i in range(num_clusters)])
    #plt.title("Heatmap of Codewords")
    plt.xlabel("Feature Index")
    plt.ylabel("Codewords")
    plt.legend()
    plt.savefig(img_path+'_heatmap.pdf', format="pdf", bbox_inches="tight")
    #plt.show()
    return

def show_profile(X, codewords, labels, id=0):
    from sklearn.preprocessing import MinMaxScaler
    img_path = get_path_name(id)
    num_clusters = len(codewords)
    plt.clf()  # Clears the current figure

    # Radar chart setup
    scaler = MinMaxScaler()
    codewords_scaled = scaler.fit_transform(codewords)
    
    labels = [f"Feature {i+1}" for i in range(X.shape[1])]
    num_vars = len(labels)

    # Compute the angle for each feature
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # angles += angles[:1]  # Close the radar chart loop

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles.append(angles[0])  # Ensure polygon closure

    # Initialize radar chart
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    # Plot each cluster
    for i in range(num_clusters):
        values = codewords_scaled[i].tolist()  # Convert to list
        values.append(values[0])  # Close the loop

        ax.plot(angles, values, linewidth=2, label=f'Cluster {i+1}')
        ax.fill(angles, values, alpha=0.1)

    # Formatting the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    #ax.set_yticklabels([])  # Hide radial labels for clarity
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.savefig(img_path+'_profile.pdf', format="pdf", bbox_inches="tight")
    #plt.show()

def show_Silhouette(X, labels, id=0):
    from sklearn.metrics import silhouette_samples
    plt.clf()  # Clears the current figure

    img_path = get_path_name(id)

    silhouette_vals = silhouette_samples(X, labels)
    sns.histplot(silhouette_vals, bins=20, kde=True)
    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--", label="Mean score")
    plt.legend()
    #plt.title("Silhouette Analysis of Codewords")    

    plt.xlabel("Silhouette value on " + dataset_name.upper())
    plt.savefig(img_path+'_silhouette.pdf', format="pdf", bbox_inches="tight")
    #plt.show()

def save_BoVW(features, labels, ds_centers, id=0):

    print('+ codewords.shape = ', ds_centers.shape)
        
    show_voronoi(features, ds_centers, labels, id)
    show_heatmap(features, ds_centers, labels, id)
    show_Silhouette(features, labels, id)
    show_profile(features, ds_centers, labels, id)

    n_images = 10
    m = len(ds_centers)
    print('+ m = ',m)

    fig, axes = plt.subplots(m, n_images+1, figsize=(10, 8))

    img_path = get_path_name()
    for k in range(0, m):
        x = ds_centers[k]
        idx_list, id = closest_k_nodes(x, features, n_images)   
        ds = show_images(x, idx_list, k)
        for j in range(len(ds)):
            axes[k, j].imshow(ds[j])
            axes[k, j].axis('off') # Hide axes

    plt.savefig(img_path + '_knn.png', format="png", bbox_inches="tight")
    return 1

def draw_tSNE(X, labels, sname):
    
    X, X_test, labels, y_test = train_test_split(X, labels, train_size=4000, random_state=11) 
    
    X_emb = TSNE(n_components=2, learning_rate='auto', init='random', random_state=13).fit_transform(X)
    FS = (6, 6)
    fig, ax = plt.subplots(figsize=FS)
    file_name = 'plots/' + dataset_name + '_' + sname              
    
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=labels, cmap='viridis', alpha=0.75, s=12, label="Data Points", edgecolors='none')    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.legend()
    plt.savefig(file_name, format="pdf", bbox_inches="tight")

def extract_resnet_features(x):
    #from keras.preprocessing.image import img_to_array, array_to_img
    from keras.utils import img_to_array, array_to_img        
    #from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input    

    im_h = 224
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(im_h, im_h, 3), classifier_activation="none")
    #model.summary()

    print('+ extract_resnet_features...')
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
    x = tf.keras.applications.resnet50.preprocess_input(x)  # data - 127. #data/255.#
    features = model.predict(x)
    print('+ Features shape = ', features.shape)
    return features

def load_cifar10(data_path='./data/cifar10'):
        
    import os.path
    if os.path.exists(data_path + '/cifar10_features.npy') and os.path.exists(data_path + '/cifar10_labels.npy'):
        return np.load(data_path + '/cifar10_features.npy'), np.load(data_path + '/cifar10_labels.npy')
    
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    # extract full features 
    features = np.zeros((60000, 2048))
    for i in range(60):
        idx = range(i*1000, (i+1)*1000)
        print("+ The %dth 1000 samples: " % i)
        features[idx] = extract_resnet_features(x[idx])    
    
    #get 50% random dataset as it is too big for loading in memory
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.5, random_state=11)

    x = X_train
    y = y_train
    print('+ cifar10: x_size = ', x.shape[0])
    print('+ cifar10: y_size = ', y.shape[0])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    x = MinMaxScaler().fit_transform(x)

    #save features
    np.save(data_path + '/cifar10_features.npy', x)
    print('+++  features saved to ' + data_path + '/cifar10_features.npy')

    np.save(data_path + '/cifar10_labels.npy', y)
    print('+++  labels saved to ' + data_path + '/cifar10_labels.npy')

    return x, y

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('+ load_mnist', x.shape)
    return x, y

def load_fashion_mnist(data_path='./data/fmnist'):
  
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x = x.reshape((x.shape[0], -1)) #x = (768)
    x = np.divide(x, 255.)

   
    print('+ Fashion fMNIST samples', x.shape)
    return x, y

def load_stl10(data_path='./data/stl', use_resnet = 1):
    use_resnet = 1 #0, 1    

    #NOTE: use_resnet = 0 => never because input dim is too big
    file_name_x = data_path + '/stl_features.npy'
    file_name_y = data_path + '/stl_labels.npy'
    if(use_resnet == 1):
        file_name_x = data_path + '/stl_features_resnet50.npy'
        file_name_y = data_path + '/stl_labels_resnet50.npy'

    import os
    #if features are ready, return them
    if os.path.exists(file_name_x) and os.path.exists(file_name_y):
        return np.load(file_name_x), np.load(file_name_y)
    
    # get labels
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))
    
    N = len(y) 
    print("+ load_stl10() => N1 = ", N) #13000
  
    # get data
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)    
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    
    # extract features 
    
    if(use_resnet == 1):
        x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
        x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
        x = np.concatenate((x1, x2)).astype(float)

        N = len(x) 
        print("+ load_stl10() => N2 = ", N)  #13000
        features = np.zeros((N, 2048))
        for i in range(13):
            idx = range(i*1000, (i+1)*1000)
            print("+ The %dth 1000 samples: " % i)
            features[idx] = extract_resnet_features(x[idx])

        # scale to [0,1]
        from sklearn.preprocessing import MinMaxScaler
        features = MinMaxScaler().fit_transform(features)

        # save features
        np.save(file_name_x, features)
        np.save(file_name_y, y)
        return features, y
    else:
        x = np.concatenate((x1, x2)).astype(float)
        x = np.divide(x, 255.)

        # save features
        np.save(file_name_x, x)
        np.save(file_name_y, y)

        return x, y
    
def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    
    from sklearn.model_selection import train_test_split
    x = x.astype(np.float32)    
    print('+ reuters: x_size_full = ', x.shape[0])
    print('+ reuters: y_size_full = ', y.shape[0])

    #extract random 10,000 samples
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=10000, random_state=13)
    x = X_train
    y = y_train
    print('+ reuters: x_size = ', x.shape[0])
    print('+ reuters: y_size = ', y.shape[0])
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
  
    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})

def load_reuters_ours(data_path='./data/reuters'):
    import os
    #best_param for this: corr_wt, reg_wt, aff_wt = ['reuters10k_ours': [0.01, 0.1, 1.0]]
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))

    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'), allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('+++++++++ REUTERSIDF10K samples', x.shape))
    return x, y

def load_reuters(data_path='./data/reuters'):

    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k_edesc.npy')):
        print('making reuters idf features')
        # make_reuters_data(data_path)
        return load_reuters_ours()
        
    data = np.load(os.path.join(data_path, 'reutersidf10k_edesc.npy'),allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y

def load_data(dataset_name):
    if dataset_name == 'fmnist':
        return load_fashion_mnist()
    elif dataset_name == 'mnist':
        return load_mnist() 
    elif dataset_name == 'stl10':
        return load_stl10()  
    elif dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        return load_reuters()   
    else:
        print('Not defined for loading', dataset_name)
        exit(0)