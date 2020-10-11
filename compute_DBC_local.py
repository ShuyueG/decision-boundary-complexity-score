'''
These codes are to compute local decision boundary complexity (DBC) scores.
It includes codes to find/generate local adversarial sets and compute DBC.
The process is speeded up by the divide-and-conquer method, which uses binary search.

Dataset:            Any two-class classification dataset, e.g. cat and dog dataset
Model:              A trained classifier; its outputs are in [0, 1]

Related paper:      Analysis of Generalizability of Deep Neural Networks Based on the Complexity of Decision Boundary
                    [In press] International Conference on Machine Learning and Applications (ICMLA), 2020
                    https://arxiv.org/abs/2009.07974

By:                 Shuyue Guan
                    https://shuyueg.github.io/
'''

import glob

import numpy as np
from PIL import Image
from keras.models import load_model
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# pre-process images for CNN input
# input:    image file name
# output:   input for CNN model
def imForCnn(imgfile):
    img = Image.open(imgfile)
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = np.array(img)
    img = img / 255

    return np.expand_dims(img, axis=0)


# classifier output of the example img on the line segment between img1 and img2
# input:    mixture rate (Lambda): 0<p<1; two images belong to different classes; the classifier model
# output:   classifier output of the new img
def pred_p(p, im1, im2, classifier):
    mix = p * im1 + (1 - p) * im2
    return classifier.predict(mix)


# Algorithm: to find an mixture rate (Lambda) of an adversarial example using binary search
# input:    two images belong to different classes;
#           the classifier model; minimum step (Epsilon); 
#           minimum and maximum mixture rate (Lambda)
# output:   the mixture rate (Lambda) of an adversarial example
def find_half_rec(im1, im2, classifier, e, l, r):
    v_l = pred_p(l, im1, im2, classifier)
    v_r = pred_p(r, im1, im2, classifier)

    half = (l + r) / 2

    if np.min([v_l, v_r]) >= (0.5 - e) and np.max([v_l, v_r]) <= (0.5 + e):
        return half

    if np.min([v_l, v_r]) < 0.5 and np.max([v_l, v_r]) > 0.5:
        if pred_p(half, im1, im2, classifier) > 0.5:
            return find_half_rec(im1, im2, classifier, e, l, half)
        else:
            return find_half_rec(im1, im2, classifier, e, half, r)


# check every pair of data are on different side
def two_sides_true(set1, set2, classifier):
    a = classifier.predict(set1)
    b = classifier.predict(set2)
    for va in a:
        for vb in b:
            if not (np.min([va, vb]) < 0.5 and np.max([va, vb]) > 0.5):
                return False
    return True


if __name__ == '__main__':

    # load the classifier model
    MOD = load_model('theCNN.h5')

    loc_num = 30  # the number of nearest neighbors

    # load images
    cat_imgs = glob.glob(r'data\train\cats\*.jpg')
    dog_imgs = glob.glob(r'data\train\dogs\*.jpg')

    # pre-process images for CNN input
    C1 = []
    C2 = []
    for f in range(len(cat_imgs)):
        C1.extend(imForCnn(cat_imgs[f]))

    C1 = np.array(C1)

    for f in range(len(dog_imgs)):
        C2.extend(imForCnn(dog_imgs[f]))

    C2 = np.array(C2)
    flat_C2 = C2.reshape(C2.shape[0], -1)

    # number of pairs == the class has minimum number of images
    Num = np.min([len(C1), len(C2)])

    # find neighbors in C2
    neigh = NearestNeighbors(n_neighbors=loc_num)  # dim=loc_num
    neigh.fit(flat_C2)

    ##############################################################
    # find adversarial examples and then compute the local DBC 
    ##############################################################
    # repeat to get local DBC values
    k = 0
    Total = 6000  # total pairs
    while (k < Total):

        # random select but keep same length
        idx1 = np.random.permutation(len(C1))[0:Num]
        idx2 = np.random.permutation(len(C2))[0:Num]

        for i in range(Num):
            loop_flag = True

            # select two images from different classes
            im1 = np.expand_dims(C1[idx1[i]], axis=0)
            im2 = np.expand_dims(C2[idx2[i]], axis=0)

            # find neighbors
            flat_im2 = np.expand_dims(flat_C2[idx2[i]], axis=0)
            im2_neigh_idx = neigh.kneighbors(flat_im2, return_distance=False)
            im2_neigh = C2[im2_neigh_idx]
            im2_neigh = np.squeeze(im2_neigh)

            # find adversarial examples
            if two_sides_true(im1, im2_neigh, MOD):  # pair of data are on different sides
                img_stack = []  # save adversarial examples
                for one_im2_neigh in im2_neigh:  # for each neighbor

                    # to find an mixture rate (Lambda) of an adversarial example
                    best_v = find_half_rec(im1, one_im2_neigh, MOD, 1 / 256, 0, 1)  # if return NONE, switch the [1,0] <-> [0,1]
                    try:
                        best_mix = best_v * im1 + (
                                    1 - best_v) * one_im2_neigh  # convert Lambda to an adversarial example
                    except:
                        print('\n', type(Exception).__name__)
                        loop_flag = False
                        break

                    # save adversarial examples
                    if len(img_stack) == 0:
                        img_stack = best_mix
                    else:
                        img_stack = np.vstack((best_mix, img_stack))

                if not loop_flag: break

                ### compute the  DBC ###  
                bd_img = img_stack.reshape(img_stack.shape[0], -1)  # flatten: (n,a,b,3)->(n,3ab)

                # compute eigenvalues
                pca = PCA()
                pca.fit(bd_img)
                cplx = pca.explained_variance_ratio_

                # compute the DBC
                cplx_ratio = entropy(cplx, base=2) / np.log2(len(cplx))

                # record results
                with open('local_DBCs.txt', 'a') as fw:
                    fw.write('\n' + str(cplx_ratio))

                if k > Total:  # end
                    break
                else:  # show the progress
                    k += 1
                    mystr = 'Now  ' + str(100 * k // Total) + '%'
                    print('\r', mystr, end='', flush=True)

                del bd_img, img_stack, pca, im2_neigh  # free memory
