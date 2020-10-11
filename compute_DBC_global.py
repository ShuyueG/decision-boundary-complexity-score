'''
These codes are to compute global decision boundary complexity (DBC) scores.
It includes codes to find/generate global adversarial sets and compute global DBC.

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


# pre-process images for CNN input
# input:    image file name
# output:   input for CNN model
def imForCnn(imgfile):
    img = Image.open(imgfile)
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = np.array(img)
    img = img / 255

    return np.expand_dims(img, axis=0)


# Algorithm: to find an adversarial example
# input:    two images belong to different classes; the classifier model
# output:   adversarial example and its output from the classifier model
def boundary_case(im1, im2, classifier):
    best_v = 0
    for p in np.arange(0, 1, 1 / 256):  # step is 1/256
        mix = p * im1 + (1 - p) * im2
        v = classifier.predict(mix)
        if np.abs(best_v - 0.5) > np.abs(v - 0.5):
            best_v = v
            best_p = p

    best_mix = best_p * im1 + (1 - best_p) * im2
    return best_mix, best_v


if __name__ == '__main__':

    # load the classifier model
    MOD = load_model('theCNN.h5')

    # load images
    cat_imgs = glob.glob(r'data\train\cats\*.jpg')
    dog_imgs = glob.glob(r'data\train\dogs\*.jpg')

    # number of pairs == the class has minimum number of images
    Num = np.min([len(cat_imgs), len(dog_imgs)])

    ##############################################################
    # find adversarial examples and then compute the global DBC 
    ##############################################################
    for k in range(50):  # repeat 50 times to get 50 global DBC values

        print('\n', 'k= ', str(k))  # repeat time

        # random select but keep same length
        idx1 = np.random.permutation(len(cat_imgs))[0:Num]
        idx2 = np.random.permutation(len(dog_imgs))[0:Num]

        img_stack = []  # save adversarial examples

        ### find adversarial examples ###
        for i in range(Num):
            # show the progress
            mystr = 'Now  ' + str(100 * i // Num) + '%'
            print('\r', mystr, end='', flush=True)

            # pre-process images for CNN input
            im1 = imForCnn(cat_imgs[idx1[i]])
            im2 = imForCnn(dog_imgs[idx2[i]])

            # get outputs from the classifier model
            p_1 = MOD.predict(im1)
            p_2 = MOD.predict(im2)

            if np.min([p_1, p_2]) < 0.5 and np.max([p_1, p_2]) >= 0.5:  # check the two images are in different classes
                best_mix, _ = boundary_case(im1, im2, MOD)  # get an adversarial example

                # save adversarial examples
                if img_stack == []:
                    img_stack = best_mix
                else:
                    img_stack = np.vstack((best_mix, img_stack))

        ### compute the global DBC ###
        bd_img = img_stack.reshape(img_stack.shape[0], -1)  # flatten: (n,a,b,3)->(n,3ab)

        # compute eigenvalues
        pca = PCA()
        pca.fit(bd_img)
        cplx = pca.explained_variance_ratio_

        # compute the global DBC
        cplx_ratio = entropy(cplx, base=2) / np.log2(len(cplx))

        # record results
        with open('global_DBCs.txt', 'a') as fw:
            fw.write('\n' + str(cplx_ratio))

        del bd_img, img_stack, pca  # free memory
