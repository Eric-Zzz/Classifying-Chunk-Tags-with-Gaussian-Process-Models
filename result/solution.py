####Group member: z5127440(YIFAN ZHAO)
##                z5137606(ZHEN YUAN)




import numpy as np
import gpflow
from scipy import sparse
import os
import os.path
import time
from sklearn.decomposition import IncrementalPCA

time1 = time.time()


def get_train_file(file, length):
    L = []
    f = open(file)
    lines = f.readlines()
    for i in lines:
        H = []
        H = i.split()
        M = []
        for j in H:
            k = int(j)
            M.append(k)

        L.append(M)
    L = np.array(L)
    # generate sparse matrix
    N = sparse.lil_matrix((length, 2035523))
    for i in range(length):
        s = L[:, 0] == i + 1
        for j in L[s][:, 1]:
            N[i, j] = 1

    N = N.A
    f.close()
    return N


def get_test_file(file, length):
    L = []
    f = open(file)
    lines = f.readlines()
    for i in lines:
        H = []
        H = i.split()
        M = []
        for j in H:
            k = int(j)
            M.append(k)

        L.append(M)
    L = np.array(L)
    #generate sparse matrix
    N = sparse.lil_matrix((length, 2035523))
    for i in range(length):
        s = L[:, 0] == i + 1
        for j in L[s][:, 1]:
            N[i, j] = 1

    N = N.A
    f.close()
    return N


def get_y_file(file):
    f = open(file)
    lines = f.readlines()
    L = []
    for i in lines:
        L.append(int(i))
    L = np.array(L)
    f.close()
    return len(lines), sparse.csr_matrix(L)


def get_y_result(file):
    f = open(file)
    lines = f.readlines()
    L = []
    for i in lines:
        L.append(int(i))
    L = np.array(L)
    f.close()
    return len(lines), L


def get_accurancy(L1, L2):
    length = len(L1)
    count = 0
    for i in range(length):
        if L1[i] == L2[i]:
            count = count + 1

    if count == 0:
        return 0
    else:
        return count / length


def print_acc(L):
    sum = 0
    for i in L:
        sum = sum + i
    return sum / len(L)


def print_predict(f, mean):
    L = mean.tolist()
    for i in range(len(L)):
        for j in range(len(L[i])):
            f.write(str(L[i][j]))
            f.write(",")
        f.write("\n")
    f.write("\n")

    return


def get_length(inputfile):
    filesize = os.path.getsize(inputfile)
    blocksize = 1024
    with open(inputfile, 'rb') as f:
        last_line = ""
        if filesize > blocksize:
            maxseekpoint = (filesize // blocksize)
            f.seek((maxseekpoint - 1) * blocksize)
        elif filesize:
            f.seek(0, 0)
        lines = f.readlines()
        if lines:
            lineno = 1
            while last_line == "":
                last_line = lines[-lineno].strip()
                lineno += 1
        print(last_line)
        length = last_line.decode().split(" ")
        print(length)
        print(length[0])
        return int(length[0])


def predict_model(x, y, z):
    acc_list = []
    # cross validation
    f = open("predict.txt", "a+")

    for i in range(x, y, z):
        x_test = 'conll_train/' + str(i) + '.x'
        y_test = 'conll_train/' + str(i) + '.y'
        length_sentence, y_test = get_y_result(y_test)
        x_test = get_test_file(x_test, length_sentence)
        x_test = reducer.transform(x_test)
        mean, var = m.predict_y(x_test)
        label = np.argmax(mean, axis=1)
        print("predict-label", label)
        print("org-label", y_test)
        acc = get_accurancy(label, y_test)
        acc_list.append(acc)
        print(f"the {i} acc is ", i, acc)
        print_predict(f, np.log(mean))
    f.close()
    print(print_acc(acc_list))
    return


def do_predict(x, y):
    f = open("predict.txt", "a+")

    for i in range(x, y):
        file_x_test = 'conll_test_features/' + str(i) + '.x'
        length_sentence = get_length(file_x_test)
        file_x_test = get_test_file(file_x_test, length_sentence)
        file_x_test = reducer.transform(file_x_test)
        mean, var = m.predict_y(file_x_test)
        print_predict(f, np.log(mean))
    f.close()
    return


if __name__ == '__main__':
    x_txt = 'conll_train/1.x'
    y_txt = 'conll_train/1.y'
    length_sentence, file_y_list = get_y_file(y_txt)
    file_x_list = get_train_file(x_txt, length_sentence)
    reducer = IncrementalPCA(n_components=10, batch_size=500)
    reducer.fit(file_x_list)
    file_x_list = reducer.transform(file_x_list)
    for i in range(2, 8937, 88):
        x_txt = 'conll_train/' + str(i) + '.x'
        y_txt = 'conll_train/' + str(i) + '.y'
        length_sentence, labels = get_y_file(y_txt)
        file_y_list = sparse.hstack((file_y_list, labels))
        x_vector = get_train_file(x_txt, length_sentence)
        x_vector = reducer.transform(x_vector)
        file_x_list = np.vstack((file_x_list, x_vector))
    print(file_x_list.shape)
    print(file_y_list.shape)

    kenel = gpflow.kernels.RBF(file_x_list.shape[1], lengthscales=1.0, variance=2.0)
    m = gpflow.models.SVGP(
        np.float64(file_x_list), np.float64(file_y_list.A.T), kern=kenel, likelihood=gpflow.likelihoods.MultiClass(23),
        Z=np.float64(file_x_list[::3]), num_latent=23, whiten=True, q_diag=True)
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, maxiter=10000)
    print('Model', m)

    #predict_model(2, 8937, 800)


    do_predict(8937,10949)

    time2 = time.time()
    print('Times：' + str(time2 - time1) + 's')
