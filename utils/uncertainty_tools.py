import numpy as np

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
def MC_dropout(model, checkpoint=249, mc_iterations=100, dropout_rate=0.5):
    temp_mc = []
    for i in range(mc_iterations):
        temp_mc.append(model.predict_val_checkpoint(checkpoint=checkpoint, dropout=dropout_rate))
    return (np.var(np.array(temp_mc), 0))


def RC_curve(residuals, confidence):

    curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov/ m, acc / len(temp1)))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        acc = acc-residuals[idx_sorted[i]]
        curve.append((cov / m, acc /(m-i)))
    AUC = sum([a[1] for a in curve])/len(curve)
    err = np.mean(residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    EAURC = AUC-kappa_star_aurc
    return curve, AUC, EAURC



def nn_uncertainty(rep_train,rep_test,y_train,y_hat,k=500,num_classes=10):
    #rep_train a matrix of mXn m samples and n features of the represetnation of the training set
    #rep_test a matrix of mXn m samples and n features of the represetnation of the testset
    m = rep_test.shape[0]
    neigh = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    y_label = np.argmax(y_train, 1)

    neigh.fit(rep_train, y_train)
    dist, id = neigh.kneighbors(rep_test, return_distance=True)
    uncertainty = np.zeros([m,1])
    for i in range(len(rep_test)):
        dist_temp = dist[i]

        all_dist =np.sum(np.exp(-1*dist_temp))
        inclass = y_label[id[i,:]]==y_hat[i]
        inclass_dist = np.sum(np.exp(-1*dist_temp[inclass]))

        uncertainty[i,0]=inclass_dist/all_dist

    return uncertainty[:,0]
