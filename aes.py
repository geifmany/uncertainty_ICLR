from models import cifar10vgg as cifar10
from models import cifar100vgg as cifar100

import numpy as np
from utils import uncertainty_tools as utils

MODELS ={"cifar100": cifar100.Cifar100Vgg, "cifar10": cifar10.Cifar10Vgg}

def aes_sr(model_name, model_cls,k=10, train=True, first_epoch=100, last_epoch=249):

    nets = list(np.linspace(first_epoch, last_epoch, k))
    nets = [int(round(a)) for a in nets]


    model = model_cls(train=train, filename=model_name)
    results = []
    for net in nets:
        results.append(model.predict_val_checkpoint(checkpoint=net))
    baseline_pred = results[-1]

    aes_pred = np.mean(np.array(results),0)

    f_pred = np.argmax(baseline_pred, 1)
    confidence = aes_pred[np.arange(aes_pred.shape[0]), f_pred]

    errors = np.argmax(baseline_pred, 1) != np.argmax(model.y_test, 1)

    curve,aurc,eaurc = utils.RC_curve(errors,confidence)
    baseline_curve, baseline_aurc, baseline_eaurc = utils.RC_curve(errors,np.max(baseline_pred,1))
    return eaurc, baseline_eaurc


def aes_MC(model_name, model_cls, k=10, train=False, first_epoch=100, last_epoch=249, mc_iterations=100):
    nets = list(np.linspace(first_epoch, last_epoch, k))
    nets = [int(round(a)) for a in nets]
    model = model_cls(train=train, filename=model_name)
    mc_uncertainty = []
    baseline_pred = (model.predict_val_checkpoint(checkpoint=nets[-1]))
    f_pred = np.argmax(baseline_pred, 1)

    for net in nets:
        mc_uncertainty.append(utils.MC_dropout(model, checkpoint=net, mc_iterations=mc_iterations))


    aes_pred = np.mean(np.array(mc_uncertainty), 0)

    confidence = 1/aes_pred[np.arange(aes_pred.shape[0]), f_pred]

    errors = f_pred != np.argmax(model.y_test, 1)


    curve,aurc,eaurc = utils.RC_curve(errors,confidence)

    confidence_baseline = 1/mc_uncertainty[-1][np.arange(baseline_pred.shape[0]), f_pred]
    baseline_curve, baseline_aurc, baseline_eaurc = utils.RC_curve(errors,confidence_baseline)
    return eaurc, baseline_eaurc



def aes_NN_dist(model_name, model_cls, k=10, train=False, first_epoch=100, last_epoch=249):
    nets = list(np.linspace(first_epoch, last_epoch, k))
    nets = [int(round(a)) for a in nets]
    model = model_cls(train=train, filename=model_name)
    NN_uncertainty = []
    baseline_pred = (model.predict_val_checkpoint(checkpoint=nets[-1]))
    f_pred = np.argmax(baseline_pred, 1)

    for net in nets:
        activations_train, activation_test = model.predict_act_checkpoints(checkpoint=net)
        uncertainty = utils.nn_uncertainty(activations_train,activation_test, model.y_train, f_pred, num_classes=30)
        NN_uncertainty.append(uncertainty)


    confidence = np.mean(np.array(NN_uncertainty), 0)

    errors = f_pred != np.argmax(model.y_test, 1)


    curve,aurc,eaurc = utils.RC_curve(errors,confidence)
    baseline_curve, baseline_aurc, baseline_eaurc = utils.RC_curve(errors,NN_uncertainty[-1])
    return eaurc, baseline_eaurc


if __name__ == '__main__':
    results = {}
    # Results for Cifar10 :
    #this line is only for training the model on first run
    aes_sr("uncertainty_cifar10", MODELS["cifar10"], k=1, train=True)

    for k in [10]:
        if k not in results.keys():
            results[k] = {}
        results[k]["SR"] = aes_sr("uncertainty_cifar10", MODELS["cifar10"], k=k, train=False)
        results[k]["MC"] = aes_MC("uncertainty_cifar10", MODELS["cifar10"], k=k, train=False, mc_iterations=100)
        print(results)
        results[k]["NN-dist"] = aes_NN_dist("uncertainty_cifar10", MODELS["cifar10"], k=k, train=False)

    print(results)

    # this line is only for training the model on first run
    aes_sr("uncertainty_cifar100__0", MODELS["cifar100"], k=1, train=True)

    results = {}
    # Results for Cifar100 :
    for k in [10]:
        if k not in results.keys():
            results[k]={}
        results[k]["SR"] = aes_sr("uncertainty_cifar100__0", MODELS["cifar100"],k=k, train=False)
        results[k]["MC"] = aes_MC("uncertainty_cifar100__0", MODELS["cifar100"],k=k, train=False, mc_iterations=100)
        print(results)
        results[k]["NN-dist"] = aes_NN_dist("uncertainty_cifar100__0", MODELS["cifar100"],k=k, train=False)

    print(results)
