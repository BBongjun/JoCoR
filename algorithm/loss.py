import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# def kl_loss_compute(pred, soft_targets, reduce=True):

#     kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

#     if reduce:
#         return torch.mean(torch.sum(kl, dim=1))
#     else:
#         return torch.sum(kl, 1)




# def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):

#     loss_pick_1 = F.cross_entropy(y_1, t, reduction = 'none') * (1-co_lambda)
#     loss_pick_2 = F.cross_entropy(y_2, t, reduction = 'none') * (1-co_lambda)
#     loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()


#     ind_sorted = np.argsort(loss_pick.data)
#     loss_sorted = loss_pick[ind_sorted]

#     remember_rate = 1 - forget_rate
#     num_remember = int(remember_rate * len(loss_sorted))

#     pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

#     ind_update=ind_sorted[:num_remember]

#     # exchange
#     loss = torch.mean(loss_pick[ind_update])

#     return loss, loss, pure_ratio, pure_ratio


# gradient vanishing 문제때문에 작은 값 더하니까, nan 증상 사라짐.
def kl_divergence_with_logits(log_probs, targets):
    kl = torch.sum(targets * (torch.log(targets + 1e-10) - log_probs), dim=1)
    if torch.any(torch.isnan(kl)):
        print("NaN detected in KL divergence")
    return kl

def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):
    y_1_log_probs = F.log_softmax(y_1, dim=1)
    y_2_log_probs = F.log_softmax(y_2, dim=1)
    y_1_probs = F.softmax(y_1, dim=1)
    y_2_probs = F.softmax(y_2, dim=1)

    if torch.any(torch.isnan(y_1_log_probs)) or torch.any(torch.isnan(y_2_log_probs)):
        print("NaN detected in log_probs")
    if torch.any(torch.isnan(y_1_probs)) or torch.any(torch.isnan(y_2_probs)):
        print("NaN detected in probs")

    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_2 = F.cross_entropy(y_2, t, reduction='none')

    if torch.any(torch.isnan(loss_1)) or torch.any(torch.isnan(loss_2)):
        print("NaN detected in Cross Entropy")

    kl_1_2 = kl_divergence_with_logits(y_1_log_probs, y_2_probs)
    kl_2_1 = kl_divergence_with_logits(y_2_log_probs, y_1_probs)

    loss_pick = (loss_1 + loss_2) / 2 + co_lambda * (kl_1_2 + kl_2_1)

    if torch.any(torch.isnan(loss_pick)):
        print("NaN detected in final loss")

    ind_sorted = torch.argsort(loss_pick)
    num_remember = int((1 - forget_rate) * len(loss_pick))
    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]]) / float(num_remember)
    ind_update = ind_sorted[:num_remember]

    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio