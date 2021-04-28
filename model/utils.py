import torch
import torch.nn.functional as F
from torch.autograd import Variable

def prelu(_x,scope='',device=None):
    #_alpha=Variable(torch.rand(_x.size()[-1])).to(device)
    _alpha = torch.full([_x.size()[-1]], fill_value=0.1).to(device)
    value = torch.zeros(_x.size()[0], _x.size()[1]).to(device)
    #_alpha=tf.get_variable("",shape=_x.get_shape()[-1],dtype=_x.dtype,initializer=tf.constant_initializer(0,1))
    # gt=torch.gt(_x,0.0).tolist()
    # comp=gt[0][0]
    # value=torch.zeros(_x.size()[0],_x.size()[1]).to(device)
    # if comp:
    #     max=_x
    #     min=value
    # else:
    #     max = value
    #     min = _x
    return torch.maximum(value,_x)+_alpha*torch.minimum(value,_x)
    #return tf.maxinum(0,0,_x)+_alpha*tf.minimum(0.0,_x)
def makeHiddenForBound(linear_weight, states, dtime, particle_num, hidden_dim, total_num):

    c, cb, d, o = states

    c = c.unsqueeze(dim=2).expand(
        particle_num, hidden_dim, total_num)
    cb = cb.unsqueeze(dim=2).expand_as(c)
    d = d.unsqueeze(dim=2).expand_as(c)
    o = o.unsqueeze(dim=2).expand_as(c)

    dtime = dtime.unsqueeze(dim=1).unsqueeze(dim=2).expand_as(c)
    linear_weight = linear_weight.unsqueeze(dim=0).expand_as(c)

    cgap = c - cb

    indices_inc_0 = (cgap > 0.0) & (linear_weight < 0.0)
    indices_inc_1 = (cgap < 0.0) & (linear_weight > 0.0)

    cgap[indices_inc_0] = 0.0
    cgap[indices_inc_1] = 0.0

    cy = cb + cgap * torch.exp(-d * dtime)
    hy = o * torch.tanh(cy)

    return hy


def makeHiddenForLambda(states, dtime):
    c, cb, d, o = states
    c = c.unsqueeze(dim=1)
    cb = cb.unsqueeze(dim=1)
    d = d.unsqueeze(dim=1)
    o = o.unsqueeze(dim=1)
    dtime = dtime.unsqueeze(dim=2)

    cy = cb + (c - cb) * torch.exp(-d * dtime)
    hy = o * torch.tanh(cy)

    return hy


def combine_dict(*dicts):
    """
    :param dict dicts:
    :rtype: dict
    """
    rst = dict()
    for single_dict in dicts:
        for key, value in single_dict.items():
            assert key not in rst
            rst[key] = value
    return rst


############mimn部分
def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc

'''
def din_attention(query, facts, attention_size, mask=None, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = torch.cat(facts, 2)
        print("query_size mismatch")
        query = torch.cat([
            query,
            query,
        ], )

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all

    if mask is not None:
        mask = tf.equal(mask, tf.ones_like(mask))
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Activation
    if softmax_stag:
        scores = F.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))

    if return_alphas:
        return output, scores
    return output
'''