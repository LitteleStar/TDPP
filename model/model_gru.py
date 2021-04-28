import torch
from torch.autograd import Variable
from loss import *

class GRU(torch.nn.Module):
    def __init__(self,n_iid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, MEMORY_SIZE,device=None):
        super(GRU, self).__init__()

        self.batch_size=BATCH_SIZE
        self.seq_len=SEQ_LEN
        self.hidden_size=HIDDEN_SIZE
        self.embedding_dim=EMBEDDING_DIM
        self.interst_num=MEMORY_SIZE
        self.n_iid=n_iid
        self.num_layers=1
        self.mid_batch_ph=None
        self.mid_his_batch_ph=None
        self.mid_neg_batch_ph=None
        self.iid_embeddings_var =torch.nn.Embedding(n_iid, EMBEDDING_DIM) # 将每个item id都给一个1*EMBEDDING_DIM的编码，总共有feature num个(算 user item cate
        self.n_negtive=10

        '''
         rnn = nn.GRU(10, 20, 2)
         input = torch.randn(5, 3, 10)
         h0 = torch.randn(2, 3, 20)
         output, hn = rnn(input, h0)
        '''

        self.gru = torch.nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers)   ##num_layer网络层数
        self.dnn=torch.nn.Linear(self.embedding_dim,self.hidden_size)
        '''
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
        '''
        self.final_activation = torch.nn.Softmax()
        #self.loss_fn = SampledCrossEntropyLoss(use_cuda=True)
        self.loss=Loss(self.n_negtive)

        device = device or 'cpu'
        self.device = torch.device(device)

    # def init(self):
    #     ##参数初始化
    #     for i in range(self.interst_num):
    #         torch.nn.weight_init.orthogonal(self.gru_i.weight_ih_l0)
    #         torch.nn.weight_init.orthogonal(self.gru_i.weight_hh_l0)
    #         # self.gru.bias_ih_l0.zero_()
    #         # self.gru.bias_hh_l0.zero_()
    #         torch.nn.init.normal(self.gru_i.bias, std=1.0)

    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def output_user(self):
        interest=self.interest_em
        return interest.cpu().detach().numpy().astype('float32')

    def output_item_em(self):
        # n=[i for i in range(self.n_iid)]  ##所有item
        # iid_var=self.iid_embeddings_var(torch.LongTensor(n).to(self.device)).cpu()
        # return iid_var.detach().numpy().astype('float32')
        return self.iid_embeddings_var.weight.cpu().detach().numpy().astype('float32')  ##item_embedding


    def get_item_emb(self):
        return self.iid_embeddings_var


    def get_embedding(self):
        self.mid_batch_embedded = self.iid_embeddings_var(self.mid_batch_ph) # 根据mid_batch_ph中的id查找mid_embeddings_var的元素，将id对应到embedding上
        self.mid_his_batch_embedded = self.iid_embeddings_var(self.mid_his_batch_ph)
        self.item_eb=self.mid_batch_embedded ##256*32
        self.item_his_eb = self.mid_his_batch_embedded * torch.reshape(self.mask, (-1, self.seq_len, 1))   ##batch_size  seq_len dim
        self.item_his_eb_sum=self.item_his_eb.sum(1)

    def build_sampled_softmax_loss(self,user_eb,neg_flag=0):
        # self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
        #                                                       tf.reshape(self.mid_batch_ph, [-1, 1]), user_eb,
        #                                                       self.neg_num * self.batch_size, self.n_mid))
        ##item_embedding   b    target_item   item个数
        b = self.iid_embeddings_var.weight.unsqueeze(0).repeat(self.batch_size, 1, 1)  ##batch_size*nid*hidden
        rank = torch.matmul(b, user_eb.unsqueeze(-1)).squeeze(-1)  # batch_size * (n_item)
        if neg_flag == 1:
            loss = self.loss(rank, self.mid_batch_ph, self.neg_item)  ##有负采样
        else:
            loss = self.loss(rank, self.mid_batch_ph)  ##采样loss
        return loss


    def forward(self, inps):
        self.mid_batch_ph= torch.LongTensor(inps[1]).to(self.device)  ##target_item
        self.mid_his_batch_ph = torch.LongTensor(inps[2]).to(self.device)
        self.mask = torch.LongTensor(inps[4]).to(self.device)
        self.neg_item=torch.LongTensor(inps[5]).to(self.device)

        self.get_embedding()
        loss = 0
        '''
        model_type=DNN
        self.item_his_eb_mean = torch.sum(self.item_his_eb, 1)  #256 ** dim
        self.user_eb=self.dnn(self.item_his_eb_mean)
        loss=self.build_sampled_softmax_loss(self.user_eb)
        '''


        hidden= torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        #o=[]
        for t in range(self.seq_len):
            output,hidden=self.gru(self.item_his_eb[:,t,:].unsqueeze(0),hidden)
            # outer=output.view(-1, output.size(-1))
            # o.append(outer)  ##相当于是tf中的  256 seqlen H

        final_state=hidden.view(-1, hidden.size(-1))  ##batch_size hidden
        self.interest_em=final_state
        self.user_eb = final_state
        loss=self.build_sampled_softmax_loss(self.user_eb)

            #这是gru pytorch源码loss
            #output = output.view(-1, output.size(-1))  # (B,H)
            # logit = self.final_activation(self.h2o(output))
            # # output sampling
            # logit_sampled = logit[:, self.mid_batch_embedded.view(-1)]   ##
            # loss_ = self.loss_func(logit_sampled)
            # loss=loss+loss_
        return loss



    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        h=list()
        try:
            for i in range(self.interst_num):
                h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
                h.append(h0)
        except:
            self.device = 'cpu'
            for i in range(self.interst_num):
                h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
                h.append(h0)
        return h

