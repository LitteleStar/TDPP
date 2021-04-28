import numpy as np
import random
import torch as t
import torch.nn.functional as F
import mimn as mimn
from util import *
from utils import *
torch.autograd.set_detect_anomaly(True)

class MIMN(t.nn.Module):
    def __init__(self,n_iid, n_cid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, MEMORY_SIZE,use_negsample=False, Flag="DNN",device=None):
        super(MIMN, self).__init__()

        self.model_flag = Flag
        self.reg = False
        self.state_list = None
        self.use_negsample = use_negsample
        self.BATCH_SIZE=BATCH_SIZE
        self.SEQ_LEN=SEQ_LEN
        self.HIDDEN_SIZE=HIDDEN_SIZE
        self.EMBEDDING_DIM=EMBEDDING_DIM
        self.MEMORY_SIZE=MEMORY_SIZE
        self.n_iid=n_iid
        self.mid_batch_ph=None
        self.mid_his_batch_ph=None
        self.cate_batch_ph=None
        self.cate_his_batch_ph=None
        self.mid_neg_batch_ph=None
        self.cate_neg_batch_ph=None
        self.iid_embeddings_var =t.nn.Embedding(n_iid, EMBEDDING_DIM) # 将每个item id都给一个1*EMBEDDING_DIM的编码，总共有feature num个(算 user item cate
        #self.cid_embeddings_var = t.nn.Embedding(n_cid, EMBEDDING_DIM)

        device = device or 'cpu'
        self.device = torch.device(device)
        ##算y_hat
        self.bn=t.nn.BatchNorm1d(128)
        self.dn=t.nn.Linear(128,200)
        self.dnn12=t.nn.Linear(200,80)
        self.dnn23=t.nn.Linear(80,2)
        ##计算负样本
        self.dnn1 = t.nn.Linear(128, 100)
        self.dnn2 = t.nn.Linear(100, 50)
        self.dnn3 = t.nn.Linear(50, 2)
        ####mimn
        self.cell = mimn.MIMNCell(controller_units=HIDDEN_SIZE, memory_size=MEMORY_SIZE,
                                  memory_vector_dim=2 * EMBEDDING_DIM,  # 以前是2*EMBEDDING_DIM
                                  read_head_num=1, write_head_num=1,
                                  reuse=False, output_dim=HIDDEN_SIZE, clip_value=20, batch_size=BATCH_SIZE,
                                  mem_induction=0, util_reg=0, device=self.device)
        #self.zero_state=self.cell.zero_state(BATCH_SIZE)


    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def output_item_em(self):
        n=[i for i in range(self.n_iid)]  ##所有item\
        iid_var=self.iid_embeddings_var(t.LongTensor(n).to(self.device)).cpu()
        return iid_var.detach().numpy().astype('float32')

    def output_user(self):
        #获取兴趣矩阵,兴趣矩阵的embedding_dim 是item的2倍，输入是[item,cate] ,取前16维  ##通过全连接网络做一个降维
        #reduceSize = torch.nn.Linear(2*self.EMBEDDING_DIM, self.EMBEDDING_DIM)
        Memory_em=t.reshape(self.Memory_Mat,[-1, self.Memory_Mat.shape[-1]])
        #user_em = reduceSize(Memory_em)
        user_em=Memory_em[:,:self.EMBEDDING_DIM].cpu()
        return user_em.detach().numpy().astype('float32')


    def get_item_emb(self):
        return self.iid_embeddings_var

    def get_embedding(self):

        self.mid_batch_embedded = self.iid_embeddings_var(self.mid_batch_ph) # 根据mid_batch_ph中的id查找mid_embeddings_var的元素，将id对应到embedding上
        self.mid_his_batch_embedded = self.iid_embeddings_var(self.mid_his_batch_ph)
        # self.cate_batch_embedded = self.cid_embeddings_var(self.cate_batch_ph)
        # self.cate_his_batch_embedded = self.cid_embeddings_var(self.cate_his_batch_ph)
        self.cate_batch_embedded = self.iid_embeddings_var(self.cate_batch_ph)
        self.cate_his_batch_embedded = self.iid_embeddings_var(self.cate_his_batch_ph)

        if self.use_negsample:
            self.neg_item_his_eb = self.iid_embeddings_var(self.mid_neg_batch_ph)
            #self.neg_cate_his_eb = self.cid_embeddings_var(self.cate_neg_batch_ph)
            self.neg_cate_his_eb = self.iid_embeddings_var(self.cate_neg_batch_ph)
            self.neg_his_eb = t.cat((self.neg_item_his_eb, self.neg_cate_his_eb), 2) * t.reshape(self.mask,(self.BATCH_SIZE, self.SEQ_LEN, 1))#
        self.item_eb=t.cat((self.mid_batch_embedded,self.cate_batch_embedded),1)  ##256*32

        self.item_his_eb=t.cat((self.mid_his_batch_embedded,self.cate_his_batch_embedded),2)*t.reshape(self.mask,(self.BATCH_SIZE, self.SEQ_LEN, 1))##*t.FloatTensor(self.BATCH_SIZE, self.SEQ_LEN, 1)  #256*200*32   256*200*1
        self.item_his_eb_sum=self.item_his_eb.sum(1)
#类MIMN
    def get_state(self,SEQ_LEN=200, Mem_Induction=0, Util_Reg=0, use_negsample=False, mask_flag=True):

        BATCH_SIZE=self.BATCH_SIZE
        HIDDEN_SIZE=self.HIDDEN_SIZE
        EMBEDDING_DIM=self.EMBEDDING_DIM
        MEMORY_SIZE=self.MEMORY_SIZE

        def clear_mask_state(state, begin_state, begin_channel_rnn_state, mask, cell, ti):
            state["controller_state"] = (1 - t.reshape(mask[:, ti], (BATCH_SIZE, 1))) * begin_state[
                "controller_state"] + t.reshape(mask[:, ti], (BATCH_SIZE, 1)) * state["controller_state"]
            state["M"] = (1 - t.reshape(mask[:, ti], (BATCH_SIZE, 1, 1))) * begin_state["M"] + t.reshape(mask[:, ti], (BATCH_SIZE, 1, 1)) * state["M"]
            state["key_M"] = (1 - t.reshape(mask[:, ti], (BATCH_SIZE, 1, 1))) * begin_state["key_M"] + t.reshape(mask[:, ti], (BATCH_SIZE, 1, 1)) * state["key_M"]
            state["sum_aggre"] = (1 - t.reshape(mask[:, ti], (BATCH_SIZE, 1, 1))) * begin_state["sum_aggre"] + t.reshape(mask[:, ti], (BATCH_SIZE, 1, 1)) * state["sum_aggre"]
            if Mem_Induction > 0:
                temp_channel_rnn_state = []
                for i in range(MEMORY_SIZE):
                    '''
                    temp_channel_rnn_state.append(
                        cell.channel_rnn_state[i] * tf.expand_dims(mask[:, t], axis=1) + begin_channel_rnn_state[i] * (
                                    1 - tf.expand_dims(mask[:, t], axis=1)))
                    '''
                    temp_channel_rnn_state.append(
                        cell.channel_rnn_state[i] *mask[:, t].unsqueeze(1) + begin_channel_rnn_state[i] * (
                                1 - mask[:, t].unsqueeze(1)))

                cell.channel_rnn_state = temp_channel_rnn_state
                temp_channel_rnn_output = []
                for i in range(MEMORY_SIZE):
                    '''
                    temp_output = cell.channel_rnn_output[i] * tf.expand_dims(mask[:, t], axis=1) + \
                                  begin_channel_rnn_output[i] * (1 - tf.expand_dims(self.mask[:, t], axis=1))
                    '''
                    temp_output = cell.channel_rnn_output[i] * mask[:, t].unsqueeze(1) + begin_channel_rnn_output[i] * (1 - self.mask[:, t].unsqueeze(1))
                    temp_channel_rnn_output.append(temp_output)
                cell.channel_rnn_output = temp_channel_rnn_output

            return state

        if Mem_Induction > 0:
            begin_channel_rnn_output = self.cell.channel_rnn_output
        else:
            begin_channel_rnn_output = 0.0

        #state=self.zero_state
        state=self.cell.zero_state(BATCH_SIZE)
        begin_state = state
        self.state_list = [state]
        self.mimn_o = []
        w_list = []  # 存储 w_list
        # read = begin_state['read_vector_list'][-3:]
        # key = begin_state['key_M'][-3:]
        # with open("../data/read.txt", "a") as f:  # 设置文件对象
        #     for i in read:  # 对于双层列表中的数据
        #         f.writelines(str(i))
        #     f.writelines("\n end 1 \n")
        # with open("../data/key.txt", "a") as f:  # 设置文件对象
        #     for i in key:  # 对于双层列表中的数据
        #         f.writelines(str(i))
        #     f.writelines("\n end 1 \n")

        for i in range(SEQ_LEN):  # SEQ_LEN=400  为什么要做SEQ_LEN次---因为在取数据时一个用户的历史记录就取了SEQ_LEN个
            # 对于每个t，传入的是[[],...[]]---代表所有batchsize个用户第t个item的embedding集合 ，输入是第t个行为的embedding
            output, state, temp_output_list = self.cell(self.item_his_eb[:, i, :], state)
            # 调用call函数  读写更新
            if mask_flag:
                state = clear_mask_state(state, begin_state, begin_channel_rnn_output, self.mask, self.cell, i)  # 遮罩？为什么
            # 递归，更新状态
            self.mimn_o.append(output)
            w_list.append(state['w_list'])
            self.state_list.append(state)

        # outp=self.mimn_o[-3:]
        # with open("../data/outp.txt", "a") as f:  # 设置文件对象
        #     for i in outp:  # 对于双层列表中的数据
        #         f.writelines(str(i))
        #     f.writelines("\n end 1 \n")

        self.Memory_Mat=state['M']
        self.mimn_o = t.stack(self.mimn_o, axis=1)  # 输出 ，stack拼接向量
        self.state_list.append(state)
        mean_memory = t.mean(state['sum_aggre'], axis=-2)

        before_aggre = state['w_aggre']
        read_out, _, _ = self.cell(self.item_eb, state)

  #loss计算
        if use_negsample:
            aux_loss_1 = self.auxiliary_loss(self.mimn_o[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.neg_his_eb[:, 1:, :], self.mask[:, 1:], stag="bigru_0")
            self.aux_loss = aux_loss_1

        if self.reg:
            self.reg_loss = self.cell.capacity_loss(before_aggre)
        else:
            self.reg_loss = t.zeros(1)
        #test= mean_memory * self.item_eb

        if Mem_Induction == 1:
            channel_memory_tensor = t.cat(temp_output_list, 1)
            '''
            multi_channel_hist = din_attention(self.item_eb, channel_memory_tensor, HIDDEN_SIZE, None, stag='pal')
            inp = t.cat([self.item_eb, self.item_his_eb_sum, read_out, t.squeeze(multi_channel_hist),
                         mean_memory * self.item_eb], 1)
            '''
        else:
            inp = t.cat([self.item_eb, self.item_his_eb_sum, read_out, mean_memory * self.item_eb], 1)

        return inp,w_list


    def forward(self,inps):
        #获取输入
        #device=inps.device  ##让数据运行的设备
        self.uid_batch_ph=t.LongTensor(inps[0]).to(self.device) # 用户id
        self.mid_batch_ph=t.LongTensor(inps[1]).to(self.device)# item id
        self.cate_batch_ph=t.LongTensor(inps[2]).to(self.device)  # category id
        self.mid_his_batch_ph=t.LongTensor(inps[3]).to(self.device)
        self.cate_his_batch_ph=t.LongTensor(inps[4]).to(self.device)  # category history id
        self.mid_neg_batch_ph=t.LongTensor(inps[5]).to(self.device)
        self.cate_neg_batch_ph=t.LongTensor(inps[6]).to(self.device)
        self.mask=t.LongTensor(inps[7]).to(self.device)
        self.target_ph=t.LongTensor(inps[8]).to(self.device)
        self.aux_loss = 0

        self.get_embedding()
        need_y_hat,w_list=self.get_state()
        # n_y=need_y_hat[-5:].cpu().detach().numpy().tolist()
        # with open("../data/n_yhat.txt", "a") as f:  # 设置文件对象
        #     for i in n_y:  # 对于双层列表中的数据
        #         f.writelines(str(i))
        #     f.writelines("\n end 1 \n")


        #操作statelist
        y_hat=self.build_fcn_net(need_y_hat, use_dice=False)
        self.metric(y_hat)
        return self.loss,self.aux_loss,self.accuracy,w_list

#对负采样计算的y_hat
    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1=self.bn(in_)
        dnn1 = F.sigmoid(self.dnn1(bn1))
        dnn2 = F.sigmoid(self.dnn2(dnn1))
        dnn3 = self.dnn3(dnn2)
        y_hat = F.softmax(dnn3)+0.000001
        return y_hat

#计算的y_hat
    def build_fcn_net(self, inp, use_dice=False):
        # 将inp放入三层的神经网络，得到y_hat
        bn1 = self.bn(inp)
        #这里要做faltten操作
        dnn1 = self.dn(bn1)
        dnn1 = prelu(dnn1,device=self.device)   ##
        dnn2 = self.dnn12(dnn1)
        dnn2 = prelu(dnn2,device=self.device)
        dnn3 = self.dnn23(dnn2)
        y_hat = F.softmax(dnn3) + 0.00000001
        # yy=y_hat[-5:].cpu().detach().numpy().tolist()
        # with open("../data/yhat.txt", "a") as f:  # 设置文件对象
        #     for i in yy:  # 对于双层列表中的数据
        #         f.writelines(str(i))
        #     f.writelines("\n end 1 \n")
        return y_hat



    def metric(self,y_hat):

        # Cross-entropy loss and optimizer initialization
        #self.target_ph=t.from_numpy(self.target_ph).float()
        ctr_loss = - t.mean(t.log(y_hat) * self.target_ph)
        self.loss = ctr_loss
        if self.use_negsample:  # 负采样
            self.loss += self.aux_loss
        if self.reg:
            self.loss += self.reg_loss

        self.accuracy = t.mean(t.eq(t.round(y_hat), self.target_ph).float())
        '''
        tf.summary.scalar('accuracy', self.accuracy)  # 准确率
        self.merged = tf.summary.merge_all()
        '''




 # 有负样本时候的计算
    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask=None, stag=None):  # noclick是负样本  h_states是mimn_o
        # mask = tf.cast(mask, tf.float32)
        click_input_ = t.cat([h_states, click_seq], -1)
        noclick_input_ = t.cat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]

        click_loss_ = - t.reshape(t.log(click_prop_), [-1, t.size(click_seq)[1]]) * mask
        noclick_loss_ = - t.reshape(t.log(1.0 - noclick_prop_), [-1, t.size(noclick_seq)[1]]) * mask

        loss_ = t.mean(click_loss_ + noclick_loss_)
        return loss_



'''
    def init_uid_weight(self, sess, uid_weight):
        sess.run(self.uid_embedding_init, feed_dict={self.uid_embedding_placeholder: uid_weight})

    def init_mid_weight(self, sess, mid_weight):
        sess.run([self.mid_embedding_init], feed_dict={self.mid_embedding_placeholder: mid_weight})

    def save_mid_embedding_weight(self, sess):
        embedding = sess.run(self.mid_embeddings_var)
        return embedding

    def save_uid_embedding_weight(self, sess):
        embedding = sess.run(self.uid_bp_memory)
        return embedding

'''










