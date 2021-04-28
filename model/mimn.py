#coding:utf-8
import numpy as np
#import tensorflow as tf
import torch as t
import torch.nn.functional as F
t.autograd.set_detect_anomaly(True)

def expand(x, dim, N):
    return t.cat([x.unsqueeze(dim) for _ in range(N)],dim)

def create_linear_initializer(input_size):
    stddev = 1.0 / np.sqrt(input_size) 
    #截断的正态分布中输出随机值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择. stddev标准偏差 dtype 数据类型
    #return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    return stddev
def add_n(list):
    if(len(list)==1):
        return list[0]
    sum=t.add(list[0],list[1])
    num=0
    for i in list:
        if(num==0 or num==1):
            continue
        sum=t.add(sum,i)
    return sum

class MIMNCell(t.nn.Module):   #t.nn.RNNCell
    def __init__(self, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num, reuse=False, 
                 output_dim=None, clip_value=20, shift_range=1, batch_size=128, mem_induction=0, util_reg=0, sharp_value=2.,device=None):
        #super(MIMNCell, self).__init__(10,controller_units)
        super(MIMNCell, self).__init__()

        device = device or 'cpu'
        self.device = t.device(device)
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.mem_induction = mem_induction
        self.util_reg = util_reg
        self.reuse = reuse
        self.clip_value = clip_value 
        self.sharp_value = sharp_value
        self.shift_range = shift_range
        self.batch_size = batch_size
        self.input_dim=64   ##s输入x的最后一维
        if self.mem_induction > 0:   #mem_induction=0
            #tf
            self.channel_rnn = t.nn.GRUCell(self.input_dim,self.memory_vector_dim)  #这里的维度不确定
            #设置初始状态
            self.channel_rnn_state = [self.channel_rnn.zero_state(batch_size) for i in range(memory_size)]
            self.channel_rnn_output = [t.zeros(((batch_size, self.memory_vector_dim))) for i in range(memory_size)]

        self.controller = t.nn.GRUCell(self.input_dim,self.controller_units)     #gru cell这里是调试tf版本直接调出来的controller_input的最后一个维度值
        self.controller.cuda(device=self.device)

        self.step = 0
        self.output_dim = output_dim

        self.o2p_initializer = create_linear_initializer(self.controller_units)  #线性随机初始化
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)


        self.par=t.nn.Linear(32, 140)    ##,total_parameter_num=140
        self.par.cuda(device=self.device)
        self.read_=t.nn.Linear(64, output_dim)  ##
        self.read_.cuda(device=self.device)

        self.learned_r = t.nn.Linear(1, self.memory_vector_dim, bias=False)
        self.learned_r.cuda(device=self.device)
        self.learned_w = t.nn.Linear(1, self.memory_size, bias=False)
        self.learned_w.cuda(device=self.device)

        self.init()

    def init(self):
        t.nn.init.normal(self.par.weight, std=self.o2p_initializer)
        t.nn.init.normal(self.read_.weight, std=self.o2o_initializer)

    def learned_init(self, layer):  # (1,units
        return t.squeeze(layer(t.ones([1, 1]))).to(self.device)

    def forward(self, x, prev_state):
        prev_read_vector_list = prev_state["read_vector_list"]
        controller_input = t.cat([x] + prev_read_vector_list, dim=1)   #一维拼接两个矩阵  [256*64]
        controller_state = self.controller(controller_input, prev_state["controller_state"])
        #controller_output =controller_state
        controller_output =controller_state.detach()

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1   #38
        num_heads = self.read_head_num + self.write_head_num  #2
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num #140
        
        if self.util_reg:
            max_q = 400.0
            prev_w_aggre = prev_state["w_aggre"] / max_q
            controller_par = t.cat([controller_output, t.detach(prev_w_aggre)], axis=1)
        else:
            controller_par = controller_output.detach() #####

        parameters = self.par(controller_par)
        parameters=t.clamp(parameters, -self.clip_value, self.clip_value)  #裁剪

        head_parameter_list = t.chunk(parameters[:, :num_parameters_per_head * num_heads], num_heads, dim=1)
        erase_add_list = t.chunk(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, dim=1)

        # prev_w_list = prev_state["w_list"]
        prev_M = prev_state["M"]
        key_M = prev_state["key_M"]
        w_list = []
        write_weight = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = t.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = (F.softplus(head_parameter[:, self.memory_vector_dim]) + 1)*self.sharp_value
            w = self.addressing(k, beta, key_M, prev_M)
            # if self.util_reg and i == 1:
            #     s = F.softmax(
            #         head_parameter[:,
            #         self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            #     )
            #     gamma = 2 * (F.softplus(head_parameter[:, -1]) + 1) * self.sharp_value
            #     w = self.capacity_overflow(w, s, gamma)
            #     write_weight.append(self.capacity_overflow(t.detach(w), s, gamma))
            w_list.append(w)  ##2   256*4


            #with tf.variable_scope('addressing_head_%d' % i):

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = t.sum(t.unsqueeze(read_w_list[i], dim=2) * prev_M, dim=1)
            read_vector_list.append(read_vector)

        write_w_list = w_list[self.read_head_num:]
            
        #channel_weight = read_w_list[0]

        if self.mem_induction == 0:
            output_list = []
#这段代码没被用到
        # elif self.mem_induction == 1:
        #     _, ind = t.topk(channel_weight, 1)
        #     mask_weight = t.sum(F.one_hot(ind, self.memory_size), dim=-2)
        #     output_list = []
        #     for i in range(self.memory_size):
        #         temp_output, temp_new_state = self.channel_rnn(t.cat([x, t.detach(prev_M[:,i]) * t.unsqueeze(mask_weight[:,i], dim=1)],dim=1), self.channel_rnn_state[i])
        #         self.channel_rnn_state[i] = temp_new_state * t.unsqueeze(mask_weight[:,i], dim=1) + self.channel_rnn_state[i]*(1- t.unsqueeze(mask_weight[:,i], dim=1))
        #         temp_output = temp_output * t.unsqueeze(mask_weight[:,i], dim=1) + self.channel_rnn_output[i]*(1- t.unsqueeze(mask_weight[:,i], dim=1))
        #         output_list.append(t.unsqueeze(temp_output,dim=1))

        #M = prev_M
        sum_aggre = prev_state["sum_aggre"]

        for i in range(self.write_head_num):
            w = t.unsqueeze(write_w_list[i], dim=2)
            erase_vector = t.unsqueeze(F.sigmoid(erase_add_list[i * 2]), dim=1)
            add_vector = t.unsqueeze(t.tanh(erase_add_list[i * 2 + 1]), dim=1)
            prev_M = prev_M * (t.ones(prev_M.size()).to(device=self.device) - t.matmul(w, erase_vector)) + t.matmul(w, add_vector)
            sum_aggre =sum_aggre+ t.matmul(t.detach(w), add_vector)

        w_aggre = prev_state["w_aggre"]
        if self.util_reg:
            w_aggre =w_aggre+ add_n(write_weight)
        else:
            w_aggre = w_aggre+write_w_list[0]

        if not self.output_dim:
            output_dim = x.size()[1]
        else:
            output_dim = self.output_dim


        inpu = t.cat([controller_output] + read_vector_list, dim=1)
        read_output = self.read_(inpu)
        read_output = t.clamp(read_output, -self.clip_value, self.clip_value)
        #read_output.float()   #数据类型同意为float类型

        self.step = self.step+ 1
        return read_output, {
                "controller_state" : controller_state,
                "read_vector_list" : read_vector_list,
                "w_list" : w_list,
                "M" : prev_M,
                "key_M": key_M,
                "w_aggre": w_aggre,
                "sum_aggre": sum_aggre
            }, output_list



    def addressing(self, k, beta, key_M, prev_M):
        # Cosine Similarity
        def cosine_similarity(key, M):
            key = t.unsqueeze(key, dim=2)
            inner_product = t.matmul(M, key)
            k_norm = t.sqrt(t.sum(t.square(key), dim=1,keepdim=True))  #这里tf版本里面keep_dim了
            #k_norm=k_norm.view(k_norm.size()[0],k_norm.size()[1],1)
            M_norm = t.sqrt(t.sum(t.square(M), dim=2,keepdim=True))
            norm_product=M_norm*k_norm
            K = t.squeeze(inner_product / (norm_product + 1e-8))         #删除tensor中所有维度为1 的
            return K

        K = 0.5*(cosine_similarity(k,key_M) + cosine_similarity(k,prev_M))
        K_amplified = t.exp(t.unsqueeze(beta, dim=1) * K)
        w_c = K_amplified / t.sum(K_amplified, dim=1,keepdim=True)

        return w_c

    # def capacity_overflow(self, w_g, s, gamma):
    #     s = t.cat([s[:, :self.shift_range + 1],
    #                    t.zeros([s.size()[0], self.memory_size - (self.shift_range * 2 + 1)]),
    #                    s[:, -self.shift_range:]], dim=1)
    #     t_ = t.cat([t.flip(s, [1]), t.flip(s, [1])], dim=1)
    #     s_matrix = t.stack(
    #         [t_[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
    #         dim=1
    #     )
    #     w_ = t.sum(t.squeeze(w_g, dim=1) * s_matrix, dim=2)
    #     w_sharpen = t.pow(w_, t.squeeze(gamma, dim=1))
    #     w = w_sharpen / t.sum(w_sharpen, dim=1)
    #
    #     return w

    def capacity_loss(self, w_aggre):
        loss = 0.001 * t.mean((w_aggre - t.mean(w_aggre, dim=-1))**2 / self.memory_size / self.batch_size)
        return loss




    def zero_state(self, batch_size):
        inpu=t.ones([1, 1]).to(self.device)

        read_vector_list = [expand(t.tanh(t.squeeze(self.learned_r(inpu))), dim=0, N=batch_size)
            for i in range(self.read_head_num)]

        w_list = [expand(F.softmax(t.squeeze(self.learned_w(inpu))), dim=0, N=batch_size)
            for i in range(self.read_head_num + self.write_head_num)]

        #controller_init_state = t.FloatTensor(batch_size,self.controller_units).fill_(0).to(self.device)
        #controller_init_state = t.nn.init.uniform(t.empty(batch_size,self.controller_units)).to(self.device)
        controller_init_state = t.zeros((batch_size,self.controller_units)).to(self.device)

#不要训练
        M_t=t.nn.init.normal_(t.empty(self.memory_size, self.memory_vector_dim), mean=0.0, std=1e-5)
        M = expand(
            t.tanh(t.autograd.Variable(M_t,requires_grad=False)).to(self.device),
            dim=0, N=batch_size)
##要训练
        # key_M = expand(
        #     t.tanh(t.nn.init.normal_(t.empty(self.memory_size, self.memory_vector_dim), mean=0.0, std=0.5)).to(self.device),
        #     dim=0, N=batch_size)
        key_t = t.nn.init.normal_(t.empty(self.memory_size, self.memory_vector_dim), mean=0.0, std=0.5)
        key_M = expand(
            t.tanh(t.autograd.Variable(key_t, requires_grad=True)).to(self.device),
            dim=0, N=batch_size)

        sum_aggre = t.from_numpy(np.zeros([batch_size, self.memory_size, self.memory_vector_dim])).float().to(self.device)  #float32类型
        zero_vector = np.zeros([batch_size, self.memory_size])
        zero_weight_vector = t.from_numpy(zero_vector).float().to(self.device)

        state = {
            "controller_state" : controller_init_state,
            "read_vector_list" : read_vector_list,
            "w_list" : w_list,
            "M" : M,
            "w_aggre" : zero_weight_vector,
            "key_M" : key_M,
            "sum_aggre" : sum_aggre
        }
        return state