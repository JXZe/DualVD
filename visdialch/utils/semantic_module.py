import torch
from torch import nn

from visdialch.utils import DynamicRNN


class Semantic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.cap_rnn = nn.LSTM(
            config["glove_embedding_size"] + config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )


        self.cap_rnn = DynamicRNN(self.cap_rnn)


        self.ques_push = nn.Linear(
            config["lstm_hidden_size"],
            config["ques_change_num"]
        )

        self.caption_push = nn.Linear(
            config["lstm_hidden_size"],
            config["caption_change_num"]
        )

        self.caption_gate = nn.Linear(
            int(2 * config["lstm_hidden_size"]),
            int(2 * config["lstm_hidden_size"])
        )

        self.caption_dim_change = nn.Linear(
            int(2 * config["lstm_hidden_size"]),
            config["lstm_hidden_size"]
        )

        self.dropout = nn.Dropout(p=config["dropout"])


        nn.init.kaiming_uniform_(self.ques_push.weight)
        nn.init.constant_(self.ques_push.bias, 0)

        nn.init.kaiming_uniform_(self.caption_push.weight)
        nn.init.constant_(self.caption_push.bias, 0)

        nn.init.kaiming_uniform_(self.caption_gate.weight)
        nn.init.constant_(self.caption_gate.bias, 0)

        nn.init.kaiming_uniform_(self.caption_dim_change.weight)
        nn.init.constant_(self.caption_dim_change.bias, 0)




    def forward(self, caption, lencap, cap_numrounds, ques, global_cap):

        #############################################################################
        ##                               Data Process                              ##
        #############################################################################

        batch = int(caption.size(0) / cap_numrounds)
        numrounds = int(ques.size(0) / batch)
        ques_emb_size = int(ques.size(-1))

        # Embed local caption
        _, (caption, _) = self.cap_rnn(caption, lencap)
        caption_local = caption.view(batch, cap_numrounds, -1)

        # Get total caption
        caption = self._cat_gl_cap(global_cap, caption_local)
        cap_numrounds = caption.size(1)
        ques = ques.view(batch, numrounds, -1)

        caption = caption.repeat(1, numrounds, 1).view(int(batch * numrounds), cap_numrounds, -1)
        ques = ques.repeat(1, 1, cap_numrounds).view(int(batch * numrounds), -1, ques_emb_size)

        #############################################################################
        ##                            Semantic Attention                           ##
        #############################################################################
        global_cap, local_cap = self.question_attention_layer(caption, ques)

        #############################################################################
        ##                              Feature Fusion                             ##
        #############################################################################
        caption = self.cap_gate_layer(global_cap,local_cap)

        return caption





        #############################################################################
        ##                                   Layer                                 ##
        #############################################################################

    def question_attention_layer(self, caption, ques):

        batch_num = int(ques.size(0))
        caption_weight = self.dropout(caption)
        caption_weight = self.caption_push(caption_weight)


        ques_weight = self.dropout(ques)
        ques_weight = self.ques_push(ques_weight)
        weight = torch.sum((caption_weight * ques_weight), -1)
        weight = torch.softmax(weight, -1).view(batch_num, -1, 1)
        caption = caption * weight

        # Get global caption and local caption
        gl_cap, lo_cap = self._gl_split(caption)
        return gl_cap, lo_cap

    def cap_gate_layer(self, glcap, locap):

        caption = torch.cat((glcap, locap), -1)
        gate_weight = self.dropout(caption)
        gate_weight = self.caption_gate(gate_weight)
        gate_weight = torch.sigmoid(gate_weight)
        caption = gate_weight * caption
        caption = self.dropout(caption)
        caption = self.caption_dim_change(caption)
        return caption


        #############################################################################
        ##                                  Fuction                                ##
        #############################################################################


    def _cat_gl_cap(self, global_cap, local_cap):


        batch_size = local_cap.size(0)
        dim = local_cap.size(-1)
        new_cap = torch.cat((global_cap[0], local_cap[0]), 0).view(-1, dim)
        for i in range(global_cap.size(0)):
            if i == 0:
                pass
            else:
                new_cap = torch.cat((new_cap, global_cap[i], local_cap[i]), 0).view(-1, dim)
        new_cap = new_cap.view(batch_size, -1, dim)
        return new_cap

    def _gl_split(self, caption):

        dim = caption.size(-1)

        gl_num = 1
        local_num = int(caption.size(1) - gl_num)
        gl_cap = caption[0][0].view(1, -1)
        lo_cap = caption[0][1:].view(-1, dim)
        for i in range(caption.size(0)):
            if i == 0:
                pass
            else:
                gl_cap = torch.cat((gl_cap, caption[i][0].view(1, -1)), 0)
                lo_cap = torch.cat((lo_cap, caption[i][1:].view(-1, dim)), 0)
        gl_cap = gl_cap.view(-1, dim)
        lo_cap = lo_cap.view(-1, local_num, dim)
        lo_cap = torch.sum(lo_cap, 0)
        return gl_cap, lo_cap



