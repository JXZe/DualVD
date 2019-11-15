import torch
from torch import nn


from visdialch.utils import DynamicRNN
from visdialch.utils.semantic_module import Semantic
from visdialch.utils.visual_module import Visual


class LateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary,glove,elmo):
        super().__init__()
        self.config = config


        self.glove_embed = nn.Embedding(
            len(vocabulary), config["glove_embedding_size"]
        )
        self.elmo_embed = nn.Embedding(
            len(vocabulary), config["elmo_embedding_size"]
        )
        self.glove_embed.weight.data = glove
        self.elmo_embed.weight.data = elmo
        #self.glove_embed.weight.requires_grad = False
        self.elmo_embed.weight.requires_grad = False
        self.embed_change = nn.Linear(
            config["elmo_embedding_size"],config["word_embedding_size"]
        )



        self.hist_rnn = nn.LSTM(
            config["glove_embedding_size"] + config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        self.ques_rnn = nn.LSTM(
            config["glove_embedding_size"] + config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)



        self.ques_hist_fuse = nn.Linear(
            int(2*config["lstm_hidden_size"]),
            config["lstm_hidden_size"]
        )
        self.ques_hist_gate = nn.Linear(
            int(2 * config["lstm_hidden_size"]),
            int(2 * config["lstm_hidden_size"])
        )



        self.semantic_module = Semantic(config)
        self.visual_module = Visual(config)



        self.img_cap_layer = nn.Linear(
            config["img_feature_size"] + config["captionsize_todecoder"],
            config["img_feature_size"] + config["captionsize_todecoder"]
        )


        # fusion layer (attended_image_features + question + history)
        fusion_size = config["img_feature_size"] + config["lstm_hidden_size"] * 2 + config["captionsize_todecoder"]
        self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

        self.dropout = nn.Dropout(p=config["dropout"])



        # initialization

        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

        nn.init.kaiming_uniform_(self.embed_change.weight)
        nn.init.constant_(self.embed_change.bias, 0)

        nn.init.kaiming_uniform_(self.ques_hist_fuse.weight)
        nn.init.constant_(self.ques_hist_fuse.bias, 0)

        nn.init.kaiming_uniform_(self.ques_hist_gate.weight)
        nn.init.constant_(self.ques_hist_gate.bias, 0)

        nn.init.kaiming_uniform_(self.img_cap_layer.weight)
        nn.init.constant_(self.img_cap_layer.bias, 0)



    def forward(self, batch):


        #############################################################################
        ##                           Data Read and Embed                           ##
        #############################################################################

        # Read img/question/dialog_history data
        img = batch["img_feat"]
        relation = batch["relations"]
        ques = batch["ques"]
        hist = batch["hist"]
        batch_size, num_rounds, max_sequence_length = ques.size()

        # Read dense caption data
        captions = batch["captions"]
        captions_rounds = captions.size(1)
        captions_maxlen = captions.size(-1)


        # Embed captions
        captions = captions.view(captions_rounds * batch_size, captions_maxlen)
        captions_glove = self.glove_embed(captions)
        captions_elmo = self.elmo_embed(captions)
        captions_elmo = self.dropout(captions_elmo)
        captions_elmo = self.embed_change(captions_elmo)
        captions = torch.cat((captions_glove, captions_elmo), -1)


        # Embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        ques_embed_glove = self.glove_embed(ques)
        ques_embed_elmo = self.elmo_embed(ques)
        ques_embed_elmo = self.dropout(ques_embed_elmo)
        ques_embed_elmo = self.embed_change(ques_embed_elmo)
        ques_embed = torch.cat((ques_embed_glove,ques_embed_elmo),-1)
        _, (ques_embed, _) = self.ques_rnn(ques_embed, batch["ques_len"])


        # Embed history
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 20)
        hist_embed_glove = self.glove_embed(hist)
        hist_embed_elmo = self.elmo_embed(hist)
        hist_embed_elmo = self.dropout(hist_embed_elmo)
        hist_embed_elmo = self.embed_change(hist_embed_elmo)
        hist_embed = torch.cat((hist_embed_glove, hist_embed_elmo), -1)
        _, (hist_embed, _) = self.hist_rnn(hist_embed, batch["hist_len"])


        # Get global caption data
        global_cap = self._get_global_caption(hist_embed,batch_size,num_rounds)

        #############################################################################
        ##                        Question-History Fusion                          ##
        #############################################################################
        # Fuse ques hist
        qh_embed = torch.cat((ques_embed,hist_embed),-1)
        qh_embed_gate = self.dropout(qh_embed)
        qh_embed_gate = self.ques_hist_gate(qh_embed_gate)
        qh_embed_gate = torch.sigmoid(qh_embed_gate)
        qh_embed = qh_embed * qh_embed_gate
        qh_embed = self.dropout(qh_embed)
        qh_embed = self.ques_hist_fuse(qh_embed)



        #############################################################################
        ##                            Semantic Module                              ##
        #############################################################################
        sem_embed = self.semantic_module(captions,
                                        batch["captions_len"],
                                        captions_rounds,
                                        qh_embed,
                                        global_cap
                                        )


        #############################################################################
        ##                             Visual Module                               ##
        #############################################################################
        vis_embed = self.visual_module(
            img,
            batch_size,
            num_rounds,
            ques_embed,
            relation,
            qh_embed
        )

        #############################################################################
        ##                         Visual-Semantic Fusion                          ##
        #############################################################################
        sight = torch.cat((vis_embed ,sem_embed ), -1)
        sight_gate = self.dropout(sight)
        sight_gate = self.img_cap_layer(sight_gate)
        sight_gate = torch.sigmoid(sight_gate)
        sight= sight_gate * sight

        #############################################################################
        ##                              Late Fusion                                ##
        #############################################################################
        fused_vector = torch.cat((sight, ques_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)
        fused_embedding = torch.tanh(self.fusion(fused_vector))
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)


        return fused_embedding






        #############################################################################
        ##                                  Fuction                                ##
        #############################################################################
    def _get_global_caption(self,hist,batch_size,numrounds):

        hist = hist.view(batch_size,numrounds,-1)
        global_cap = hist[0][0]
        for i in range(len(hist)):
            if i==0 :
                pass
            else:
                global_cap = torch.cat((global_cap,hist[i][0]),-1)
        global_cap = global_cap.view(batch_size,1,-1)
        return global_cap







