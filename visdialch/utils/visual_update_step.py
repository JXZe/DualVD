import torch
from torch import nn



class Visual_update(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.change_relation = nn.Linear(config["relation_dims"],config["relation_change_num"])

        if config["relation_change_num"] != config["lstm_hidden_size"]:
            self.queschange = nn.Linear(
                config["lstm_hidden_size"],
                config["relation_change_num"]
            )

            nn.init.kaiming_uniform_(self.queschange.weight)
            nn.init.constant_(self.queschange.bias, 0)


        self.rel_weight = nn.Linear(
            config["relation_change_num"],
            1
        )

        self.cat_img_rel_push = nn.Linear(
            config["relation_change_num"] + config["img_feature_size"],
            config["relation_change_num"]
        )

        self.graph_weight_layer = nn.Linear(
            config["relation_change_num"],
            1
        )

        self.cat_img_img_weight_layer = nn.Linear(
            int(2*config["img_feature_size"]),
            int(2 * config["img_feature_size"])
        )

        self.img_change = nn.Linear(
            int(2 * config["img_feature_size"]),
            config["img_feature_size"]
        )

        self.dropout = nn.Dropout(p=config["dropout"])

        # initialization
        nn.init.kaiming_uniform_(self.change_relation.weight)
        nn.init.constant_(self.change_relation.bias, 0)

        nn.init.kaiming_uniform_(self.rel_weight.weight)
        nn.init.constant_(self.rel_weight.bias, 0)

        nn.init.kaiming_uniform_(self.cat_img_rel_push.weight)
        nn.init.constant_(self.cat_img_rel_push.bias, 0)

        nn.init.kaiming_uniform_(self.graph_weight_layer.weight)
        nn.init.constant_(self.graph_weight_layer.bias, 0)

        nn.init.kaiming_uniform_(self.cat_img_img_weight_layer.weight)
        nn.init.constant_(self.cat_img_img_weight_layer.bias, 0)

        nn.init.kaiming_uniform_(self.img_change.weight)
        nn.init.constant_(self.img_change.bias, 0)




    def forward(self,relation,img,ques):


        image_prosize = img.size(1)
        batchsize = relation.size(0)
        numrounds = int(ques.size(0) / batchsize)

        #############################################################################
        ##                           Relation Attention                            ##
        #############################################################################

        relation = self.dropout(relation)
        relation = self.change_relation(relation)
        relation = torch.sigmoid(relation)
        relation_dims_new = relation.size(-1)
        relation = relation.repeat(1,1,numrounds,1).view(-1,image_prosize,image_prosize,relation_dims_new)
        if self.config["lstm_hidden_size"] != self.config["relation_change_num"]:
            ques1 = self.queschange(ques)

        ques1 = ques1.repeat(1,image_prosize*image_prosize).view(int(batchsize*numrounds),image_prosize,image_prosize,-1)
        ques1 = torch.sigmoid(ques1)

        relation_weight = ques1 * relation
        relation_weight = self.dropout(relation_weight)
        relation_weight = self.rel_weight(relation_weight)
        relation_weight = torch.softmax(relation_weight,-2)
        relation = relation_weight * relation


        #############################################################################
        ##                            Graph Convolution                            ##
        #############################################################################

        img_ch = img.repeat(1,1,image_prosize,1).view(int(batchsize*numrounds),image_prosize,image_prosize,-1)
        cat_relation_img = torch.cat((relation,img_ch),-1)
        cat_relation_img = self.dropout(cat_relation_img)
        ques2 = ques1
        cat_relation_img = self.cat_img_rel_push(cat_relation_img)
        graph_weitht = ques2*cat_relation_img

        graph_weitht = self.dropout(graph_weitht)
        graph_weitht = self.graph_weight_layer(graph_weitht)
        graph_weitht = torch.softmax(graph_weitht,-2)

        img_weighted_sum = torch.sum(graph_weitht*img_ch,-2)
        #############################################################################
        ##                    Object-Relation Information Fusion                   ##
        #############################################################################

        cat_img_img = torch.cat((img,img_weighted_sum),-1)
        cat_img_img_weight = self.dropout(cat_img_img)
        cat_img_img_weight = self.cat_img_img_weight_layer(cat_img_img_weight)
        cat_img_img_weight = torch.sigmoid(cat_img_img_weight)
        cat_img_img = cat_img_img_weight * cat_img_img
        visual_update = self.dropout(cat_img_img)
        visual_update = self.img_change(visual_update)


        return visual_update