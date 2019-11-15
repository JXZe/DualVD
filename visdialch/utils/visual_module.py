import torch
from torch import nn
from torch.nn import functional as F
from visdialch.utils.visual_update_step import Visual_update



class Visual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # project image features to lstm_hidden_size for computing attention
        self.image_features_projection = nn.Linear(
            config["img_feature_size"], config["lstm_hidden_size"]
        )

        self.attention_proj = nn.Linear(config["lstm_hidden_size"], 1)

        self.visual_update = Visual_update(config)

        self.dropout = nn.Dropout(p=config["dropout"])

        # initialization
        nn.init.kaiming_uniform_(self.image_features_projection.weight)
        nn.init.constant_(self.image_features_projection.bias, 0)




    def forward(self, img,batch_size,num_rounds,ques_embed,relation,qh_embed):

        #############################################################################
        ##               Original step (get original attention weights)            ##
        #############################################################################

        projected_image_features = self.image_features_projection(img)
        # Repeat image feature vectors to be provided for every round
        projected_image_features = projected_image_features.view(
            batch_size, 1, -1, self.config["lstm_hidden_size"]
        ).repeat(1, num_rounds, 1, 1).view(
            batch_size * num_rounds, -1, self.config["lstm_hidden_size"]
        )
        # Computing attention weights
        projected_ques_features = ques_embed.unsqueeze(1).repeat(
            1, img.shape[1], 1)
        projected_ques_image = projected_ques_features * projected_image_features
        projected_ques_image = self.dropout(projected_ques_image)
        image_attention_weights = self.attention_proj(
            projected_ques_image).squeeze()
        image_attention_weights1 = F.softmax(image_attention_weights, dim=-1)

        img = img.view(
            batch_size, 1, -1, self.config["img_feature_size"]).repeat(
            1, num_rounds, 1, 1).view(
            batch_size * num_rounds, -1, self.config["img_feature_size"]
        )

        #############################################################################
        ##                             Update step                                 ##
        #############################################################################


        img = self.visual_update(relation, img, qh_embed)

        #############################################################################
        ##                       Original-Update Feature Fuse                      ##
        #############################################################################


        image_attention_weights = image_attention_weights1.unsqueeze(-1).repeat(
            1, 1, self.config["img_feature_size"]
        )

        attended_image_features = (image_attention_weights * img).sum(0)
        visul = attended_image_features


        return visul

