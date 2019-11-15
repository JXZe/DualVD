import torch
from torch import nn

from visdialch.utils import DynamicRNN


class DiscriminativeDecoder(nn.Module):
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
            config["elmo_embedding_size"], config["word_embedding_size"]
        )


        self.option_rnn = nn.LSTM(config["glove_embedding_size"] + config["word_embedding_size"],
                                  config["lstm_hidden_size"],
                                  config["lstm_num_layers"],
                                  batch_first=True,
                                  dropout=config["dropout"])
        self.option_rnn = DynamicRNN(self.option_rnn)

        self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self, encoder_output, batch):

        #############################################################################
        ##                           Data Read and Embed                           ##
        #############################################################################
        options = batch['opt']
        batch_size, num_rounds, num_options, max_sequence_length = options.size()
        options = options.view(batch_size * num_rounds * num_options, max_sequence_length)

        options_length = batch['opt_len']
        options_length = options_length.view(batch_size * num_rounds * num_options)

        # Pick non-zero length options for processing (relevant for test split).
        nonzero_options_length_indices = options_length.nonzero().squeeze()
        nonzero_options_length = options_length[nonzero_options_length_indices]
        nonzero_options = options[nonzero_options_length_indices]

        nonzero_options_embed_glove = self.glove_embed(nonzero_options)
        nonzero_options_embed_elmo = self.elmo_embed(nonzero_options)
        nonzero_options_embed_elmo = self.dropout(nonzero_options_embed_elmo)
        nonzero_options_embed_elmo = self.embed_change(nonzero_options_embed_elmo)
        nonzero_options_embed = torch.cat((nonzero_options_embed_glove,nonzero_options_embed_elmo),-1)

        _, (nonzero_options_embed, _) = self.option_rnn(
            nonzero_options_embed, nonzero_options_length
        )

        options_embed = torch.zeros(
            batch_size * num_rounds * num_options, nonzero_options_embed.size(-1),
            device=nonzero_options_embed.device
        )
        options_embed[nonzero_options_length_indices] = nonzero_options_embed

        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, num_options, 1)


        #############################################################################
        ##                            Generate Score                               ##
        #############################################################################

        encoder_output = encoder_output.view(
            batch_size * num_rounds * num_options, self.config["lstm_hidden_size"]
        )

        scores = torch.sum(options_embed * encoder_output, 1)

        scores = scores.view(batch_size, num_rounds, num_options)
        return scores
