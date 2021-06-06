import torch
from torch import nn
from visdialch.attention import BertRefEmbedding


class GenerativeDecoder(nn.Module):
    def __init__(self, config, vocabulary, bert_model=None, stage=1):
        super().__init__()
        self.config = config
        self.stage = stage

        if config['word_embedding_type'] == 'glove':
            self.word_embed = nn.Embedding.from_pretrained(vocabulary.get_vocab_emb_tensors())
            self.padding_idx = vocabulary.PAD_INDEX
        elif config['word_embedding_type'] == 'bert':
            self.word_embed = BertRefEmbedding(bert_model)
            self.padding_idx = vocabulary.PAD_SEQ_INDEX
        else:
            self.word_embed = nn.Embedding(
                len(vocabulary),
                config["word_embedding_size"],
                padding_idx=vocabulary.PAD_INDEX,
            )
            self.padding_idx = vocabulary.PAD_INDEX
        
        self.answer_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["decoder_lstm_hidden_size"],
            config["decoder_lstm_num_layers"],
            batch_first=True,
            dropout=config["decoder_lstm_dropout"],
        )

        self.use_attention = config['use_attention']
        print('Decoder use attention:', self.use_attention)
        
        if self.use_attention:
            self.query_linear = nn.Linear(config['decoder_lstm_hidden_size'], 
                                          config['decoder_lstm_hidden_size'])
            self.key_linear = nn.Linear(config['decoder_lstm_hidden_size'], 
                                        config['decoder_lstm_hidden_size'])
            self.value_linear = nn.Linear(config['decoder_lstm_hidden_size'],
                                          config['decoder_lstm_hidden_size'])
            self.register_buffer(
                name='attention_mask', 
                tensor=torch.tril(
                    torch.ones(
                        config['max_sequence_length'], 
                        config['max_sequence_length']
                    ), 
                    diagonal=0
                ).unsqueeze(0)
            ) # (1, S, S)

        self.lstm_to_words = nn.Linear(
            self.config["decoder_lstm_hidden_size"], len(vocabulary)
        )

        self.dropout = nn.Dropout(p=config["answer_dropout"])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_output, batch):
        """Given `encoder_output`, learn to autoregressively predict
        ground-truth answer word-by-word during training.

        During evaluation, assign log-likelihood scores to all answer options.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        if self.training and self.stage == 1:

            ans_in = batch["ans_in"]
            batch_size, num_rounds, max_sequence_length = ans_in.size()

            ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)

            # shape: (batch_size * num_rounds, word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            init_hidden = init_hidden.repeat(
                self.config["decoder_lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )
            ans_out = self.dropout(ans_out)                     # (B, S, H)

            if self.use_attention:
                mixed_query = self.query_linear(ans_out)        # (B, S, H)
                mixed_key = self.key_linear(ans_out)            # (B, S, H)
                mixed_value = self.value_linear(ans_out)        # (B, S, H)

                mixed_key = mixed_key.permute(0, 2, 1)          # (B, H, S)
                scores = torch.matmul(mixed_query, mixed_key)   # (B, S, S)
                scores = (scores * self.attention_mask
                        - 1e10 * (1 - self.attention_mask))     # (B, S, S)
                prob = nn.Softmax(dim=-1)(scores)               # (B, S, S)

                ans_out = torch.matmul(prob, mixed_value)       # (B, S, H)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         vocabulary_size)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores

        else:

            ans_in = batch["opt_in"]
            batch_size, num_rounds, num_options, max_sequence_length = (
                ans_in.size()
            )

            ans_in = ans_in.view(
                batch_size * num_rounds * num_options, max_sequence_length
            )

            # shape: (batch_size * num_rounds * num_options,
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds * num_options,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
            init_hidden = init_hidden.repeat(1, 1, num_options, 1)
            init_hidden = init_hidden.view(
                1, batch_size * num_rounds * num_options, -1
            )
            init_hidden = init_hidden.repeat(
                self.config["decoder_lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )

            if self.use_attention:
                mixed_query = self.query_linear(ans_out)        # (B, S, H)
                mixed_key = self.key_linear(ans_out)            # (B, S, H)
                mixed_value = self.value_linear(ans_out)        # (B, S, H)

                mixed_key = mixed_key.permute(0, 2, 1)          # (B, H, S)
                scores = torch.matmul(mixed_query, mixed_key)   # (B, S, S)
                scores = (scores * self.attention_mask 
                        - 1e10 * (1 - self.attention_mask))     # (B, S, S)
                prob = nn.Softmax(dim=-1)(scores)               # (B, S, S)

                ans_out = torch.matmul(prob, mixed_value)       # (B, S, H)

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, vocabulary_size)
            ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            target_ans_out = batch["opt_out"].view(
                batch_size * num_rounds * num_options, -1
            )

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            ans_word_scores = torch.gather(
                ans_word_scores, -1, target_ans_out.unsqueeze(-1)
            ).squeeze()
            ans_word_scores = (
                ans_word_scores * (target_ans_out != self.padding_idx).float().cuda()
            )  # ugly

            ans_scores = torch.sum(ans_word_scores, -1)
            sequence_length = (target_ans_out != self.padding_idx).float().cuda().sum(-1)
            ans_scores = ans_scores / torch.sqrt(sequence_length)
            ans_scores = ans_scores.view(batch_size, num_rounds, num_options)

            return ans_scores
