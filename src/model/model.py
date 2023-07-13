import math
import torch
from torch import nn
from transformers.utils import logging
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput, BertPreTrainedModel

class E_BERT(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.config = config

        self.embeddings = E_BertEmbeddings(config)
        self.encoder = E_BertEncoder(config)
        self.pooler = Pooler(self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids = None,
        entity_ids = None,
        attention_mask = None,
        local_attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
    ):
        input_shape = input_ids.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            entity_ids=entity_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            local_attention_mask=local_attention_mask,
        )

        pooled_output = self.pooler(encoder_outputs) 
        output = self.dropout(pooled_output)
        logits = self.classifier(output)

        return logits
    
class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class E_BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.entity_embeddings = nn.Embedding(4, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids = None,
        entity_ids = None,
        position_ids = None,
        token_type_ids = None,
        inputs_embeds = None,
    ):
        seq_length = input_ids.size()[1]
        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        if entity_ids is not None:
            entity_embeddings = self.entity_embeddings(entity_ids)
            embeddings += entity_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
class E_BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        local_attention_mask = None,
        gate_outputs = None
    ):
        mixed_query_layer = self.query(hidden_states)


        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores_g = attention_scores + attention_mask
        attention_probs_g = nn.functional.softmax(attention_scores_g, dim=-1)
        attention_probs_g = self.dropout(attention_probs_g)
        context_layer_g = torch.matmul(attention_probs_g, value_layer)

        if local_attention_mask is not None:
            local_attention_mask = (1.0 - local_attention_mask.to(dtype=torch.float32)) * torch.finfo(torch.float32).min
            attention_scores_l = attention_scores + local_attention_mask

            attention_probs_l = nn.functional.softmax(attention_scores_l, dim=-1)        
            attention_probs_l = self.dropout(attention_probs_l)
            context_layer_l = torch.matmul(attention_probs_l, value_layer)

            context_layer   = gate_outputs * context_layer_l + (1 - gate_outputs) * context_layer_g
            attention_probs = gate_outputs * attention_probs_l + (1 - gate_outputs) * attention_probs_g
        else:
            context_layer = context_layer_g
            attention_probs = attention_probs_g

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        return outputs
    
class Gate_layer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.linear = nn.Linear(self.hidden_size, 1)
        self._init_weights(self.linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, hidden_states):
        scores = torch.sigmoid(self.linear(hidden_states))

        return scores.squeeze(-1)
    
class E_BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layer = nn.ModuleList([E_BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gate = nn.ModuleList([Gate_layer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        local_attention_mask = None,
    ):
        
        for i, (layer_module) in enumerate(self.layer):

            gate_outputs = self.gate[i](hidden_states)
            extended_gate_outputs = gate_outputs[:, None, :, None]

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                local_attention_mask,
                extended_gate_outputs,
            )
                
            hidden_states = layer_outputs[0]

        return hidden_states

class E_BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = E_BertAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        local_attention_mask = None,
        gate_outputs = None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            local_attention_mask,
            gate_outputs
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output

class E_BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.self = E_BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        local_attention_mask = None,
        gate_outputs = None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            local_attention_mask,
            gate_outputs
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
