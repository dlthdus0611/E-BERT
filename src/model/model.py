from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel

from .modeling_bert import BertPooler, BertEmbeddings, BertEncoder 
    
class E_BERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
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
        output_attentions = None
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
            output_attentions=output_attentions,
        )

        pooled_output = self.pooler(encoder_outputs) 
        output = self.dropout(pooled_output)
        logits = self.classifier(output)

        return logits