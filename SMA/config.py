from transformers import PretrainedConfig

class MultiModalQformerConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # attention parameters
        self.num_heads = 8
        self.in_dim = 1024
        self.hidden_dim = 1024 * 2
        self.out_dim = 1024 // 8
        self.num_layers = 3
        self.cross_att_fequency = 3

        #trainer parameters
        self.freeze_encoder = True