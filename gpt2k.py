# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# import numpy as np
from utils import tldr
# from utils import load_gpt2_params_from_tf_ckpt
# from utils import text_to_token_ids
# from utils import token_ids_to_text
# from utils import generate_text_simple
# !pip install -q tiktoken

class Embedding(tf.keras.layers.Layer):
    def __init__(self, embedding_size, vocab_size, max_position_length, dtype=tf.float32, debug=False):
        super().__init__(name="embedding", dtype=dtype)
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_position_length = max_position_length
        self.word_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size, name="word_embedding")
        # self.word_embedding.build((None, self.embedding_size))
        self.position_embedding = tf.keras.layers.Embedding(input_dim=self.max_position_length, output_dim=self.embedding_size, name="position_embedding")
        # self.position_embedding.build((None, self.embedding_size))
        self.debug = debug

    def build(self, input_shape):
        # input_shape[-1] is the number of features from the previous layer
        if self.debug:       
            print("..Embedding input_shape=", input_shape)
        super().build(input_shape)
        
    def call(self, inputs):
        we = self.word_embedding(inputs)
        pe = self.position_embedding(tf.range(self.max_position_length))
        pe_corrected = pe[:we.shape[1], :]
        x = we + pe_corrected
        if self.debug:
            print(f".. Embedding output: {tldr(x)}")
        return x

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, b, use_outproj=False, debug=False): # , use_outproj=False):
        super().__init__(name="self")
        self.debug = debug
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        qkv_bias=True # ?
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # self.query_layer = tf.keras.layers.Dense(units=self.head_dim, activation=None, name=f"query-{b}", use_bias=qkv_bias)
        self.query_layer = tf.keras.layers.Dense(units=self.d_out, activation=None, name=f"query-{b}", use_bias=qkv_bias)
        # self.query_layer.build((None, d_in))
        self.key_layer = tf.keras.layers.Dense(units=self.d_out, activation=None, name=f"key-{b}", use_bias=qkv_bias)
        # self.key_layer.build((None, d_in))
        self.value_layer = tf.keras.layers.Dense(units=self.d_out, activation=None, name=f"value-{b}", use_bias=qkv_bias)
        # self.value_layer.build((None, d_in))
        self.use_outproj = use_outproj

        self.out_proj = tf.keras.layers.Dense(units=d_out, activation=None, name=f"proj-{b}", use_bias=True)
        # self.out_proj.build((None, d_out))

        mask = tf.ones((context_length, context_length), dtype=tf.bool) # square matrix of True
        causal_mask = tf.linalg.band_part(mask, num_lower=-1, num_upper=0) # upper right becomes False
        additive_mask = 1.0 - tf.cast(causal_mask, dtype=tf.float32) # upper right becomes 1.0
        self.additive_mask_applied = additive_mask * -1e9   # upper right is large negative value

    def build(self, input_shape):
        # input_shape[-1] is the number of features from the previous layer
        if self.debug:         
            print("..SelfAttention input_shape=", input_shape)
        super().build(input_shape) 
        
    def call(self, inputs):
        if self.debug:
            print(f".... input to    SelfAttention: {tldr(inputs)}")
        batch_size, num_tokens, d_in = inputs.shape

        keys = self.key_layer(inputs)      
        queries = self.query_layer(inputs)
        values = self.value_layer(inputs)
        
        keys = tf.reshape(keys, [batch_size, num_tokens, self.num_heads, self.head_dim ]) # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim): 2, 6, 2 -> 2, 6, 2, 1        
        queries = tf.reshape(queries, [batch_size, num_tokens, self.num_heads, self.head_dim ]) # 2, 6, 2, 1
        values = tf.reshape(values, [batch_size, num_tokens, self.num_heads, self.head_dim ]) # 2, 6, 2, 1

        keys = tf.transpose(keys, perm=[0, 2, 1, 3])      # [2,6,2,1] ->  [2, 2, 6, 1]        
        values = tf.transpose(values, perm=[0, 2, 1, 3])      # [2,6,2,1] ->  [2, 2, 6, 1]
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])      # [2,6,2,1] ->  [2, 2, 6, 1]
        
        attn_scores = tf.matmul(queries, tf.transpose(keys, perm=[0, 1, 3, 2]))
        trimmed_additive_mask_applied = self.additive_mask_applied[:num_tokens, :num_tokens]
        attn_scores = attn_scores + trimmed_additive_mask_applied
        attn_weights = tf.nn.softmax(attn_scores / keys.shape[-1]**0.5, axis=-1)

        context_vec = tf.matmul(attn_weights, values)        
        context_vec = tf.transpose(context_vec, perm=[0, 2, 1, 3])
        context_vec = tf.reshape(context_vec, [batch_size, num_tokens, self.d_out]) # (b, num_tokens, self.d_out)

        if self.use_outproj: # llmfs uses this!
            context_vec = self.out_proj(context_vec)
        if self.debug:
            print(f".... output from SelfAttention: {tldr(context_vec)}")
        return context_vec

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, b, debug=False):
        super().__init__(name="attention")
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm") 
        self.self_attention = SelfAttention(d_in, d_out, context_length, dropout, num_heads, b, debug=False)
        self.projection = tf.keras.layers.Dense(units=d_out, activation=None, name=f"projection")
        self.debug = debug

    def build(self, input_shape):
        # input_shape[-1] is the number of features from the previous layer
        if self.debug:         
            print("..AttentionLayer input_shape=", input_shape)
        super().build(input_shape)  
        
    def call(self, inputs):
        if self.debug:
            print(f".... input to    layer_norm: {tldr(inputs)}")
        x = self.layer_norm(inputs)
        if self.debug:
            print(f".... output from layer_norm: {tldr(x)}")
        x = self.self_attention(x)
        if self.debug:
            print(f".... input to    projection: {tldr(x)}")
        x = self.projection(x)
        if self.debug:
            print(f".... output from projection: {tldr(x)}")
        return x

class MultiLayerPerceptron(tf.keras.layers.Layer):
    def __init__(self, d_out, b, debug=False):
        super().__init__(name="mlp")
        self.layer_norm = tf.keras.layers.LayerNormalization(name=f"layer_norm")
        self.perceptron = tf.keras.layers.Dense(units=d_out * 4, activation=tf.keras.activations.gelu, name=f"perceptron")
        self.projection = tf.keras.layers.Dense(units=d_out, name=f"projection")
        self.debug = debug

    def call(self, inputs):
        if self.debug:
            print(f".... input to    mlp: {tldr(inputs)}")
        x = self.layer_norm(inputs)
        x = self.perceptron(x)
        x = self.projection(x)
        if self.debug:
            print(f".... output from mlp: {tldr(x)}")
        return x

class Block(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, b, qkv_bias=False, debug=False):
        super().__init__(name=f'block-{b}')
        self.b = b       

        self.attention = AttentionLayer(d_in, d_out, context_length, dropout, num_heads, b, debug=False)
        self.mlp = MultiLayerPerceptron(d_out, b, debug=False)
        self.debug = debug

    def build(self, input_shape):
        # input_shape[-1] is the number of features from the previous layer
        if self.debug:         
            print("..Block input_shape=", input_shape)
        super().build(input_shape)          

    def call(self, inputs):
        if self.debug:
            print()
            print(f".. input to    block {self.b}: {tldr(inputs)}")
        x = inputs
        a = self.attention(x)
        x = x + a
        m = self.mlp(x)
        x = x + m
        if self.debug:
            print(f".. output from block {self.b}: {tldr(x)}")
        return x

class Transformer(tf.keras.layers.Layer):
    def __init__(self, blocks_num, d_in, d_out, context_length, dropout, num_heads, debug=False):
        super().__init__(name="transformer")
        self.blocks_num = blocks_num
        self.blocks = []
        self.debug = debug
        for b in range(blocks_num):
            block = Block(d_in, d_out, context_length, dropout, num_heads, b, debug=False)
            self.blocks.append(block)
        self.layer_norm = tf.keras.layers.LayerNormalization(name=f"layer_norm")

    def build(self, input_shape):
        # input_shape[-1] is the number of features from the previous layer
        if self.debug:        
            print("..Transformer input_shape=", input_shape)
        super().build(input_shape)        
        
    def call(self, inputs):
        x = inputs
        for b in range(self.blocks_num):
            x = self.blocks[b](x)

        x = self.layer_norm(x)
        return x

@tf.keras.saving.register_keras_serializable()
class GPT2k(tf.keras.Model):
    def __init__(self, config, name=None, trainable=True, dtype=None, debug=False, logger=None):
        super().__init__(name=name)
        self.trainable = trainable
        self.embedding_size=config['n_embd']
        self.vocab_size=config['n_vocab']
        self.max_position_length=config['n_ctx']
        self.blocks_num = config["n_layer"]
        d_in=config['n_embd']
        d_out=config['n_embd']
        context_length = config['n_ctx']
        num_heads = config['n_head']
        self.embedding = Embedding(embedding_size=self.embedding_size, vocab_size=self.vocab_size, max_position_length=self.max_position_length, debug=False)
        self.transformer = Transformer(self.blocks_num, d_in, d_out, context_length, dropout=None, num_heads=num_heads, debug=False)
        self.debug = debug
        self.logger = logger
        if logger:
            logger.info("GPT2k init")
        
    def call(self, inputs):         
        x = self.embedding(inputs)
        # return x
        x = self.transformer(x)

        if self.debug:
            print(f".. input to final matmul: {tldr(x)}")
        logits = tf.matmul(x, self.embedding.word_embedding.weights[0], transpose_b=True)
        if self.debug:
            print(f".. final logits: {tldr(logits)}")
        return logits
    