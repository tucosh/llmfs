import tensorflow as tf
import numpy as np
from utils import tldr

class Embedding(tf.Module):
    def __init__(self, embedding_size, vocab_size, max_position_length=None, name=None):
        super().__init__(name=name)
        self.max_position_length = max_position_length
        self.word_embedding = tf.Variable(tf.random.normal([vocab_size, embedding_size]), name = 'word_embedding')
        self.position_embedding = tf.Variable(tf.random.normal([max_position_length, embedding_size]), name = 'position_embedding') # , dtype=tf.dtypes.int32'
        self.debug = False
    def __call__(self, inputs):
        if self.debug:
            print("inputs shape:", inputs.shape)
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        x = tf.gather(self.word_embedding, inputs)
        if self.debug:
            print("x:", x)
        pe = self.position_embedding[0:seq_length]
        if self.debug:
            print("pe:", pe)
        result = x + pe
        return result

class Normal(tf.Module):
    def __init__(self, d_in, name=None):
        super().__init__(name=name)
        self.beta = tf.Variable(tf.zeros([d_in]), name = 'beta')
        self.gamma = tf.Variable(tf.ones([d_in]), name = 'gamma')
    def __call__(self, inputs):
        epsilon = 1e-5
        mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)
        rdev = tf.math.rsqrt(variance + epsilon)
        x = (inputs - mean) * rdev
        x = x * self.gamma + self.beta
        return x

class Dense(tf.Module):
    def __init__(self, d_in, d_out, activation=None, use_bias=True, name=None):
        super().__init__(name=name)
        self.activation = activation
        if d_in and d_out:
            self.w = tf.Variable(tf.random.normal([d_in, d_out]), name = 'w')
            self.b = tf.Variable(tf.random.normal([d_out]), name = 'b')
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.activation:
            y = self.activation(y)
        return y

class SelfAttention(tf.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, b=0, use_outproj=False, name=None):
        super().__init__(name=name)
        qkv_bias=True # ?
        self.debug = True
        self.d_out = d_out
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.query_layer = Dense(d_in=d_in, d_out=d_out, activation=None, name=f"query", use_bias=qkv_bias)
        self.key_layer = Dense(d_in=d_in, d_out=d_out, activation=None, name=f"query", use_bias=qkv_bias)
        self.value_layer = Dense(d_in=d_in, d_out=d_out, activation=None, name=f"query", use_bias=qkv_bias)

        mask = tf.ones((context_length, context_length), dtype=tf.bool) # square matrix of True
        causal_mask = tf.linalg.band_part(mask, num_lower=-1, num_upper=0) # upper right becomes False
        additive_mask = 1.0 - tf.cast(causal_mask, dtype=tf.float32) # upper right becomes 1.0
        self.additive_mask_applied = additive_mask * -1e9   # upper right is large negative value

    def __call__(self, inputs):
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
        
        return context_vec 
    
class Attention(tf.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, b=0):
        super().__init__(name="attention")
        self.layer_norm = Normal(d_in=d_in, name="layer_norm") 
        self.self_attention = SelfAttention(d_in, d_out, context_length, dropout, num_heads, b)
        self.projection = Dense(d_in=d_in, d_out=d_out, activation=None, name=f"projection")
        self.debug = False

    def __call__(self, inputs):
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

class MultiLayerPerceptron(tf.Module):
    def __init__(self, d_in, d_out, b=0):
        super().__init__(name="mlp")
        self.layer_norm = Normal(d_in=d_in, name=f"layer_norm")
        self.perceptron = Dense(d_in=d_in, d_out=d_out * 4, activation=tf.nn.gelu, name=f"perceptron")
        self.projection = Dense(d_in=d_out * 4, d_out=d_out, name=f"projection")
        self.debug = False
    def __call__(self, inputs):
        if self.debug:
            print(f".... input to    mlp: {tldr(inputs)}")
        x = self.layer_norm(inputs)
        x = self.perceptron(x)
        x = self.projection(x)
        if self.debug:
            print(f".... output from mlp: {tldr(x)}")
        return x


class Block(tf.Module):
    def __init__(self, d_in=None, d_out=None, context_length=None, dropout=None, num_heads=None, b=0, qkv_bias=False):
        super().__init__(name=f'block_{b}')
        self.b = b  
        self.attention = Attention(d_in, d_out, context_length, dropout, num_heads, b)
        self.mlp = MultiLayerPerceptron(d_in=d_in, d_out=d_out, b=b)
        self.debug = False        

    def __call__(self, inputs):
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

class Transformer(tf.Module):
    def __init__(self, blocks_num, d_in, d_out, context_length, dropout, num_heads):
        super().__init__(name="transformer")
        self.blocks_num = blocks_num
        self.blocks = []
        for b in range(blocks_num):
            block = Block(d_in, d_out, context_length, dropout, num_heads, b)
            self.blocks.append(block)
        self.layer_norm = Normal(d_in=d_in, name=f"layer_norm")
             
    def __call__(self, inputs):
        x = inputs
        for b in range(self.blocks_num):
            x = self.blocks[b](x)

        x = self.layer_norm(x)
        return x

class GPT2u(tf.Module):

    def __init__(self, config, name=None, trainable=True, dtype=None):
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
        self.embedding = Embedding(embedding_size=self.embedding_size, vocab_size=self.vocab_size, max_position_length=self.max_position_length)
        self.transformer = Transformer(self.blocks_num, d_in, d_out, context_length, dropout=None, num_heads=num_heads)
        self.debug = False
        

    def __call__(self, inputs):
        x = self.embedding(inputs)
        x = self.transformer(x)
        if self.debug:
            print(f".. input to final matmul: {tldr(x)}")
        logits = tf.matmul(x, self.embedding.word_embedding, transpose_b=True)
        print(f".. final logits: {tldr(logits)}")
        return logits 