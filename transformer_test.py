import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

# Import tf_text to load the ops used by the tokenizer saved model
import tensorflow_text  # pylint: disable=unused-import

#disable warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#globals
MAX_TOKENS = 128
BUFFER_SIZE = 10000
BATCH_SIZE = 32 

def print_samples(train_examples):
    """print some training instances""" 
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print()
        for en in en_examples.numpy():
            print(en.decode('utf-8'))
    return [pt_examples,en_examples]

def get_tokenizers(model_name):
    """ grab a pretrained word embedder to tokenize words"""
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)
    return tokenizers

def test_tokenizers(en_examples,tokenizers):
    """ do a round trip tokenization on some data"""
    for en in en_examples.numpy():
        print(en.decode('utf-8'))
    #perform word embedding
    encoded = tokenizers.en.tokenize(en_examples)
    for row in encoded.to_list():
        print(row)
    #decode
    decoded = tokenizers.en.detokenize(encoded)
    for line in decoded.numpy():
        print(line.decode('utf-8'))

def token_max_length(train_examples,tokenizers):
    """ grab the max length for a subset of the tokens """
    lengths = []
    for pt_examples, en_examples in train_examples.batch(1024):
        pt_tokens = tokenizers.en.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())
        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)
    all_lengths = np.concatenate(lengths)
    max_length = max(all_lengths)
    print("Max token length is ",max_length)

def filter_max_tokens(pt,en):
    """drop tokens longer than max token length"""
    num_tokens = tf.maximum(tf.shape(pt)[1],tf.shape(en)[1])
    return num_tokens < MAX_TOKENS

def tokenize_pairs(pt,en):
    """ tokenize batches of raw text """
    pt = tokenizers.pt.tokenize(pt)
    en = tokenizers.en.tokenize(en)

    #convert from ragge tensor to dense and pad with zeros
    pt = pt.to_tensor()
    en = en.to_tensor()
    return pt,en

def make_batches(ds):
    """ input pipeline to process,
        shuffle and batch data """
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(filter_max_tokens)
        .prefetch(tf.data.AUTOTUNE))

def get_angles(pos,i,d_model):
    """ get angle for positional encoding"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """ perform positional encoding on word embedding """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    """ mask all of the padded tokens in the batch of sequences 
        do this to avoid treating padding as input """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """ mask all of the future tokens in a sequence.
        i.e to predict the third token only the first and
        second tokens should be used. """
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)), -1, 0)
    return mask

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    
    A good explanation of self-attention mechanism can be found in the second 
    half of these lecture slides:
    http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf

    """
    
    """ 
    seq_len is the number of samples you are putting in
    depth is the length of the samples
    for example: 

    temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    seq_len_q = 4
    depth = 3

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    seq_len_v = 4
    depth_v = 2

    #This `query` aligns with the second `key`,
    #so the second `value` is returned.
    
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    
    seq_len_q = 1
    depth (same as above with k) = 3

    return attention weights of shape (seq_len_q, seq_len_k)
    i.e one set of weights or "correlations" for each sample 
    of length seq_len_q.

    output is in the shape seq_len_q, depth_v
    v extracts seq_len_q features of length depth_v
    """

    print("Q is ", q)
    print("Seq_len_q is ", np.shape(q)[0])
    print("Depth is ", np.shape(q)[1])

    matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)

    #apply scaling to QK^(T)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    #mask the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * 1e-9)

    #apply softmax to squeeze into [0,1] range
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # output softmax(Q*K^T / sqrt(d_k)) * V
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi head attention is like bundling a bunch of self-attention 
        heads together and the averaging the output.
        Each self-attention head learns to recognize different things in
        the image. 

        Each attention block gets three inputs: Q,K,V
        Q: Query (The value we are searching for) 
        K: Key (A possible Match) 
        V: Value -> extracts features from <Q,K> similarity weights
        
        softmax (dot(Q,K)/scaling) = attention weights

        These tell us where to attend to for each word, 
        i.e how much each word correlates with every other word
        (each word gets a vector the same length as the input string)
        This is scaled so its correlation with itself doesn't blow out
        the result

        Now we multiply this with the value to get the feature
        with high attention 

        A(Q,K,V) = softmax (dot(Q,K) / scaling) * V

        In this implementation, the dense layers for each head are squished
        into a single layer with num_heads outputs for computational 
        efficiency.
    """

    def __init__(self,*,d_model,num_heads):
        super(MultiHeadAttention, self).__init__()
        #inherits methods from tf.keras.layers.Layer
        self.num_heads = num_heads
        #dimensionality of input, same size of embedding size
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """ split the heads into (num_heads, depth)
        Transform the result so that the shape is 
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """ run the self attention head on q,k,v 
            and return output and attention weights."""
        batch_size = tf.shape(q)[0]
        #we have batch_size number of batches...

        # size for q,k,v
        # (batch_size, seq_len, d_model)

        q = self.wq(q) 
        k = self.wk(k)
        v = self.wv(v)

        # (batch_size, num_heads, seq_len_{q or k or v}, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape = (batch_size, num_heads, seq_len_q, depth)
        # 4d tensor of batch_size batches of 3d tensors
        # num_heads layers of attention activations in shape 
        # seq_len_q (number of queries) x depth (length of queries)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q,k,v,mask)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])
        # squash multi attention outputs into (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, 
                            (batch_size, -1, self.d_model))
        # output shape = (batch_size, seq_len_q, d_model)
        # just a dense layer
        output = self.dense(concat_attention)

        return output, attention_weights

if __name__=="__main__":
    #grab dataset
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    [pt_examples, en_examples] = print_samples(train_examples)
    #apply tokenizer model
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tokenizers = get_tokenizers(model_name)
    test_tokenizers(en_examples,tokenizers)
    token_max_length(train_examples, tokenizers)
    #make some batches
    train_batches = make_batches(train_examples)
    #print("training batches",train_batches)
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    #print("padding mask", create_padding_mask(x))
    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    #print("look ahead mask", temp)
    np.set_printoptions(suppress=True)
    #test the self-attention mechanism
    temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)
    temp_v = tf.constant([[1, 0],
                      [12, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 2)
    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print("Running attention head example...")
    print_out(temp_q, temp_k, temp_v)
    # This query aligns equally with the first and second key,
    # so their values get averaged.
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)
    # pass all the queries together
    print("Running second attention head example...")
    temp_q = tf.constant([[0, 0, 10],
                      [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    print_out(temp_q, temp_k, temp_v)
    
    print("Testing multi head attention")
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512)) #(batch_size, encoder_sequence_len, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print("Printing output and attention weight shape")
    print(out.shape, attn.shape)
