

import gc
import sys
import numpy as np
import tensorflow as tf
import tiktoken
import os
import urllib.request
from tqdm import tqdm
import json

# load weights into a "params" dict
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def tldr(tensor):
    return f"{tensor.dtype.name} {tensor.shape} {tensor[0][0][:3]}"

# def color_print(text, condition):
#     """Prints text in green if condition is True, or red if condition is False."""
#     # ANSI escape code for Red
#     RED = '\033[91m'
#     # ANSI escape code for Green
#     GREEN = '\033[92m'
#     # ANSI escape code to reset color
#     ENDC = '\033[0m'    
#     if condition:
#         print(f"{GREEN}{text}{ENDC}")
#     else:
#         print(f"{RED}{text}{ENDC}")
def print_layer_structure(layer_or_model, level=0):
    """Recursively prints the structure of a Keras layer or Model."""
    indent = "  " * level
    # Print the current layer's name and class
    print(f"{indent}- {layer_or_model.name} ({type(layer_or_model).__name__})")

    # The .layers property lists layers that are children of the current one
    if hasattr(layer_or_model, 'layers') and layer_or_model.layers:
        for inner_layer in layer_or_model.layers:
            # Recursively call the function for nested components
            print_layer_structure(inner_layer, level + 1)
            


def clean_up(vars_to_delete=None):
    """
    Safely deletes specified variables and clears GPU/RAM memory.
    
    Args:
        vars_to_delete (list): A list of string names of variables to delete.
                               Example: ['model', 'optimizer', 'large_df']
    """
    if vars_to_delete is None:
        vars_to_delete = []

    # 1. Delete specific variables safely
    for var_name in vars_to_delete:
        if var_name in globals():
            del globals()[var_name]
            print(f"Deleted: {var_name}")
        else:
            print(f"Skipped: {var_name} (not found)")

    # 2. Run Python Garbage Collector (clears RAM)
    gc.collect()

    # 3. Clear GPU VRAM (Framework specific)
    # Check for PyTorch
    if 'torch' in sys.modules:
        import torch
        torch.cuda.empty_cache()
        print("PyTorch GPU cache cleared.")
    
    # Check for TensorFlow
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        print("TensorFlow session cleared.")

    print("Memory cleanup finished.")

def generate_text_simple(model, idx, max_new_tokens, context_size):
    idx = tf.cast(idx, dtype=tf.int64)
    for i in range(max_new_tokens):
        idx_cond = idx
        # print("i=", i, "idx_cond=", idx_cond)
        logits = model(idx)
        logits = logits[:, -1, :]
        probas = tf.nn.softmax(logits, -1)
        idx_next = tf.argmax(logits)
        idx_next = tf.argmax(probas, -1)        
        # print("idx_next:", idx_next)
        idx_next_expanded = tf.expand_dims(idx_next, axis=0)
        # print("idx_next_expanded:", idx_next_expanded)
        idx = tf.concat((idx, idx_next_expanded), axis=1)
        # print("idx_next_expanded after concat:", idx_next_expanded)
        # print("idx               after concat:", idx)
    return idx

tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, tokenizer=tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    encoded_tensor = tf.constant(encoded) #.unsqueeze(0) # add batch dimension
    encoded_tensor = tf.expand_dims(encoded_tensor, axis=0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer=tokenizer):
    # flat = token_ids.squeeze(0) # remove batch dimension
    # return tokenizer.decode(flat.tolist())
    return tokenizer.decode(token_ids[-1])

def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# settings = download_gpt2(model_size="355M", models_dir="gpt2") # 124M 355M        
def download_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    # params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings

def load_weights_into_gpt(gpt, params):
    n_embd = params['g'].shape[0]
    _ = gpt.embedding(tf.constant([[1]]))
    assert gpt.embedding.word_embedding.built
    assert gpt.embedding.position_embedding.built
    gpt.embedding.word_embedding.set_weights([np.array(params['wte'])])
    gpt.embedding.position_embedding.set_weights([np.array(params['wpe'])])
    for b in range(len(params["blocks"])):
        gpt.transformer.blocks[b].attention.layer_norm.beta_initializer = tf.keras.initializers.Constant(params["blocks"][b]["ln_1"]["b"]) 
        gpt.transformer.blocks[b].attention.layer_norm.gamma_initializer = tf.keras.initializers.Constant(params["blocks"][b]["ln_1"]["g"]) 
        gpt.transformer.blocks[b].attention.layer_norm.build((None, None, n_embd))
        assert gpt.transformer.blocks[b].attention.layer_norm.built

        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        
        assert gpt.transformer.blocks[b].attention.self_attention.query_layer.built
        gpt.transformer.blocks[b].attention.self_attention.query_layer.set_weights([q_w, q_b])
        assert gpt.transformer.blocks[b].attention.self_attention.key_layer.built
        gpt.transformer.blocks[b].attention.self_attention.key_layer.set_weights([k_w, k_b])
        assert gpt.transformer.blocks[b].attention.self_attention.value_layer.built
        gpt.transformer.blocks[b].attention.self_attention.value_layer.set_weights([v_w, v_b])
    
        # AttentionLayer projection
        gpt.transformer.blocks[b].attention.projection.build((None, n_embd))
        assert gpt.transformer.blocks[b].attention.projection.built
        gpt.transformer.blocks[b].attention.projection.set_weights([params["blocks"][b]["attn"]["c_proj"]["w"], params["blocks"][b]["attn"]["c_proj"]["b"]])      
    
        # MultiLayerPerceptron layer_norm
        gpt.transformer.blocks[b].mlp.layer_norm.beta_initializer = tf.keras.initializers.Constant(params["blocks"][b]["ln_2"]["b"]) 
        gpt.transformer.blocks[b].mlp.layer_norm.gamma_initializer = tf.keras.initializers.Constant(params["blocks"][b]["ln_2"]["g"])
        gpt.transformer.blocks[b].mlp.layer_norm.build((None, None, n_embd))
        assert gpt.transformer.blocks[b].mlp.layer_norm.built
    
        # MultiLayerPerceptron perceptron
        gpt.transformer.blocks[b].mlp.perceptron.build((None, n_embd))
        assert gpt.transformer.blocks[b].mlp.perceptron.built
        gpt.transformer.blocks[b].mlp.perceptron.set_weights([params["blocks"][b]["mlp"]["c_fc"]["w"], params["blocks"][b]["mlp"]["c_fc"]["b"]])
        # MultiLayerPerceptron projection
        mlp_proj_embd = params["blocks"][b]["mlp"]["c_proj"]["w"].shape[0]
        gpt.transformer.blocks[b].mlp.projection.build((None, mlp_proj_embd))
        assert gpt.transformer.blocks[b].mlp.projection.built
        gpt.transformer.blocks[b].mlp.projection.set_weights([params["blocks"][b]["mlp"]["c_proj"]["w"], params["blocks"][b]["mlp"]["c_proj"]["b"]])    
            
    gpt.transformer.layer_norm.beta_initializer = tf.keras.initializers.Constant(params["b"])
    gpt.transformer.layer_norm.gamma_initializer = tf.keras.initializers.Constant(params["g"])
    gpt.transformer.layer_norm.build((None, None, n_embd))
    assert gpt.transformer.layer_norm.built   