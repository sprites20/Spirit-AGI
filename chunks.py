from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize

def chunk_tokens(tokens, max_length, overlap_size):
    chunks = []
    chunk_indices = []
    i = 0
    while i < len(tokens):
        end_idx = min(i + max_length, len(tokens))
        chunk = tokens[i:end_idx]
        chunks.append(chunk)
        chunk_indices.append((i, end_idx))
        i += max_length - overlap_size
    return chunks, chunk_indices

# Example document (a very long string)
document = "This is a very long document. " * 100  # Adjust length as needed

# Load model and tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3')
model.eval()

# Tokenize the entire document
tokens = tokenizer.tokenize(document)
token_offsets = tokenizer(document, return_offsets_mapping=True)['offset_mapping']

# Define maximum token length and overlap size
max_length = 128  # Maximum number of tokens in each chunk (adjust as needed for your model)
overlap_size = 20  # Overlap size in tokens

# Chunk the tokens
chunks, chunk_indices = chunk_tokens(tokens, max_length, overlap_size)

# Get character indices for chunks
char_indices = [(token_offsets[start_idx][0], token_offsets[end_idx-1][1]) for start_idx, end_idx in chunk_indices]

# Convert token chunks back to strings for processing
chunk_strings = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

# Print the chunks and their character indices
for i, (start_idx, end_idx) in enumerate(char_indices):
    print(f"Chunk {i + 1}:")
    print(f"Start char index: {start_idx}, End char index: {end_idx}")
    print(chunk_strings[i])
    
    # Tokenize sentences
    encoded_input = tokenizer(chunk_strings[i], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        print(f"Embeddings: {sentence_embeddings}")
    print()
    
"""
# Tokenize chunks (again for model input, since we need tensors)
encoded_input = tokenizer(chunk_strings, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    chunk_embeddings = model_output.last_hidden_state[:, 0, :]

# Normalize embeddings
chunk_embeddings_normalized = normalize(chunk_embeddings, p=2, dim=1)

# Scale the embeddings
chunk_embeddings_scaled = chunk_embeddings_normalized / 20.0

# Print the normalized embeddings
for i, embedding in enumerate(chunk_embeddings_scaled):
    print(f"Chunk {i + 1} embedding:\n{embedding}\n")
"""
# Untokenize back to original document from chunks
reconstructed_document = " ".join(chunk_strings)
print(f"Reconstructed Document: {reconstructed_document}")
