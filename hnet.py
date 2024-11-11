import torch
import torch.nn as nn
from hflayers import HopfieldLayer
from typing import List
from digitize import Digitize

class SentenceHopfield(nn.Module):
    def __init__(self, 
                 max_seq_length: int = 512,
                 hidden_dim: int = 768,
                 num_patterns: int = 4):
        """
        Initialize the Hopfield network for sentence pattern storage and retrieval.
        
        Args:
            max_seq_length: Maximum sequence length for padding
            hidden_dim: Dimension of the hidden representations
            num_patterns: Number of patterns to store
        """
        super().__init__()
        
        self.max_seq_length = max_seq_length
        self.num_patterns = num_patterns
        self.hidden_dim = hidden_dim
        
        # Linear projection to get hidden representations
        self.input_proj = nn.Linear(max_seq_length, hidden_dim)
        
        # Hopfield layer for pattern storage and retrieval
        self.hopfield = HopfieldLayer(
            input_size=hidden_dim,
            quantity=num_patterns,
            lookup_weights_as_separated=True,
            lookup_targets_as_trainable=False,
            normalize_stored_pattern_affine=True,
            normalize_pattern_projection_affine=True
        )
        
        # Output projection back to sequence length
        self.output_proj = nn.Linear(hidden_dim, max_seq_length)
        
    def store_patterns(self, patterns: List[str]):
        """
        Store patterns in the Hopfield layer's lookup weights.
        
        Args:
            patterns: List of strings to store
        """
        pattern_tensors = []
        for pattern in patterns[:self.num_patterns]:
            # Use Digitize to encode each pattern
            digitizer = Digitize(pattern, padding=self.max_seq_length)
            encoded = digitizer.encode()
            pattern_tensors.append(torch.tensor(encoded, dtype=torch.float32))
        
        # Stack all pattern tensors
        pattern_tensor = torch.stack(pattern_tensors)  # [num_patterns, max_seq_length]
        
        # Project to hidden dimension
        hidden_patterns = self.input_proj(pattern_tensor)  # [num_patterns, hidden_dim]
        
        # Reshape to match HopfieldLayer's expected dimensions
        hidden_patterns = hidden_patterns.unsqueeze(0)  # [1, num_patterns, hidden_dim]
        
        # Store patterns
        with torch.no_grad():
            self.hopfield.lookup_weights[:] = hidden_patterns
            
    def forward(self, text: str) -> str:
        """
        Retrieve the most similar stored pattern given input text.
        
        Args:
            text: Input string
            
        Returns:
            Retrieved pattern as string
        """
        # Encode input using Digitize
        digitizer = Digitize(text, padding=self.max_seq_length)
        encoded = digitizer.encode()
        input_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)  # [1, max_seq_length]
        
        # Project to hidden dimension
        hidden_input = self.input_proj(input_tensor)  # [1, hidden_dim]
        hidden_input = hidden_input.unsqueeze(1)  # [1, 1, hidden_dim]
        
        # Pass through Hopfield layer
        retrieved = self.hopfield(hidden_input)  # [1, num_patterns, hidden_dim]
        
        # Project back to sequence length
        output = self.output_proj(retrieved.squeeze(0))  # [num_patterns, max_seq_length]
        
        # Take the first pattern (most similar)
        token_ids = torch.round(output[0]).long()  # [max_seq_length]
        
        # Convert to list and decode
        token_list = token_ids.tolist()
        retrieved_text = digitizer.decode(token_list)
        
        return retrieved_text

def train_model(model: SentenceHopfield, 
                patterns: List[str], 
                num_epochs: int = 250,
                learning_rate: float = 1e-3,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the Hopfield network on the given patterns.
    """
    model = model.to(device)
    model.store_patterns(patterns)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for pattern in patterns:
            optimizer.zero_grad()
            
            # Get target encoding
            digitizer = Digitize(pattern, padding=model.max_seq_length)
            
            target_encoded = torch.tensor(digitizer.encode(), dtype=torch.float32).to(device)
            
            # Forward pass
            print(target_encoded)
            retrieved = model(pattern)

            print(retrieved)
            output_digitizer = Digitize(retrieved, padding=model.max_seq_length)
            
            output_encoded = torch.tensor(output_digitizer.encode(), dtype=torch.float32).to(device)
            print(target_encoded.size(), output_encoded.size())

            exit()
            # Compute MSE loss since we're dealing with continuous values
            loss = nn.functional.mse_loss(output_encoded, target_encoded)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(patterns)}")

# Example usage
if __name__ == "__main__":
    model = SentenceHopfield()
    
    patterns = [
        "The quick brown fox jumps over the lazy dog",
        "She sells seashells by the seashore",
        "How much wood would a woodchuck chuck",
        "Peter Piper picked a peck of pickled peppers"
    ]
    
    train_model(model, patterns)
    
    query = "The quick brown"
    retrieved = model(query)
    print(f"Query: {query}")
    print(f"Retrieved: {retrieved}")


    digitizer = Digitize("The quick brown fox", padding=512)
    print(len(digitizer.encode()))

    digitizer2 = Digitize("The quick brown", padding=512)
    print(len(digitizer2.encode()))
