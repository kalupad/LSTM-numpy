import numpy as np

from dataset import create_datasets, generate_semantic_dataset, generate_numeric_dataset
from encoding import one_hot_encode_sequence, sequences_to_dicts
from lstm import init_lstm, update_parameters
from pass_functions import backward_pass, forward_pass


def main(hidden_size, sequence_type, num_sequences, num_words, num_epochs, verbose_n):
    """
    Initializes an LSTM infrastructure and trains it.
    
    Args:
     `hidden_size`: the dimensions of the hidden state
     `sequence_type`: type of sequential data to be generated [semantic, financial]
     `num_sequences`: number of sequences to generate for dataset
     `num_words`: maximum size of the vocabulary expressed in words
     `num_epochs`: number of epochs to train
     `verbose_n`: after how many epochs print training status
    """
    if sequence_type=='semantic':
        sequences = generate_semantic_dataset(num_sequences=num_sequences)
    elif sequence_type=='numeric':
        sequences = generate_numeric_dataset(num_sequences=num_sequences)
    
    word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences, num_words)
    training_set, validation_set, test_set = create_datasets(sequences)
    z_size = hidden_size + vocab_size 
    params = init_lstm(hidden_size=hidden_size, vocab_size=vocab_size, z_size=z_size)

    # For each epoch
    for i in range(num_epochs):
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        # For each sentence in validation set
        for j in range(len(validation_set)):
            inputs = validation_set[0][j]
            targets = validation_set[1][j]
            
            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(inputs, word_to_idx, vocab_size)
            targets_one_hot = one_hot_encode_sequence(targets, word_to_idx, vocab_size)

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

            # Forward pass
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward_pass(hidden_size, inputs_one_hot, h, c, params)
            
            # Backward pass
            loss, _ = backward_pass(hidden_size, z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)
            
            # Update loss
            epoch_validation_loss += loss
        
        # For each sentence in training set
        for j in range(len(training_set)):
            inputs = training_set[0][j]
            targets = training_set[1][j]
            
            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(inputs, word_to_idx, vocab_size)
            targets_one_hot = one_hot_encode_sequence(targets, word_to_idx, vocab_size)

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

            # Forward pass
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward_pass(hidden_size, inputs_one_hot, h, c, params)
            
            # Backward pass
            loss, grads = backward_pass(hidden_size, z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)
            
            # Update parameters
            params = update_parameters(params, grads, lr=1e-1)
            
            # Update loss
            epoch_training_loss += loss

        # Print loss every verbose_n epochs
        if i % verbose_n == 0:
            print(f'Epoch {i}, training loss: {epoch_training_loss/len(training_set)}, validation loss: {epoch_validation_loss/len(validation_set)}')


if __name__ == '__main__':
    main(100, 'numeric', 20, 20, 1000, 1)