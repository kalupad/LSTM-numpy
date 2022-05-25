import numpy as np


def generate_semantic_dataset(num_sequences):
    """
    Generates a number of semantic sequences as the dataset.
    
    Args:
     `num_sequences`: the number of sequences to be generated.
     
    Returns a list of sequences.
    """
    samples = []
    
    for _ in range(num_sequences): 
        num_tokens = np.random.randint(1, 10)
        # END denotes end of the sequence
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['END']
        samples.append(sample)
        
    return samples


def generate_numeric_dataset(num_sequences):
    """
    Generates a number of numeric sequences as the dataset.
    
    Args:
     `num_sequences`: the number of sequences to be generated.
     `ticker`: ticker for the searched company.
     
    Returns a list of sequences.
    """
    samples = []

    for _ in range(num_sequences): 
        step = np.random.randint(1, 9)
        start = np.random.randint(4, 9) * step
        stop = np.random.randint(14, 19) * step

        # Append sample to list
        sample = list(range(start, stop, step))
        samples.append(sample)
        
    return samples


def create_datasets(sequences, p_train=0.7, p_val=0.15, p_test=0.15):
    """
    Generates a train, val, test datasets from input sequences.
    
    Args:
     `sequences`: sequences used in split.
     `p_train`: size of the train split.
     `p_val`: size of the val split.
     `p_test`: size of the test split.
     
    Returns arrays [X, y] for each of train, val, test datasets.
    """
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []
        
        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = [inputs_train, targets_train]
    validation_set = [inputs_val, targets_val]
    test_set = [inputs_test, targets_test]

    return training_set, validation_set, test_set