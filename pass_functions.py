import numpy as np

from activation_functions import sigmoid, softmax, tanh


def forward_pass(hidden_size, inputs, h_prev, C_prev, p):
    """
    Arguments:
    `hidden_size`: Number of dimensions in the hidden state
    `x`: Input data at timestep "t", numpy array of shape (n_x, m).
    `h_prev`: Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    `C_prev`: Memory state at timestep "t-1", numpy array of shape (n_a, m)
    `p`: Python list containing:
                        `W_f`: Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        `b_f`: Bias of the forget gate, numpy array of shape (n_a, 1)
                        `W_i`: Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        `b_i`: Bias of the update gate, numpy array of shape (n_a, 1)
                        `W_g`: Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        `b_g`: Bias of the first "tanh", numpy array of shape (n_a, 1)
                        `W_o`: Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        `b_o`: Bias of the output gate, numpy array of shape (n_a, 1)
                        `W_v`: Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                        `b_v`: Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
    Returns:
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s - lists of size m containing the computations in each forward pass
    outputs - prediction at timestep "t", numpy array of shape (n_v, m)
    """
    assert h_prev.shape == (hidden_size, 1)
    assert C_prev.shape == (hidden_size, 1)

    # First we unpack our parameters
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p
    
    # Save a list of computations for each of the components in the LSTM
    x_s, z_s, f_s, i_s,  = [], [] ,[], []
    g_s, C_s, o_s, h_s = [], [] ,[], []
    v_s, output_s =  [], [] 
    
    # Append the initial cell and hidden state to their respective lists
    h_s.append(h_prev)
    C_s.append(C_prev)
    
    for x in inputs:
        # Concatenate input and hidden state
        z = np.row_stack((h_prev, x))
        z_s.append(z)
        
        # Calculate forget gate
        f = sigmoid(np.dot(W_f, z) + b_f)
        f_s.append(f)
        
        # Calculate input gate
        i = sigmoid(np.dot(W_i, z) + b_i)
        i_s.append(i)
        
        # Calculate candidate
        g = tanh(np.dot(W_g, z) + b_g)
        g_s.append(g)

        # Calculate memory state
        C_prev = f * C_prev + i * g 
        C_s.append(C_prev)
        
        # Calculate output gate
        o = sigmoid(np.dot(W_o, z) + b_o)
        o_s.append(o)
        
        # Calculate hidden state
        h_prev = o * tanh(C_prev)
        h_s.append(h_prev)

        # Calculate logits
        v = np.dot(W_v, h_prev) + b_v
        v_s.append(v)
        
        # Calculate softmax
        output = softmax(v)
        output_s.append(output)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s


def clip_gradient_norm(grads, max_norm=0.25):
    """
    Clips gradients to have a maximum norm of `max_norm`.
    This is to prevent the exploding gradients problem.
    """ 
    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0
    
    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm
    
    total_norm = np.sqrt(total_norm)
    
    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef
    
    return grads


def backward_pass(hidden_size, z, f, i, g, C, o, h, v, outputs, targets, p):
    """
    Arguments:
    `hidden_size`: number of dimensions in the hidden state
    `z`: your concatenated input data  as a list of size m.
    `f`: your forget gate computations as a list of size m.
    `i`: your input gate computations as a list of size m.
    `g`: your candidate computations as a list of size m.
    `C`: your Cell states as a list of size m+1.
    `o`: your output gate computations as a list of size m.
    `h`: your Hidden state computations as a list of size m+1.
    `v`: your logit computations as a list of size m.
    `outputs`: your outputs as a list of size m.
    `targets`: your targets as a list of size m.
    `p`: python list containing:
                        `W_f`: Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        `b_f`: Bias of the forget gate, numpy array of shape (n_a, 1)
                        `W_i`: Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        `b_i`: Bias of the update gate, numpy array of shape (n_a, 1)
                        `W_g`: Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        `b_g`: Bias of the first "tanh", numpy array of shape (n_a, 1)
                        `W_o`: Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        `b_o`: Bias of the output gate, numpy array of shape (n_a, 1)
                        `W_v`: Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                        `b_v`: Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
    Returns:
    loss - crossentropy loss for all elements in output
    grads - lists of gradients of every element in p
    """

    # Unpack parameters
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p

    # Initialize gradients as zero
    W_f_d = np.zeros_like(W_f)
    b_f_d = np.zeros_like(b_f)

    W_i_d = np.zeros_like(W_i)
    b_i_d = np.zeros_like(b_i)

    W_g_d = np.zeros_like(W_g)
    b_g_d = np.zeros_like(b_g)

    W_o_d = np.zeros_like(W_o)
    b_o_d = np.zeros_like(b_o)

    W_v_d = np.zeros_like(W_v)
    b_v_d = np.zeros_like(b_v)
    
    # Set the next cell and hidden state equal to zero
    dh_next = np.zeros_like(h[0])
    dC_next = np.zeros_like(C[0])
        
    # Track loss
    loss = 0
    
    for t in reversed(range(len(outputs))):
        # Compute the cross entropy
        loss += -np.mean(np.log(outputs[t]) * targets[t])
        # Get the previous hidden cell state
        C_prev= C[t-1]
        
        # Compute the derivative of the relation of the hidden-state to the output gate
        dv = np.copy(outputs[t])
        dv[np.argmax(targets[t])] -= 1

        # Update the gradient of the relation of the hidden-state to the output gate
        W_v_d += np.dot(dv, h[t].T)
        b_v_d += dv

        # Compute the derivative of the hidden state and output gate
        dh = np.dot(W_v.T, dv)        
        dh += dh_next
        do = dh * tanh(C[t])
        do = sigmoid(o[t], derivative=True)*do
        
        # Update the gradients with respect to the output gate
        W_o_d += np.dot(do, z[t].T)
        b_o_d += do

        # Compute the derivative of the cell state and candidate g
        dC = np.copy(dC_next)
        dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
        dg = dC * i[t]
        dg = tanh(g[t], derivative=True) * dg
        
        # Update the gradients with respect to the candidate
        W_g_d += np.dot(dg, z[t].T)
        b_g_d += dg

        # Compute the derivative of the input gate and update its gradients
        di = dC * g[t]
        di = sigmoid(i[t], True) * di
        W_i_d += np.dot(di, z[t].T)
        b_i_d += di

        # Compute the derivative of the forget gate and update its gradients
        df = dC * C_prev
        df = sigmoid(f[t]) * df
        W_f_d += np.dot(df, z[t].T)
        b_f_d += df

        # Compute the derivative of the input and update the gradients of the previous hidden and cell state
        dz = (np.dot(W_f.T, df)
             + np.dot(W_i.T, di)
             + np.dot(W_g.T, dg)
             + np.dot(W_o.T, do))
        dh_prev = dz[:hidden_size, :]
        dC_prev = f[t] * dC
        
    grads= W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d
    
    # Clip gradients
    grads = clip_gradient_norm(grads)
    
    return loss, grads
