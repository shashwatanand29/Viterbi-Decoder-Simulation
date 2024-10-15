% Viterbi Decoder Simulation Program

% Parameters
constraint_length = 3;
generator_matrix = [1 1 1; 1 0 1]; % Rate 1/2 convolutional code
SNR_dB = 5; % Set one SNR value for practical simulation

% Prompt user for input message
message = input('Enter a binary message as a vector of 0s and 1s (e.g., [1 0 1 1 0]): ');

% Main simulation for a practical example
[BER, decoded_message, noisy_signal, modulated_signal] = simulate_viterbi_practical(message, generator_matrix, constraint_length, SNR_dB);

% Display original message
disp('Original Message:');
disp(message);

% Display decoded message
disp('Decoded Message:');
disp(decoded_message);

% Plot the message transmission and decoding
figure;
subplot(3,1,1);
stem(message, 'filled');
title('Original Binary Message');
ylim([-0.5 1.5]);
grid on;

subplot(3,1,2);
plot(modulated_signal, 'LineWidth', 1.5);
title('BPSK Modulated Signal');
grid on;

subplot(3,1,3);
plot(noisy_signal, 'LineWidth', 1.5);
title('Noisy Received Signal');
grid on;

% BER Result
disp(['Bit Error Rate (BER): ', num2str(BER)]);

% Simulation function for practical use
function [ber, decoded_message, noisy_signal, modulated_signal] = simulate_viterbi_practical(message, generator_matrix, constraint_length, SNR_dB)
    % Encode message
    encoded_message = convolutional_encode(message, generator_matrix, constraint_length);
    
    % Simulate BPSK modulation and AWGN channel
    modulated_signal = 2 * encoded_message - 1; % BPSK modulation
    noisy_signal = awgn(modulated_signal, SNR_dB, 'measured');
    
    % Demodulate (soft decision)
    demodulated_signal = noisy_signal;
    
    % Viterbi decoding
    decoded_message = viterbi_decoder(demodulated_signal, generator_matrix, constraint_length);
    
    % Calculate BER
    ber = sum(decoded_message ~= message) / length(message);
end

% Convolutional Encoder
function encoded_bits = convolutional_encode(message, generator_matrix, constraint_length)
    [num_outputs, ~] = size(generator_matrix);
    message_length = length(message);
    encoded_bits = zeros(1, num_outputs * message_length);
    
    % Initialize shift register
    shift_register = zeros(1, constraint_length);
    
    for i = 1:message_length
        % Shift in new bit
        shift_register = [message(i), shift_register(1:end-1)];
        
        % Generate output bits
        for j = 1:num_outputs
            encoded_bits((i-1)*num_outputs + j) = mod(sum(shift_register .* generator_matrix(j,:)), 2);
        end
    end
end

% Viterbi Decoder
function decoded_bits = viterbi_decoder(received_signal, generator_matrix, constraint_length)
    [num_outputs, ~] = size(generator_matrix);
    num_states = 2^(constraint_length - 1);
    num_symbols = length(received_signal) / num_outputs;
    
    % Initialize trellis
    trellis = initialize_trellis(generator_matrix, constraint_length);
    
    % Initialize path metrics and survivors
    path_metrics = zeros(num_states, num_symbols + 1);
    path_metrics(2:end, 1) = Inf; % Set initial path metrics for non-zero states to infinity
    survivors = zeros(num_states, num_symbols);
    
    % Forward pass
    for i = 1:num_symbols
        received = received_signal((i-1)*num_outputs+1 : i*num_outputs);
        for state = 0:num_states-1
            for input = 0:1
                next_state = trellis(input+1).next_state(state+1);
                output = trellis(input+1).output(state+1, :);
                branch_metric = sum((received - (2*output-1)).^2);
                
                new_metric = path_metrics(state+1, i) + branch_metric;
                if new_metric < path_metrics(next_state+1, i+1)
                    path_metrics(next_state+1, i+1) = new_metric;
                    survivors(next_state+1, i) = state;
                end
            end
        end
    end
    
    % Traceback
    [~, best_state] = min(path_metrics(:, end));
    decoded_bits = zeros(1, num_symbols);
    
    for i = num_symbols:-1:1
        prev_state = survivors(best_state, i);
        decoded_bits(i) = bitand(best_state-1, 2^(constraint_length-2)) > 0;
        best_state = prev_state + 1;
    end
end

% Initialize Trellis
function trellis = initialize_trellis(generator_matrix, constraint_length)
    num_states = 2^(constraint_length - 1);
    [num_outputs, ~] = size(generator_matrix);
    
    trellis = struct('next_state', cell(1,2), 'output', cell(1,2));
    
    for input = 0:1
        trellis(input+1).next_state = zeros(1, num_states);
        trellis(input+1).output = zeros(num_states, num_outputs);
        
        for state = 0:num_states-1
            next_state = bitshift(state, -1);
            next_state = bitor(bitshift(input, constraint_length-2), next_state);
            trellis(input+1).next_state(state+1) = next_state;
            
            reg = [input, de2bi(state, constraint_length-1, 'left-msb')];
            for j = 1:num_outputs
                trellis(input+1).output(state+1, j) = mod(sum(reg .* generator_matrix(j,:)), 2);
            end
        end
    end
end
