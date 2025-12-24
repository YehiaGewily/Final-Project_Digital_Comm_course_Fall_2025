%% OFDM Project - Phase 4: Pilot-Based Channel Estimation & ZF Equalization (SISO 16-QAM)
% 1x1 SISO, 16-QAM, fixed multipath fading (L=50) + AWGN
% Pilot scheme: 1st OFDM symbol is all pilots (known), rest are data
% Channel estimation: Hest = Ypilot ./ Xpilot
% Equalization: Yeq = Ydata ./ Hest

clear; clc; close all;
rng(1);

%% Parameters
N          = 1024;       % FFT size
L          = 50;         % channel taps
cp_len     = 72;         % CP length
EbNo_dB    = 0:3:40;     % Eb/N0 sweep
numRuns    = 1000;       % runs per Eb/N0
numSymRun  = 100;        % total OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

% 16-QAM
M  = 16;
k0 = log2(M);

% Pilot (known)
pilotSequence = (randn(N,1) + 1i*randn(N,1)) / sqrt(2);  % Random pilot with unit energy

BER = zeros(size(EbNo_dB));
codeRate = 4/7;

%% Main simulation - PARALLELIZED
tic;
parfor e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr = 0; 
    totalBits = 0;
    
    for run = 1:numRuns
        % Generate random channel per run for proper fading simulation
        h = (randn(1,L) + 1i*randn(1,L));
        h = h / norm(h);

        %% ---------- TRANSMITTER ----------
        X = zeros(N, numSymRun);
        X(:,1) = pilotSequence;  % Insert pilot sequence
        
        % 1. Calculate Capacity
        % Total available data subcarriers in the frame
        totalSubcarriers = N * numDataSym;
        nCodedBitsAvailable = totalSubcarriers * k0;
        
        % 2. Bit Alignment
        % Must be multiple of 28 (LCM of 7 for Hamming and 4 for 16-QAM)
        nCodedBits = floor(nCodedBitsAvailable/28) * 28;
        nDataBits  = nCodedBits * codeRate;
        
        txDataBits = randi([0 1], nDataBits, 1);
        
        % 3. Encode
        txCodedBits = encode74(txDataBits);
        
        % 4. Modulate
        dataSyms = qam_gray_mod(txCodedBits, M);
        
        % 5. Frame Assembly (with Padding)
        % We have 'length(dataSyms)' symbols, but grid has 'totalSubcarriers' spots.
        % Pad the rest with zeros (or random noise) to fill the grid.
        numValidSyms = length(dataSyms);
        padding = zeros(totalSubcarriers - numValidSyms, 1);
        
        fullGridVec = [dataSyms; padding];
        X(:, 2:end) = reshape(fullGridVec, N, numDataSym);
        
        % 6. OFDM Modulation
        x_time = ifft(X, N, 1);
        x_cp = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        %% ---------- CHANNEL SIMULATION ----------
        % Calculate Noise
        Ps_tx = 1; % Normalized transmit power
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        sigma2 = Ps_tx / SNRlin;
        
        % Apply Fading
        rx_serial_clean = filter(h, 1, tx_serial);
        
        % Add AWGN
        noise = sqrt(sigma2/2) * (randn(size(rx_serial_clean)) + 1i*randn(size(rx_serial_clean)));
        rx_serial = rx_serial_clean + noise;
        
        %% ---------- RECEIVER ----------
        % 1. CP Removal and FFT
        rx_parallel = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp = rx_parallel(cp_len+1:end, :);       
        Yf = fft(rx_no_cp, N, 1);                          
        
        % 2. Channel Estimation
        % Use the known pilot sequence to estimate H
        H_est = Yf(:,1) ./ X(:,1);
        
        % Replicate for all data columns
        H_grid = repmat(H_est, 1, numDataSym);
        
        % Regularize
        H_grid(abs(H_grid) < 1e-12) = 1e-12;                   
        
        % 3. Equalization (Zero-Forcing)
        Y_data = Yf(:, 2:end);
        Yeq = Y_data ./ H_grid;
        
        % 4. Extract Valid Symbols (Discard Padding)
        Yeq_vec = Yeq(:);
        Y_valid = Yeq_vec(1:numValidSyms);
        
        % 5. Demap & Decode
        rxCodedBits = qam_gray_demod(Y_valid, M);
        rxDecodedBits = decode74(rxCodedBits); 
        
        %% ---------- BER CALCULATION ----------
        % Lengths should now match exactly
        err = sum(txDataBits ~= rxDecodedBits);
        
        totalErr = totalErr + err; 
        totalBits = totalBits + length(txDataBits);
    end
    
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end
elapsed = toc;

%% Print results
fprintf('\n========== SIMULATION COMPLETE ==========\n');
fprintf('Total Time: %.2f seconds\n\n', elapsed);

%% Plot
figure; semilogy(EbNo_dB, BER, 'r-o', 'LineWidth', 1.5); grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('16-QAM OFDM SISO: Fixed Multipath + AWGN (Pilot Hest + ZF EQ)');
legend('1x1 fading + 16-QAM EQ');
ylim([1e-5 1]);

%% -------------------- Helper Functions --------------------
function syms = qam_gray_mod(bits, M)
    % Square QAM Gray mapping, normalized to theoretical unit average energy.
    k = log2(M);
    if mod(numel(bits), k) ~= 0
        error("Bit length must be multiple of log2(M).");
    end
    m = round(sqrt(M));
    if m*m ~= M
        error("M must be a perfect square (4,16,64...).");
    end
    bps = k/2;
    bits = bits(:);
    B = reshape(bits, k, []).';        % [nSyms x k]
    bI = B(:,1:bps);
    bQ = B(:,bps+1:end);
    gI = bi2int(bI);                  % Gray index
    gQ = bi2int(bQ);
    iI = gray2bin_int(gI);            % binary index
    iQ = gray2bin_int(gQ);
    aI = 2*iI - (m-1);                % PAM levels
    aQ = 2*iQ - (m-1);
    % Theoretical normalization for square QAM:
    % E[|aI + j aQ|^2] = 2*(m^2-1)/3
    normFactor = sqrt(2*(m^2-1)/3);
    syms = (aI + 1i*aQ) / normFactor; % unit average energy
end

function bits = qam_gray_demod(syms, M)
    % Hard decision Gray demapper for square QAM.
    k = log2(M);
    m = round(sqrt(M));
    if m*m ~= M
        error("M must be a perfect square (4,16,64...).");
    end
    bps = k/2;
    % Undo normalization back to integer-grid levels
    normFactor = sqrt(2*(m^2-1)/3);
    syms = syms * normFactor;
    I = real(syms);
    Q = imag(syms);
    levels = (-(m-1):2:(m-1));        % allowed PAM levels
    idxI = slicer_to_index(I, levels); % binary index 0..m-1
    idxQ = slicer_to_index(Q, levels);
    gI = bin2gray_int(idxI);           % Gray index
    gQ = bin2gray_int(idxQ);
    bI = int2bits(gI, bps);            % Gray bits
    bQ = int2bits(gQ, bps);
    bitsMat = [bI bQ];                 % [nSyms x k]
    bits = reshape(bitsMat.', [], 1);
end

function idx = slicer_to_index(x, levels)
    % OPTIMIZED: vectorized slicing with proper broadcasting
    x = x(:);  % Ensure column vector
    levels = levels(:).';  % Ensure row vector
    [~, idx] = min(abs(x - levels), [], 2);  % Find nearest level per symbol
    idx = idx - 1;  % 0-based indexing (0 to m-1)
end

function v = bi2int(B)
    b = size(B,2);
    w = 2.^(b-1:-1:0);
    v = B * w.';
end

function B = int2bits(v, b)
    v = v(:);
    n = numel(v);
    B = zeros(n, b);
    for i = 1:b
        B(:,i) = bitget(v, b - i + 1);
    end
end

function b = gray2bin_int(g)
    g = uint32(g);
    b = g;
    while any(g)
        g = bitshift(g, -1);
        b = bitxor(b, g);
    end
    b = double(b);
end

function g = bin2gray_int(b)
    b = uint32(b);
    g = bitxor(b, bitshift(b, -1));
    g = double(g);
end