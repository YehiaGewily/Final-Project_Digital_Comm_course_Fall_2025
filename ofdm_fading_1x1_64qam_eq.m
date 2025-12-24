%% OFDM Project - Phase 4: Pilot-Based Channel Estimation & ZF Equalization (SISO 64-QAM)
% 1x1 SISO, 64-QAM, fixed multipath fading (L=50) + AWGN
% Pilot scheme: 1st OFDM symbol is all pilots (known), rest are data
% Channel estimation: Hest = Ypilot ./ Xpilot
% Equalization: Yeq = Ydata ./ Hest
% Expected: BER improves significantly vs. "no EQ" fading case.

clear; clc; close all;
rng(1);

%% Parameters
N          = 1024;       % FFT size
L          = 50;         % channel taps
cp_len     = 72;         % CP length
EbNo_dB    = 0:3:40;     % Eb/N0 sweep
numRuns    = 50;        % runs per Eb/N0
numSymRun  = 100;        % total OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

% 64-QAM
M  = 64;
k0 = log2(M);

% Pilot (known) - Use varying pilot across subcarriers
pilotSequence = (randn(N,1) + 1i*randn(N,1)) / sqrt(2);  % Random pilot with unit energy

% Fixed deterministic channel realization
% Normalized to unit power
h = (randn(1,L) + 1i*randn(1,L));
h = h / norm(h);

% Pre-compute ideal channel frequency response (for debugging)
H_ideal = fft([h, zeros(1, N-L)], N, 1).';  % [N x 1] column vector

BER = zeros(size(EbNo_dB));
codeRate = 4/7;

%% Main simulation - PARALLELIZED
tic;
parfor e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr = 0; totalBits = 0;
    
    for run = 1:numRuns
        %% ---------- TRANSMITTER ----------
        X = zeros(N, numSymRun);
        X(:,1) = pilotSequence;  % Insert pilot sequence (CHANGED)
        
        % 1. Calculate how many RAW bits we need (4/7ths of the coded capacity)
        nCodedBitsAvailable = N * k0 * numDataSym;
        
        % FIXED: Ensure multiple of LCM(7, k0) for proper alignment
        % For 64-QAM: k0=6, LCM(7,6)=42
        lcm_val = lcm(7, k0);
        nCodedBitsNeeded = floor(nCodedBitsAvailable / lcm_val) * lcm_val;
        
        % Calculate raw bits
        numRawBits = nCodedBitsNeeded * codeRate;
        
        txDataBits = randi([0 1], numRawBits, 1);
        
        % 2. ENCODE the bits using (7,4) Linear Block Code
        txCodedBits = encode74(txDataBits); 
        
        % CRITICAL FIX: Truncate to exact multiple of k0 BEFORE reshape
        nSymsAvailable = floor(length(txCodedBits) / k0);
        nSymsPerSubframe = floor(nSymsAvailable / N) * N;  % Must be multiple of N
        nBitsToUse = nSymsPerSubframe * k0;
        
        txCodedBits = txCodedBits(1:nBitsToUse);
        
        % 3. Map bits to symbols and pack the grid
        dataSyms = qam_gray_mod(txCodedBits, M);
        
        % Now reshape will work because length(dataSyms) = nSymsPerSubframe = multiple of N
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        % 4. OFDM Modulation (IFFT) and Cyclic Prefix
        x_time = ifft(X, N, 1);
        x_cp = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        %% ---------- CHANNEL SIMULATION ----------
        % Calculate Noise and Apply Fading
        % FIXED: Use normalized transmit power instead of received power
        Ps_tx = 1;  % Normalized transmit power
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        sigma2 = Ps_tx / SNRlin;
        
        % Apply Fading using FILTER
        rx_serial_clean = filter(h, 1, tx_serial);
        
        % Add AWGN - VECTORIZED
        noise = sqrt(sigma2/2) * (randn(size(rx_serial_clean)) + 1i*randn(size(rx_serial_clean)));
        rx_serial = rx_serial_clean + noise;
        
        %% ---------- RECEIVER ----------
        % 1. CP Removal and FFT
        rx_parallel = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp = rx_parallel(cp_len+1:end, :);       
        Yf = fft(rx_no_cp, N, 1);                          
        
        % 2. Channel Estimation and Equalization
        % FIXED: Use pilot-based estimation with varying pilot
        Hest = Yf(:,1) ./ pilotSequence;  % (CHANGED from X(:,1) to pilotSequence)
        
        % Replicate for all data columns
        H_grid = repmat(Hest, 1, colsFilled);
        
        % Regularize
        H_grid(abs(H_grid) < 1e-12) = 1e-12;                   
        
        % Equalize
        Y_data = Yf(:, 2:1+colsFilled);
        Yeq = Y_data ./ H_grid;                         
        
        % 3. Demapping (Symbol -> Coded Bits)
        rxSyms = Yeq(:);
        rxCodedBits = qam_gray_demod(rxSyms, M);
        
        % 4. DECODE the bits back to raw data 
        rxDecodedBits = decode74(rxCodedBits); 
        
        %% ---------- BIT ERROR RATE (BER) ----------
        % Compare decoded bits to original raw bits
        nCompare = min(length(txDataBits), length(rxDecodedBits));
        err = sum(txDataBits(1:nCompare) ~= rxDecodedBits(1:nCompare));
        
        totalErr = totalErr + err; 
        totalBits = totalBits + nCompare;
    end
    BER(e) = totalErr / totalBits;
end

elapsed = toc;

%% Print results
fprintf('\n========== SIMULATION COMPLETE ==========\n');
fprintf('Total Time: %.2f seconds\n\n', elapsed);
for e = 1:length(EbNo_dB)
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end

%% Plot
figure; semilogy(EbNo_dB, BER, 'k-s', 'LineWidth', 1.5); grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('64-QAM OFDM SISO: Fixed Multipath + AWGN (Pilot Hest + ZF EQ)');
legend('1x1 fading + 64-QAM EQ');
grid on; ylim([1e-5 1]);

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
    % OPTIMIZED: vectorized slicing
    x = x(:);
    levels = levels(:).';
    [~, idx] = min(abs(x - levels), [], 2);
    idx = idx - 1;  % 0-based
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