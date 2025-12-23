%% OFDM Project - Phase 3: SISO Fixed Multipath Fading + ZF Equalization
% 1 Tx -> 1 Rx antenna (SISO)
% Channel: Fixed multipath fading (L taps) + AWGN
% Coding: Hamming (7,4) [External functions]
% Pilots: 1st OFDM symbol is all pilots
% Equalization: Zero-Forcing (ZF)

clear; clc; close all;
rng(1);

%% 1. Parameters
N           = 1024;      % FFT size
L           = 50;        % Channel taps
cp_len      = 72;        % CP length (must be > L)
EbNo_dB     = 0:3:30;    % Eb/N0 sweep
numRuns     = 200;       % Runs per Eb/N0
numSymRun   = 50;        % OFDM symbols per run
numPilotSym = 1;         % First symbol is pilot
numDataSym  = numSymRun - numPilotSym;

% Modulation (QPSK)
M  = 4;
k0 = log2(M);

% Pilot Symbol (Known to Rx)
pilotSym = (1+1i)/sqrt(2) * ones(N, 1); 

% Channel Setup: Fixed Fading (SISO)
% Normalized to unit power
h = (randn(1,L) + 1i*randn(1,L));
h = h / norm(h);

BER = zeros(size(EbNo_dB));

%% 2. Main Simulation Loop
fprintf('Starting SISO Simulation...\n');

for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr  = 0;
    totalBits = 0;
    
    % Coding rate for (7,4)
    codeRate = 4/7;
    
    for run = 1:numRuns
        %% --- TRANSMITTER ---
        
        % 1. Data Generation
        nCodedBitsNeeded = N * k0 * numDataSym;
        nCodedBitsNeeded = floor(nCodedBitsNeeded/7) * 7; % Multiple of 7
        nRawBits = nCodedBitsNeeded * codeRate;
        
        txDataBits = randi([0 1], nRawBits, 1);
        
        % 2. Coding (External Function)
        txCodedBits = encode74(txDataBits);
        
        % 3. Mapping
        dataSyms = qam_gray_mod(txCodedBits, M);
        
        % 4. Frame Assembly
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym; % Symbol 1 is Pilot
        
        % Fill data
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        % 5. IFFT & CP
        x_time = ifft(X, N, 1);
        x_cp   = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        %% --- CHANNEL (SISO) ---
        
        % Calculate Noise
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        sigPower = mean(abs(tx_serial).^2);
        noisePower = sigPower / SNRlin;
        
        % Apply Fading (Filter across serial stream)
        rx_serial_clean = filter(h, 1, tx_serial);
        
        % Add AWGN
        noiseScale = sqrt(noisePower/2);
        noise = noiseScale * (randn(size(rx_serial_clean)) + 1i*randn(size(rx_serial_clean)));
        rx_serial = rx_serial_clean + noise;
        
        %% --- RECEIVER ---
        
        % 1. Parallelization & CP Removal
        rx_mat = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp = rx_mat(cp_len+1:end, :);
        
        % 2. FFT
        Y = fft(rx_no_cp, N, 1);
        
        % 3. Channel Estimation (Using Pilot)
        % H_est = Y_pilot / X_pilot
        H_est = Y(:,1) ./ X(:,1);
        
        % Replicate estimate for all data symbols (Static Channel)
        H_grid = repmat(H_est, 1, colsFilled);
        
        % 4. Zero-Forcing Equalization
        % Y_eq = Y_data / H_est
        Y_data = Y(:, 2:1+colsFilled);
        
        % Protect against divide by zero/noise enhancement
        % (Simple regularization)
        H_grid(abs(H_grid) < 1e-10) = 1e-10; 
        
        Y_equalized = Y_data ./ H_grid;
        
        % 5. Demapping
        rxCodedBits = qam_gray_demod(Y_equalized(:), M);
        
        % 6. Decoding (External Function)
        rxDecodedBits = decode74(rxCodedBits);
        
        % 7. Error Counting
        len = min(length(txDataBits), length(rxDecodedBits));
        err = sum(txDataBits(1:len) ~= rxDecodedBits(1:len));
        
        totalErr  = totalErr + err;
        totalBits = totalBits + len;
    end
    
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB | BER = %.5e\n', EbNo_dB(e), BER(e));
end

%% 3. Visualization
figure;
semilogy(EbNo_dB, BER, 'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
grid on;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('OFDM SISO 1x1 with Fixed Fading & ZF Equalization');
legend('SISO (Estimated Channel)', 'Location', 'southwest');
ylim([1e-5 1]);

%% -------------------- Helper Functions (QAM Only) --------------------

function syms = qam_gray_mod(bits, M)
    k = log2(M);
    m = round(sqrt(M));
    bps  = k/2;
    bits = bits(:);
    B = reshape(bits, k, []).';     
    bI = B(:,1:bps);
    bQ = B(:,bps+1:end);
    gI = bi2int(bI);
    gQ = bi2int(bQ);
    iI = gray2bin_int(gI);
    iQ = gray2bin_int(gQ);
    aI = 2*iI - (m-1);
    aQ = 2*iQ - (m-1);
    normFactor = sqrt(2*(m^2-1)/3);
    syms = (aI + 1i*aQ) / normFactor;
end

function bits = qam_gray_demod(syms, M)
    k = log2(M);
    m = round(sqrt(M));
    bps = k/2;
    normFactor = sqrt(2*(m^2-1)/3);
    syms = syms * normFactor;
    I = real(syms);
    Q = imag(syms);
    levels = (-(m-1):2:(m-1));
    idxI = slicer_to_index(I, levels);  
    idxQ = slicer_to_index(Q, levels);
    gI = bin2gray_int(idxI);
    gQ = bin2gray_int(idxQ);
    bI = int2bits(gI, bps);
    bQ = int2bits(gQ, bps);
    bitsMat = [bI bQ];
    bits = reshape(bitsMat.', [], 1);
end

function idx = slicer_to_index(x, levels)
    idx = zeros(size(x));
    for n = 1:numel(x)
        [~, ii] = min(abs(x(n) - levels));
        idx(n) = ii - 1; 
    end
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