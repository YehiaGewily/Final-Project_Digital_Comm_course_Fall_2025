%% OFDM Project - Phase 5B: SIMO 1x2 Fixed Fading + Pilot Estimation + MRC (64-QAM)
% 1 Tx -> 2 Rx antennas (SIMO) using 64-QAM
% Channel: Fixed multipath fading (L taps) + AWGN
% Coding: Hamming (7,4)
% Pilots: 1st OFDM symbol is all pilots
% Combining: MRC (Maximum Ratio Combining)

clear; clc; close all;
rng(1);

%% Parameters
N          = 1024;      % FFT size
L          = 50;        % channel taps
cp_len     = 72;        % CP length
EbNo_dB    = 0:3:40;    % Eb/N0 sweep
numRuns    = 1000;      % runs per Eb/N0
numSymRun  = 100;       % total OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

% 64-QAM
M  = 64;
k0 = log2(M); % 6 bits per symbol

% Pilot (known)
pilotSym = (1+1i)/sqrt(2) * ones(N, 1); 

% Two independent fixed fading channels (Normalized to unit power)
h1 = (randn(1,L) + 1i*randn(1,L)); h1 = h1 / norm(h1);
h2 = (randn(1,L) + 1i*randn(1,L)); h2 = h2 / norm(h2);

BER = zeros(size(EbNo_dB));

%% Main simulation
for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr = 0; totalBits = 0;
    codeRate = 4/7;

    for run = 1:numRuns
        %% ---------- TRANSMITTER ----------
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym; 
        
        % 1. Data Generation & Hamming Encoding
        nCodedBitsAvailable = N * k0 * numDataSym;
        nBlocks = floor(nCodedBitsAvailable / 7);
        numRawBits = nBlocks * 4;
        
        txDataBits = randi([0 1], numRawBits, 1);
        txCodedBits = encode74(txDataBits); 
        
        % --- Padding: Ensure length is multiple of k0 (6) ---
        remainder = mod(length(txCodedBits), k0);
        if remainder ~= 0
            padding = zeros(k0 - remainder, 1);
            txCodedBits = [txCodedBits; padding];
        end
        
        % 2. Modulation
        dataSyms = qam_gray_mod(txCodedBits, M);                
        
        % Fill grid
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        % 3. IFFT and CP addition
        x_time = ifft(X, N, 1);
        x_cp = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        %% ---------- CHANNEL (SIMO 1x2) ----------
        % Calculate Noise variance
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        Ps = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;
        
        % Fading + Noise per antenna
        rx1_s = filter(h1, 1, tx_serial) + sqrt(sigma2/2)*(randn(size(tx_serial)) + 1i*randn(size(tx_serial)));
        rx2_s = filter(h2, 1, tx_serial) + sqrt(sigma2/2)*(randn(size(tx_serial)) + 1i*randn(size(tx_serial)));
        
        %% ---------- RECEIVER ----------
        rx1_mat = reshape(rx1_s, N+cp_len, numSymRun);
        rx2_mat = reshape(rx2_s, N+cp_len, numSymRun);
        
        Y1f = fft(rx1_mat(cp_len+1:end, :), N, 1); 
        Y2f = fft(rx2_mat(cp_len+1:end, :), N, 1);
        
        % Channel Estimation
        H1_est = Y1f(:,1) ./ X(:,1); 
        H2_est = Y2f(:,1) ./ X(:,1);
        
        % MRC Combining
        
        Y1_d = Y1f(:, 2:1+colsFilled);
        Y2_d = Y2f(:, 2:1+colsFilled);
        H1_g = repmat(H1_est, 1, colsFilled);
        H2_g = repmat(H2_est, 1, colsFilled);
        
        num_mrc = conj(H1_g).*Y1_d + conj(H2_g).*Y2_d;
        den_mrc = abs(H1_g).^2 + abs(H2_g).^2;
        Y_eq = num_mrc ./ (den_mrc + 1e-12);
        
        % Demapping & Decoding
        rxCodedBits = qam_gray_demod(Y_eq(:), M);
        
        % --- Remove Padding before Decoding ---
        rxCodedBits = rxCodedBits(1:nBlocks*7);
        rxDecodedBits = decode74(rxCodedBits); 
        
        % Error Counting
        totalErr = totalErr + sum(txDataBits ~= rxDecodedBits);
        totalBits = totalBits + length(txDataBits);
    end
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end

%% ---------- SAVE RESULTS FOR REPORT ----------
save('results_simo_64qam_fading.mat', 'EbNo_dB', 'BER');
fprintf('Results saved to results_simo_64qam_fading.mat\n');

%% Plot
figure; semilogy(EbNo_dB, BER, 'm-d', 'LineWidth', 1.5); grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('64-QAM OFDM SIMO 1x2: Fading + MRC');

%% -------------------- Robust Helper Functions --------------------

function coded = encode74(bits)
    G = [1 1 0 1 0 0 0; 0 1 1 0 1 0 0; 1 1 1 0 0 1 0; 1 0 1 0 0 0 1];
    nBlocks = length(bits)/4;
    bMat = reshape(bits, 4, []).';
    codedMat = mod(bMat * G, 2);
    coded = reshape(codedMat.', [], 1);
end

function decoded = decode74(coded)
    H = [1 0 0 1 0 1 1; 0 1 0 1 1 1 0; 0 0 1 0 1 1 1];
    nBlocks = length(coded)/7;
    cMat = reshape(coded, 7, []).';
    decodedMat = zeros(nBlocks, 4);
    for i = 1:nBlocks
        block = cMat(i, :).';
        syn = mod(H * block, 2);
        if any(syn)
            for col = 1:7
                if all(H(:,col) == syn)
                    block(col) = mod(block(col) + 1, 2);
                    break;
                end
            end
        end
        decodedMat(i, :) = block(4:7).'; 
    end
    decoded = reshape(decodedMat.', [], 1);
end

function syms = qam_gray_mod(bits, M)
    k = log2(M); m = round(sqrt(M));
    B = reshape(bits, k, []).'; 
    gI = bi2int_manual(B(:,1:k/2)); 
    gQ = bi2int_manual(B(:,k/2+1:end));
    iI = gray2bin_manual(gI); 
    iQ = gray2bin_manual(gQ);
    aI = 2*iI - (m-1); aQ = 2*iQ - (m-1);
    syms = (aI + 1i*aQ) / sqrt(2*(M-1)/3);
end

function bits = qam_gray_demod(syms, M)
    k = log2(M); m = round(sqrt(M));
    syms = syms * sqrt(2*(M-1)/3);
    I = real(syms); Q = imag(syms);
    levels = (-(m-1):2:(m-1));
    idxI = zeros(length(I), 1); idxQ = zeros(length(Q), 1);
    for n = 1:length(I)
        [~, ii] = min(abs(I(n) - levels)); idxI(n) = ii - 1;
        [~, iq] = min(abs(Q(n) - levels)); idxQ(n) = iq - 1;
    end
    gI = bin2gray_manual(idxI); gQ = bin2gray_manual(idxQ);
    bI = int2bits_manual(gI, k/2); bQ = int2bits_manual(gQ, k/2);
    bits = reshape([bI bQ].', [], 1);
end

function v = bi2int_manual(B)
    % Manual binary to integer to avoid dimension errors
    [rows, cols] = size(B);
    v = zeros(rows, 1);
    for c = 1:cols
        v = v + B(:,c) .* 2^(cols-c);
    end
end

function B = int2bits_manual(v, b)
    % Manual integer to bits using bitget safely
    v = uint32(v);
    B = zeros(length(v), b);
    for i = 1:b
        B(:,i) = double(bitget(v, b-i+1));
    end
end

function b = gray2bin_manual(g)
    g = uint32(g); b = g;
    while any(g > 0)
        g = bitshift(g, -1);
        b = bitxor(b, g);
    end
    b = double(b);
end

function g = bin2gray_manual(b)
    g = double(bitxor(uint32(b), bitshift(uint32(b), -1)));
end
