%% OFDM Project - Phase 5B: SIMO 1x2 Fixed Fading + Pilot Estimation + MRC (16-QAM)
% 1 Tx -> 2 Rx antennas (SIMO) using 16-QAM
% Channel: Fixed multipath fading (L taps) + AWGN
% Coding: Hamming (7,4) (External functions)
% Pilots: 1st OFDM symbol is all pilots
% Combining: MRC in frequency domain

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

% 16-QAM
M  = 16;
k0 = log2(M);

% Pilot (known)
pilotSym = (1+1i)/sqrt(2) * ones(N, 1); 

% Two independent fixed fading channels
h1 = (randn(1,L) + 1i*randn(1,L));
h1 = h1 / norm(h1);
h2 = (randn(1,L) + 1i*randn(1,L));
h2 = h2 / norm(h2);

BER = zeros(size(EbNo_dB));

%% Main simulation (Integrated SIMO 1x2 + Channel Coding)
for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr = 0; totalBits = 0;
    
    % Code Rate
    codeRate = 4/7;

    for run = 1:numRuns
        %% ---------- TRANSMITTER ----------
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym;  % First OFDM symbol is for pilots
        
        % 1. Calculate RAW bit requirements
        nCodedBitsAvailable = N * k0 * numDataSym;
        
        % Ensure multiple of 7 for Hamming
        nCodedBitsNeeded = floor(nCodedBitsAvailable/7) * 7;
        
        % Calculate raw bits
        numRawBits = nCodedBitsNeeded * codeRate;
        
        txDataBits = randi([0 1], numRawBits, 1); 

        % 2. ENCODE (Channel Coding)
        txCodedBits = encode74(txDataBits); 
        
        % 3. Map bits to symbols
        dataSyms = qam_gray_mod(txCodedBits, M);               
        
        % Fill grid
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        % 4. IFFT and Cyclic Prefix (CP) addition
        x_time = ifft(X, N, 1);
        x_cp = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        %% ---------- CHANNEL SIMULATION (SIMO 1x2) ----------
        % Calculate Noise variance
        % MUST account for Code Rate (4/7)
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        
        Ps = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;
        
        % Apply two independent fading channels (h1 and h2) using FILTER
        rx1_serial_clean = filter(h1, 1, tx_serial);
        rx2_serial_clean = filter(h2, 1, tx_serial);
        
        % Add independent AWGN to both antenna branches
        n1 = sqrt(sigma2/2) * (randn(size(rx1_serial_clean)) + 1i*randn(size(rx1_serial_clean)));
        n2 = sqrt(sigma2/2) * (randn(size(rx2_serial_clean)) + 1i*randn(size(rx2_serial_clean)));
        
        rx1_serial = rx1_serial_clean + n1; 
        rx2_serial = rx2_serial_clean + n2;
        
        %% ---------- RECEIVER ----------
        % 1. CP Removal and FFT for both branches
        rx1_mat = reshape(rx1_serial, N+cp_len, numSymRun);
        rx2_mat = reshape(rx2_serial, N+cp_len, numSymRun);
        
        rx1_no_cp = rx1_mat(cp_len+1:end, :);
        rx2_no_cp = rx2_mat(cp_len+1:end, :);
        
        Y1f = fft(rx1_no_cp, N, 1); 
        Y2f = fft(rx2_no_cp, N, 1);
        
        % 2. Channel Estimation from Pilots
        H1_est = Y1f(:,1) ./ X(:,1); 
        H2_est = Y2f(:,1) ./ X(:,1);
        
        % Replicate estimate for all data symbols
        H1_grid = repmat(H1_est, 1, colsFilled);
        H2_grid = repmat(H2_est, 1, colsFilled);
        
        % 3. MRC Combining (Maximum Ratio Combining)
        Y1_data = Y1f(:, 2:1+colsFilled);
        Y2_data = Y2f(:, 2:1+colsFilled);
        
        numerator = conj(H1_grid).*Y1_data + conj(H2_grid).*Y2_data;
        denominator = abs(H1_grid).^2 + abs(H2_grid).^2;
        denominator(denominator < 1e-10) = 1e-10;
        
        Y_equalized = numerator ./ denominator;
        
        % 4. Demapping (Symbols -> Coded Bits)
        rxCodedBits = qam_gray_demod(Y_equalized(:), M);
        
        % 5. DECODE (Channel Decoding)
        rxDecodedBits = decode74(rxCodedBits); 
        
        %% ---------- BIT ERROR RATE (BER) ----------
        % Compare recovered information against original input
        nCompare = min(length(txDataBits), length(rxDecodedBits));
        err = sum(txDataBits(1:nCompare) ~= rxDecodedBits(1:nCompare));
        
        totalErr = totalErr + err; 
        totalBits = totalBits + nCompare;
    end
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end

%% Plot
figure; semilogy(EbNo_dB, BER, 'm-d', 'LineWidth', 1.5); grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('16-QAM OFDM SIMO 1x2: Fixed Multipath + AWGN (Pilot Hest + MRC)');
legend('1x2 SIMO 16-QAM MRC');

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
    bps  = k/2;
    bits = bits(:);
    B = reshape(bits, k, []).';     % [nSyms x k]
    bI = B(:,1:bps);
    bQ = B(:,bps+1:end);
    gI = bi2int(bI);
    gQ = bi2int(bQ);
    iI = gray2bin_int(gI);
    iQ = gray2bin_int(gQ);
    aI = 2*iI - (m-1);
    aQ = 2*iQ - (m-1);
    % Theoretical normalization for square QAM:
    % E[|aI + j aQ|^2] = 2*(m^2-1)/3
    normFactor = sqrt(2*(m^2-1)/3);
    syms = (aI + 1i*aQ) / normFactor;
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
    levels = (-(m-1):2:(m-1));
    idxI = slicer_to_index(I, levels);  % binary index 0..m-1
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
        idx(n) = ii - 1; % 0-based
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
    B = zeros(n,b);
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