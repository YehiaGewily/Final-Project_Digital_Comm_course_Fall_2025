%% OFDM Project - Phase 5A: SIMO 1x2 over AWGN (Equal-Gain Combining)
% 1 Tx -> 2 Rx antennas
% Channel: AWGN only (no fading)
% Receiver: CP remove per branch -> time-domain combine (sum) -> FFT -> demap -> BER

clear; clc; close all;
rng(1);

%% Parameters
N          = 1024;        % FFT size
cp_len     = 72;          % CP length
EbNo_dB    = 0:3:40;      % Eb/N0 sweep (dB)
numRuns    = 1000;        % runs per Eb/N0
numSymRun  = 100;         % OFDM symbols per run

% QPSK
M  = 4;
k0 = log2(M);

% Pilot (needed for structure, even if unused in pure AWGN)
pilotSym = (1+1i)/sqrt(2) * ones(N, 1);

BER = zeros(size(EbNo_dB));

%% Main simulation
for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr  = 0;
    totalBits = 0;
    
    % Code Rate
    codeRate = 4/7;

    for run = 1:numRuns
        %% ---------- TRANSMITTER ----------
        % 1. Build the frequency-domain grid
        X = zeros(N, numSymRun);
        
        % 2. Insert Pilot Symbols
        X(:,1) = pilotSym; 
        
        % 3. Calculate Bit Requirements for (7,4) Channel Coding
        numDataSym = numSymRun - 1; 
        
        % Capacity in coded bits
        nCodedBitsAvailable = N * k0 * numDataSym; 
        
        % Ensure multiple of 7 for Hamming
        nCodedBitsNeeded = floor(nCodedBitsAvailable/7) * 7;
        
        % Calculate raw bits
        numRawBits = nCodedBitsNeeded * codeRate;
        
        txDataBits = randi([0 1], numRawBits, 1); 
        
        % 4. ADD CHANNEL CODING (External Function)
        txCodedBits = encode74(txDataBits); 
        
        % 5. Map the CODED bits to symbols
        dataSyms = qam_gray_mod(txCodedBits, M);               
        
        % 6. Pack symbols into the OFDM grid
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        % 7. OFDM modulation (IFFT)
        x_time = ifft(X, N, 1);                           
        
        % 8. Add Cyclic Prefix (CP) and serialize
        x_cp = [x_time(end-cp_len+1:end, :); x_time];     
        tx_serial = x_cp(:);
        
        %% ---------- CHANNEL (2 independent AWGN branches) ----------
        % Eb/N0 -> SNR per time-domain sample
        % MUST account for Code Rate (4/7)
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        
        Ps     = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;
        
        noise1 = sqrt(sigma2/2) * (randn(size(tx_serial)) + 1i*randn(size(tx_serial)));
        noise2 = sqrt(sigma2/2) * (randn(size(tx_serial)) + 1i*randn(size(tx_serial)));
        
        rx1_serial = tx_serial + noise1;
        rx2_serial = tx_serial + noise2;
        
        %% ---------- RECEIVER ----------
        % 1. Reshape and Remove CP per branch
        rx1_par = reshape(rx1_serial, N+cp_len, numSymRun);
        rx2_par = reshape(rx2_serial, N+cp_len, numSymRun);
        
        rx1_no_cp = rx1_par(cp_len+1:end, :);
        rx2_no_cp = rx2_par(cp_len+1:end, :);
        
        % 2. Combining Strategy (Equal Gain for AWGN)
        rx_combined_time = rx1_no_cp + rx2_no_cp;
        
        % 3. FFT (Demodulation)
        Yf = fft(rx_combined_time, N, 1);
        
        % 4. Data Extraction
        % Use colsFilled to ensure we only grab valid data
        Ydata = Yf(:, 2:1+colsFilled);
        
        % Important: Since we summed 2 signals (Signal + Signal = 2*Signal), 
        % we must scale down by 2 before demapping to match unit energy grid.
        Ydata = Ydata / 2;
        
        rxSyms = Ydata(:);
        
        % 5. Demap Symbols to Coded Bits
        rxCodedBits = qam_gray_demod(rxSyms, M);
        
        % 6. ADD CHANNEL DECODING (External Function)
        rxDecodedBits = decode74(rxCodedBits); 
        
        %% ---------- BIT ERROR RATE (BER) ----------
        nCompare = min(length(txDataBits), length(rxDecodedBits));
        err = sum(txDataBits(1:nCompare) ~= rxDecodedBits(1:nCompare));
        
        totalErr  = totalErr + err;
        totalBits = totalBits + nCompare;
    end
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end

%% Plot
figure;
semilogy(EbNo_dB, BER, 'g-^', 'LineWidth', 1.5);
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
title('QPSK-OFDM SIMO 1x2 over AWGN (Equal-Gain Combining)');
legend('SIMO 1x2 AWGN', 'Location', 'southwest');
ylim([1e-5 1]);

%% -------------------- Helper Functions --------------------
function syms = qam_gray_mod(bits, M)
    % Square QAM Gray mapping, normalized to unit average energy.
    k = log2(M);
    if mod(numel(bits), k) ~= 0
        error("Bit length must be multiple of log2(M).");
    end
    m = round(sqrt(M));
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
    syms = (aI + 1i*aQ);
    % Normalize to unit avg energy
    syms = syms / sqrt(mean(abs(syms).^2));
end

function bits = qam_gray_demod(syms, M)
    % Hard decision Gray demapper for square QAM
    k = log2(M);
    m = round(sqrt(M));
    bps = k/2;
    % Undo theoretical normalization for slicing
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