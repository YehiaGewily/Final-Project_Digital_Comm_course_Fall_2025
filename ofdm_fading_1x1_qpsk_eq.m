%% OFDM Project - Phase 4: Pilot-Based Channel Estimation & ZF Equalization (SISO)
% 1x1 SISO, QPSK, fixed multipath fading (L=50) + AWGN
% Pilot scheme: 1st OFDM symbol is all pilots (known), rest are data
% Channel estimation: Hest = Ypilot ./ Xpilot
% Equalization: Yeq = Ydata ./ Hest
% Expected: BER improves significantly vs. "no EQ" fading case.

clear; clc; close all;
rng(1);

%% Parameters
N           = 1024;       % FFT size
L           = 50;         % channel taps
cp_len      = 72;         % CP length (>= L-1 recommended)
EbNo_dB     = 0:3:40;     % Eb/N0 sweep
numRuns     = 1000;       % runs per Eb/N0
numSymRun   = 100;        % total OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

% QPSK
M  = 4;
k0 = log2(M);

% Pilot (known)
pilotSym = (1+1i)/sqrt(2);  % unit-energy

% Fixed deterministic channel realization (constant for entire simulation)
% Normalized to unit power
h = (randn(1,L) + 1i*randn(1,L));
h = h / norm(h);

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
        % 1. Build frequency-domain grid: N subcarriers x numSymRun symbols
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym;  % Pilots in 1st OFDM symbol
        
        % 2. Calculate bit requirements for (7,4) coding
        % Calculate total capacity in data symbols
        nCodedBitsAvailable = N * k0 * numDataSym;
        
        % Ensure exact multiple of 7 for the code
        nCodedBitsNeeded = floor(nCodedBitsAvailable/7) * 7; 
        
        % Calculate corresponding raw bits (multiple of 4)
        numRawBits = nCodedBitsNeeded * codeRate;
        
        txDataBits = randi([0 1], numRawBits, 1);
        
        % 3. ADD CHANNEL CODING (External Function)
        txCodedBits = encode74(txDataBits); 
        
        % 4. Map the CODED bits to symbols
        dataSyms = qam_gray_mod(txCodedBits, M);               
        
        % 5. Pack symbols into the OFDM grid
        % Reshape carefully in case floor operation reduced size slightly
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        % 6. OFDM modulation (IFFT)
        x_time = ifft(X, N, 1);                           
        
        % 7. Add Cyclic Prefix (CP) and serialize
        x_cp = [x_time(end-cp_len+1:end, :); x_time];     
        tx_serial = x_cp(:);
        
        %% ---------- CHANNEL ----------
        % Eb/N0 -> SNR per time-domain sample
        % MUST account for Code Rate (4/7) here!
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        
        Ps = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;
        
        % Apply multipath using FILTER (Corrects timing vs. conv)
        rx_serial_clean = filter(h, 1, tx_serial);
        
        % Add AWGN
        noise = sqrt(sigma2/2) * (randn(size(rx_serial_clean)) + 1i*randn(size(rx_serial_clean)));
        rx_serial = rx_serial_clean + noise;
        
        %% ---------- RECEIVER ----------
        % 1. Reshape, Remove CP, and Perform FFT
        rx_parallel = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp    = rx_parallel(cp_len+1:end, :);
        Yf = fft(rx_no_cp, N, 1);
        
        % 2. Channel Estimation from Pilot Symbol
        Hest = Yf(:,1) ./ X(:,1);
        
        % Replicate Hest for all data columns (Static Channel)
        H_grid = repmat(Hest, 1, colsFilled);
        
        % Regularize to avoid divide-by-zero
        H_grid(abs(H_grid) < 1e-12) = 1e-12;
        
        % 3. Equalization (Zero-Forcing)
        Y_data = Yf(:, 2:1+colsFilled);
        Yeq = Y_data ./ H_grid;
        
        % 4. Demap Symbols back to Coded Bits
        rxCodedBits = qam_gray_demod(Yeq(:), M);
        
        % 5. CHANNEL DECODING (External Function)
        rxDecodedBits = decode74(rxCodedBits); 
        
        %% ---------- BIT ERROR RATE (BER) ----------
        % Compare DECODED bits to ORIGINAL raw bits
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
semilogy(EbNo_dB, BER, 'b-o', 'LineWidth', 1.5);
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
title('QPSK-OFDM SISO: Fixed Multipath + AWGN (Pilot Hest + ZF EQ)');
legend('1x1 fading + pilot EQ', 'Location', 'southwest');
ylim([1e-5 1]);

%% -------------------- Helper Functions --------------------
function syms = qam_gray_mod(bits, M)
    k = log2(M);
    if mod(numel(bits), k) ~= 0
        error("Bit length must be multiple of log2(M).");
    end
    m = round(sqrt(M));
    bps = k/2;
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