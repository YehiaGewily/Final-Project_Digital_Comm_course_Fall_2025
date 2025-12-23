%% OFDM Project - Phase 4: Pilot-Based Channel Estimation & ZF Equalization (SISO)
% 1x1 SISO, QPSK, fixed multipath fading (L=50) + AWGN
% Pilot scheme: 1st OFDM symbol is all pilots (known), rest are data
% Channel estimation: Hest = Ypilot ./ Xpilot
% Equalization: Yeq = Ydata ./ Hest
%
% Expected: BER improves significantly vs. "no EQ" fading case.

clear; clc; close all;
rng(1);

%% Parameters (project)
N          = 1024;       % FFT size
L          = 50;         % channel taps
cp_len     = 72;         % CP length (>= L-1 recommended)
EbNo_dB    = 0:3:40;     % Eb/N0 sweep
numRuns    = 1000;       % runs per Eb/N0
numSymRun  = 100;        % total OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

% QPSK
M  = 4;
k0 = log2(M);

% Pilot (known)
pilotSym = (1+1i)/sqrt(2);  % unit-energy

% Fixed deterministic channel realization (constant for entire simulation)
h = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);

BER = zeros(size(EbNo_dB));

%% Main simulation
for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);

    totalErr  = 0;
    totalBits = 0;

    for run = 1:numRuns

%% ---------- TRANSMITTER ----------
% 1. Build frequency-domain grid: N subcarriers x numSymRun symbols
X = zeros(N, numSymRun);
X(:,1) = pilotSym;  % Pilots in 1st OFDM symbol for channel estimation [cite: 104-106]

% 2. Calculate bit requirements for (7,4) coding
% numDataSym symbols are available for data (Symbols 2 to end) [cite: 106-107]
nCodedBitsNeeded = N * k0 * numDataSym; 

% Since (7,4) code turns 4 raw bits into 7 coded bits, generate 4/7ths of the capacity
numRawBits = floor(nCodedBitsNeeded * (4/7));
txDataBits = randi([0 1], numRawBits, 1); % These are your original "fun" info bits

% 3. ADD CHANNEL CODING [cite: 82-84, 97-101]
% This turns your raw information bits into "Coded Bits" with redundancy
txCodedBits = encode74(txDataBits); 

% 4. Map the CODED bits to symbols (QPSK, 16-QAM, or 64-QAM) [cite: 102, 157-160]
dataSyms = qam_gray_mod(txCodedBits, M);               

% 5. Pack symbols into the OFDM grid (Starting from column 2)
X(:,2:end) = reshape(dataSyms, N, numDataSym);

% 6. OFDM modulation (IFFT) [cite: 93, 119]
% Converts the frequency subcarriers into a time-domain signal
x_time = ifft(X, N, 1);                           

% 7. Add Cyclic Prefix (CP) and serialize [cite: 95, 120-121]
% Copies the last part of the symbol to the front to prevent interference (ISI)
x_cp = [x_time(end-cp_len+1:end, :); x_time];     
tx_serial = x_cp(:);


        %% ---------- CHANNEL ----------
        % Eb/N0 -> SNR per time-domain sample (accounts for CP overhead)
        SNRlin = EbNoLin * k0 * (N / (N + cp_len));

        Ps = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;

        % Apply multipath per OFDM symbol block (no cross-symbol mixing)
        tx_blocks = reshape(tx_serial, N+cp_len, numSymRun);
        rx_blocks = zeros(size(tx_blocks));
        for s = 1:numSymRun
            rx_blocks(:,s) = conv(tx_blocks(:,s), h, 'same');
        end
        rx_faded = rx_blocks(:);

        % Add AWGN
        noise = sqrt(sigma2/2) * (randn(size(rx_faded)) + 1i*randn(size(rx_faded)));
        rx_serial = rx_faded + noise;

%% ---------- RECEIVER ----------
        % 1. Reshape, Remove CP, and Perform FFT [cite: 122-123]
        rx_parallel = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp    = rx_parallel(cp_len+1:end, :);       % N x numSymRun
        
        % Convert back to frequency domain
        Yf = fft(rx_no_cp, N, 1);                          % N x numSymRun

        % 2. Channel Estimation from Pilot Symbol [cite: 104, 107, 149]
        % We use the 1st symbol (column 1) to determine how the channel affected the signal
        Hest = Yf(:,1) ./ X(:,1);
        Hest(abs(Hest) < 1e-12) = 1e-12;                   % avoid divide-by-zero

        % 3. Equalization (Zero-Forcing) [cite: 122, 128]
        % Apply the inverse of the channel to the DATA symbols (columns 2 to end)
        Yeq = Yf(:,2:end) ./ Hest;                         % N x numDataSym
        rxSyms = Yeq(:);

        % 4. Demap Symbols back to Coded Bits [cite: 102]
        % These bits still contain the (7,4) redundancy
        rxCodedBits = qam_gray_demod(rxSyms, M);

        % 5. CHANNEL DECODING [cite: 82, 101]
        % Use the (7,4) code to fix errors and recover the original 4-bit blocks
        rxDecodedBits = decode74(rxCodedBits); 

        %% ---------- BIT ERROR RATE (BER) ----------
        % IMPORTANT: Compare the DECODED bits to the ORIGINAL raw bits (txDataBits)
        % Ensure the lengths match for the comparison [cite: 23, 155]
        nCompare = min(length(txDataBits), length(rxDecodedBits));
        
        % Final error count of the actual information
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
    % Hard decision Gray demapper for square QAM, assumes mod used theoretical normalization.
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
