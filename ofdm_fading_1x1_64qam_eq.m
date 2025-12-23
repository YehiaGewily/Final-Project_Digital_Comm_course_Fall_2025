%% OFDM Project - Phase 4: Pilot-Based Channel Estimation & ZF Equalization (SISO 16-QAM)
% 1x1 SISO, 16-QAM, fixed multipath fading (L=50) + AWGN
clear; clc; close all;
rng(1);

%% Parameters (project)
N          = 1024;       % FFT size
L          = 50;         % channel taps
cp_len     = 72;         % CP length
EbNo_dB    = 0:3:40;     % Eb/N0 sweep
numRuns    = 1000;       % runs per Eb/N0
numSymRun  = 100;        % total OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

% 64-QAM
M  = 64;
k0 = log2(M);

% Pilot (known)
pilotSym = (1+1i)/sqrt(2);  % unit-energy

% Fixed deterministic channel realization
h = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);
BER = zeros(size(EbNo_dB));

%% Main simulation (Integrated with (7,4) Channel Coding)
for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr = 0; totalBits = 0;
    
    for run = 1:numRuns
        %% ---------- TRANSMITTER ----------
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym;  % Insert pilot symbol in the first column [cite: 104-106]
        
        % 1. Calculate RAW bit requirements (4/7 code rate)
        nCodedBitsNeeded = N * k0 * numDataSym;
        numRawBits = floor(nCodedBitsNeeded * (4/7));
        txDataBits = randi([0 1], numRawBits, 1); % Your original information

        % 2. ENCODE (Channel Coding) [cite: 82, 99]
        txCodedBits = encode74(txDataBits); 

        % 3. Map bits to symbols and pack the grid [cite: 102]
        dataSyms = qam_gray_mod(txCodedBits, M);               
        X(:,2:end) = reshape(dataSyms, N, numDataSym);
        
        % 4. OFDM Modulation (IFFT) and Cyclic Prefix [cite: 93, 95, 119-121]
        x_time = ifft(X, N, 1);
        x_cp = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);

        %% ---------- CHANNEL SIMULATION ----------
        % Calculate Noise and Apply Fading [cite: 133-137]
        SNRlin = EbNoLin * k0 * (N / (N + cp_len));
        Ps = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;
        
        tx_blocks = reshape(tx_serial, N+cp_len, numSymRun);
        rx_blocks = zeros(size(tx_blocks));
        for s = 1:numSymRun
            rx_blocks(:,s) = conv(tx_blocks(:,s), h, 'same');
        end
        
        rx_faded = rx_blocks(:);
        noise = sqrt(sigma2/2) * (randn(size(rx_faded)) + 1i*randn(size(rx_faded)));
        rx_serial = rx_faded + noise;

        %% ---------- RECEIVER ----------
        % 1. CP Removal and FFT [cite: 122-123]
        rx_parallel = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp = rx_parallel(cp_len+1:end, :);       
        Yf = fft(rx_no_cp, N, 1);                          
        
        % 2. Channel Estimation and Equalization [cite: 104, 107]
        Hest = Yf(:,1) ./ X(:,1);
        Hest(abs(Hest) < 1e-12) = 1e-12;                   
        Yeq = Yf(:,2:end) ./ Hest;                         
        
        % 3. Demapping (Symbols -> Coded Bits)
        rxSyms = Yeq(:);
        rxCodedBits = qam_gray_demod(rxSyms, M);

        % 4. DECODE (Channel Decoding) [cite: 101]
        rxDecodedBits = decode74(rxCodedBits); 

        %% ---------- BIT ERROR RATE (BER) ----------
        % Compare decoded data back to original source bits [cite: 23-24]
        nCompare = min(length(txDataBits), length(rxDecodedBits));
        err = sum(txDataBits(1:nCompare) ~= rxDecodedBits(1:nCompare));
        
        totalErr = totalErr + err; 
        totalBits = totalBits + nCompare;
    end
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end

%% Plot
figure; semilogy(EbNo_dB, BER, 'r-o', 'LineWidth', 1.5); grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('16-QAM OFDM SISO: Fixed Multipath + AWGN (Pilot Hest + ZF EQ)');
legend('1x1 fading + 16-QAM EQ');
% (Helper functions identical to QPSK EQ base)

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
