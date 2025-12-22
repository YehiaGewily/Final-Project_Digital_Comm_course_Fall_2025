%% OFDM Project - Phase 3: Fixed Multipath Fading (No Equalization)
% QPSK-OFDM SISO (1x1) over:
% tx -> CP -> per-OFDM-symbol multipath (L taps) -> AWGN -> receiver
% Expected: BER becomes significantly worse (no channel estimation/equalization yet).

clear; clc; close all;
rng(1);

%% Parameters (from project)
N         = 1024;        % FFT size
L         = 50;          % multipath taps
cp_len    = 72;          % CP length (>= L-1 recommended)
EbNo_dB   = 0:3:40;      % Eb/N0 sweep
numRuns   = 1000;        % runs per Eb/N0
numSymRun = 100;         % OFDM symbols per run

% QPSK
M  = 4;
k0 = log2(M);

% Fixed deterministic fading channel realization (constant for whole simulation)
h = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);

BER = zeros(size(EbNo_dB));

%% Main simulation
for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);

    totalErr  = 0;
    totalBits = 0;

    for run = 1:numRuns

        %% ---------- TRANSMITTER ----------
        numBits = N * k0 * numSymRun;
        txBits  = randi([0 1], numBits, 1);

        % QPSK mapping (Gray, unit avg energy)
        txSyms = qam_gray_mod(txBits, M);            % length = N*numSymRun

        % Pack into OFDM grid (N subcarriers x numSymRun symbols)
        X = reshape(txSyms, N, numSymRun);

        % IFFT (OFDM modulation)
        x_time = ifft(X, N, 1);                      % N x numSymRun

        % Add CP
        x_cp = [x_time(end-cp_len+1:end, :); x_time]; % (N+cp) x numSymRun

        % Serialize
        tx_serial = x_cp(:);

        %% ---------- CHANNEL ----------
        % Eb/N0 -> SNR per time-domain sample (accounts for CP overhead)
        SNRlin = EbNoLin * k0 * (N / (N + cp_len));

        % Signal power and noise variance
        Ps     = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;

        % IMPORTANT: apply multipath per OFDM symbol block (prevents symbol-to-symbol mixing)
        tx_blocks = reshape(tx_serial, N+cp_len, numSymRun);

        rx_blocks = zeros(size(tx_blocks));
        for s = 1:numSymRun
            % FIR channel (same length out)
            rx_blocks(:,s) = filter(h, 1, tx_blocks(:,s));
        end

        rx_faded = rx_blocks(:);

        % Add AWGN
        noise = sqrt(sigma2/2) * (randn(size(rx_faded)) + 1i*randn(size(rx_faded)));
        rx_serial = rx_faded + noise;

        %% ---------- RECEIVER (no equalization) ----------
        rx_parallel = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp    = rx_parallel(cp_len+1:end, :);  % N x numSymRun

        % FFT (OFDM demodulation)
        Yf     = fft(rx_no_cp, N, 1);
        rxSyms = Yf(:);

        % Demap QPSK -> bits
        rxBits = qam_gray_demod(rxSyms, M);

        % BER count
        err = sum(txBits ~= rxBits);
        totalErr  = totalErr + err;
        totalBits = totalBits + length(txBits);
    end

    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end

%% Plot
figure;
semilogy(EbNo_dB, BER, 'r-s', 'LineWidth', 1.5);
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
title('QPSK-OFDM over Fixed Multipath Fading + AWGN (No Equalization)');
legend('SISO 1x1, L=50, No EQ', 'Location', 'southwest');
ylim([1e-5 1]);

%% -------------------- Helper Functions --------------------
function syms = qam_gray_mod(bits, M)
    % Square QAM Gray mapping, normalized to unit average energy.
    % For QPSK (M=4), this produces standard Gray QPSK on {±1±j}/sqrt(2).
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
    B = reshape(bits, k, []).';  % [nSyms x k]

    bI = B(:,1:bps);
    bQ = B(:,bps+1:end);

    gI = bi2int(bI);
    gQ = bi2int(bQ);

    iI = gray2bin_int(gI);
    iQ = gray2bin_int(gQ);

    aI = 2*iI - (m-1);
    aQ = 2*iQ - (m-1);

    syms = aI + 1i*aQ;

    % Normalize to unit average energy
    syms = syms / sqrt(mean(abs(syms).^2));
end

function bits = qam_gray_demod(syms, M)
    % Hard decision Gray demapper for square QAM, assumes unit-average-energy symbols.
    k = log2(M);
    m = round(sqrt(M));
    if m*m ~= M
        error("M must be a perfect square (4,16,64...).");
    end
    bps = k/2;

    % Undo the standard square-QAM normalization (ideal average Es = 2*(m^2-1)/3)
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

    bitsMat = [bI bQ];           % [nSyms x k]
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
