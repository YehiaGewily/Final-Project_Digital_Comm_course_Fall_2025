%% OFDM Project - Phase 5B: SIMO 1x2 Fixed Fading + Pilot Estimation + MRC
% 1 Tx -> 2 Rx antennas (SIMO)
% Channel: fixed multipath fading (L taps) + AWGN
% Pilots: 1st OFDM symbol is all pilots (known)
% Channel estimation per antenna: H = Ypilot ./ Xpilot
% Combining: MRC in frequency domain on data symbols
%
% Expected: Better BER than SISO fading with EQ (diversity gain).

clear; clc; close all;
rng(1);

%% Parameters (project)
N          = 1024;      % FFT size
L          = 50;        % channel taps
cp_len     = 72;        % CP length (>= L-1 recommended)
EbNo_dB    = 0:3:40;    % Eb/N0 sweep
numRuns    = 1000;      % runs per Eb/N0
numSymRun  = 100;       % total OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

% QPSK
M  = 4;
k0 = log2(M);

% Pilot (known)
pilotSym = (1+1i)/sqrt(2);  % unit-energy

% Two independent fixed fading channels (deterministic realizations)
h1 = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);
h2 = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);

BER = zeros(size(EbNo_dB));

%% Main simulation
for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);

    totalErr  = 0;
    totalBits = 0;

    for run = 1:numRuns

        %% ---------- TRANSMITTER ----------
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym;  % pilot OFDM symbol

        % Data bits only (symbols 2..end)
        numDataBits = N * k0 * numDataSym;
        txBits = randi([0 1], numDataBits, 1);

        dataSyms = qam_gray_mod(txBits, M);
        X(:,2:end) = reshape(dataSyms, N, numDataSym);

        % OFDM modulation + CP
        x_time = ifft(X, N, 1);
        x_cp   = [x_time(end-cp_len+1:end, :); x_time];   % (N+cp) x numSymRun
        tx_serial = x_cp(:);

        %% ---------- CHANNEL ----------
        % Eb/N0 -> SNR per time-domain sample (includes CP overhead)
        SNRlin = EbNoLin * k0 * (N / (N + cp_len));

        Ps     = mean(abs(tx_serial).^2);
        sigma2 = Ps / SNRlin;

        % Apply fading per OFDM symbol block (no cross-symbol mixing)
        tx_blocks = reshape(tx_serial, N+cp_len, numSymRun);

        rx1_blocks = zeros(size(tx_blocks));
        rx2_blocks = zeros(size(tx_blocks));
        for s = 1:numSymRun
            rx1_blocks(:,s) = conv(tx_blocks(:,s), h1, 'same');
            rx2_blocks(:,s) = conv(tx_blocks(:,s), h2, 'same');
        end

        rx1_faded = rx1_blocks(:);
        rx2_faded = rx2_blocks(:);

        % Add independent AWGN per branch
        noise1 = sqrt(sigma2/2) * (randn(size(rx1_faded)) + 1i*randn(size(rx1_faded)));
        noise2 = sqrt(sigma2/2) * (randn(size(rx2_faded)) + 1i*randn(size(rx2_faded)));

        rx1_serial = rx1_faded + noise1;
        rx2_serial = rx2_faded + noise2;

        %% ---------- RECEIVER ----------
        % CP removal
        rx1_par = reshape(rx1_serial, N+cp_len, numSymRun);
        rx2_par = reshape(rx2_serial, N+cp_len, numSymRun);

        rx1_no_cp = rx1_par(cp_len+1:end, :);
        rx2_no_cp = rx2_par(cp_len+1:end, :);

        % FFT per branch
        Y1f = fft(rx1_no_cp, N, 1);  % N x numSymRun
        Y2f = fft(rx2_no_cp, N, 1);

        % Channel estimation from pilot symbol
        H1 = Y1f(:,1) ./ X(:,1);
        H2 = Y2f(:,1) ./ X(:,1);

        % Avoid divide-by-zero / numerical issues
        H1(abs(H1) < 1e-12) = 1e-12;
        H2(abs(H2) < 1e-12) = 1e-12;

        % Data symbols
        Y1d = Y1f(:,2:end);   % N x numDataSym
        Y2d = Y2f(:,2:end);

        % MRC combining in frequency domain:
        % Y_mrc = (conj(H1).*Y1 + conj(H2).*Y2) / (|H1|^2 + |H2|^2)
        den = (abs(H1).^2 + abs(H2).^2);
        den(den < 1e-12) = 1e-12;

        Yeq = (conj(H1).*Y1d + conj(H2).*Y2d) ./ den;

        % Serialize equalized data symbols and demap
        rxSyms = Yeq(:);
        rxBits = qam_gray_demod(rxSyms, M);

        % BER
        err = sum(txBits ~= rxBits);
        totalErr  = totalErr + err;
        totalBits = totalBits + length(txBits);
    end

    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB, BER = %.3e\n', EbNo_dB(e), BER(e));
end

%% Plot
figure;
semilogy(EbNo_dB, BER, 'm-d', 'LineWidth', 1.5);
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('BER');
title('QPSK-OFDM SIMO 1x2: Fixed Multipath + AWGN (Pilot Hest + MRC)');
legend('1x2 SIMO MRC', 'Location', 'southwest');
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
