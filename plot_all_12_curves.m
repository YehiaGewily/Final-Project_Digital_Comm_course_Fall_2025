%% MASTER SCRIPT: 12 BER Curves (OFDM: AWGN vs Fixed Multipath, SISO vs SIMO, QPSK/16QAM/64QAM)
% Generates BER curves for:
%   Modulation:   M = 4, 16, 64
%   Antennas:     SISO 1x1, SIMO 1x2 (MRC in fading)
%   Channel:      AWGN, Fixed multipath fading (L=50)
%
% Pilots: 1st OFDM symbol is all pilots on all subcarriers.
% Equalization:
%   - AWGN: no pilot estimation needed (H = 1)
%   - Fading: pilot-based channel estimation + (ZF for SISO) / (MRC for SIMO)

clear; clc; close all;
rng(1);

%% Parameters (Project Table)
N         = 1024;         % FFT size
L         = 50;           % channel taps
cp_len    = 72;           % CP length
EbNo_dB   = 0:3:40;       % Eb/N0 sweep
numRuns   = 1000;         % runs per Eb/N0
numSymRun = 100;          % OFDM symbols per run
numPilotSym = 1;
numDataSym  = numSymRun - numPilotSym;

mods = [4 16 64];                 % QPSK, 16QAM, 64QAM
mimoNames = ["SISO 1x1","SIMO 1x2"];
chanNames = ["AWGN","FADING_FIXED"];

pilotSym = (1+1i)/sqrt(2);        % known pilot (unit energy)

% Pre-allocate BER results: [mod x mimo x chan x EbNo]
BER = zeros(numel(mods), 2, 2, numel(EbNo_dB));

%% Fixed deterministic channel realizations (constant for whole simulation)
h_siso  = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);
h_simo1 = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);
h_simo2 = (randn(1,L) + 1i*randn(1,L)) / sqrt(2*L);

%% Main loops
for mi = 1:numel(mods)
    M  = mods(mi);
    k0 = log2(M);

    for mimoIdx = 1:2
        isSIMO = (mimoIdx == 2);

        for ci = 1:2
            isFading = (ci == 2);

            fprintf('Running: M=%d, %s, %s\n', M, mimoNames(mimoIdx), chanNames(ci));

            for ei = 1:numel(EbNo_dB)
                EbNoLin = 10^(EbNo_dB(ei)/10);
                totalErr = 0;
                totalBits = 0;

                for run = 1:numRuns
                    %% ---------- TRANSMITTER ----------
                    % Build grid with 1 pilot symbol + data symbols
                    X = zeros(N, numSymRun);
                    X(:,1) = pilotSym;

                    numDataBits = N * k0 * numDataSym;
                    txBits = randi([0 1], numDataBits, 1);

                    dataSyms = qam_gray_mod(txBits, M);
                    X(:,2:end) = reshape(dataSyms, N, numDataSym);

                    x_time = ifft(X, N, 1);
                    x_cp   = [x_time(end-cp_len+1:end, :); x_time];
                    txSerial = x_cp(:);

                    %% ---------- NOISE POWER ----------
                    % Eb/N0 -> SNR per time-domain sample (accounts for CP overhead)
                    SNRlin = EbNoLin * k0 * (N/(N + cp_len));
                    Ps = mean(abs(txSerial).^2);
                    sigma2 = Ps / SNRlin;

                    %% ---------- CHANNEL ----------
                    if ~isFading
                        % AWGN
                        y1 = txSerial + sqrt(sigma2/2)*(randn(size(txSerial)) + 1i*randn(size(txSerial)));
                        if isSIMO
                            y2 = txSerial + sqrt(sigma2/2)*(randn(size(txSerial)) + 1i*randn(size(txSerial)));
                        end
                    else
                        % Fixed multipath fading (apply per OFDM symbol to avoid cross-symbol mixing)
                        tx_blocks = reshape(txSerial, N+cp_len, numSymRun);

                        % Select channels
                        if ~isSIMO
                            h1 = h_siso;
                        else
                            h1 = h_simo1;
                            h2 = h_simo2;
                        end

                        % Branch 1
                        rx1_blocks = zeros(size(tx_blocks));
                        for s = 1:numSymRun
                            rx1_blocks(:,s) = conv(tx_blocks(:,s), h1, 'same');
                        end
                        y1 = rx1_blocks(:) + sqrt(sigma2/2)*(randn(size(txSerial)) + 1i*randn(size(txSerial)));

                        % Branch 2 (SIMO only)
                        if isSIMO
                            rx2_blocks = zeros(size(tx_blocks));
                            for s = 1:numSymRun
                                rx2_blocks(:,s) = conv(tx_blocks(:,s), h2, 'same');
                            end
                            y2 = rx2_blocks(:) + sqrt(sigma2/2)*(randn(size(txSerial)) + 1i*randn(size(txSerial)));
                        end
                    end

                    %% ---------- RECEIVER ----------
                    % Branch 1 FFT
                    rx1_par = reshape(y1, N+cp_len, numSymRun);
                    rx1_no_cp = rx1_par(cp_len+1:end, :);
                    Y1f = fft(rx1_no_cp, N, 1);

                    if ~isSIMO
                        % -------- SISO --------
                        if ~isFading
                            % AWGN: perfect channel
                            H1 = ones(N,1);
                        else
                            % Fading: pilot estimate
                            H1 = Y1f(:,1) ./ X(:,1);
                            H1(abs(H1) < 1e-12) = 1e-12;
                        end

                        % ZF equalize data symbols
                        Yeq = Y1f(:,2:end) ./ H1;

                    else
                        % -------- SIMO 1x2 --------
                        % Branch 2 FFT
                        rx2_par = reshape(y2, N+cp_len, numSymRun);
                        rx2_no_cp = rx2_par(cp_len+1:end, :);
                        Y2f = fft(rx2_no_cp, N, 1);

                        if ~isFading
                            % AWGN: channels are 1
                            H1 = ones(N,1);
                            H2 = ones(N,1);
                        else
                            % Fading: pilot estimates
                            H1 = Y1f(:,1) ./ X(:,1);
                            H2 = Y2f(:,1) ./ X(:,1);
                            H1(abs(H1) < 1e-12) = 1e-12;
                            H2(abs(H2) < 1e-12) = 1e-12;
                        end

                        % MRC on data symbols
                        Y1d = Y1f(:,2:end);
                        Y2d = Y2f(:,2:end);

                        den = abs(H1).^2 + abs(H2).^2;
                        den(den < 1e-12) = 1e-12;

                        Yeq = (conj(H1).*Y1d + conj(H2).*Y2d) ./ den;
                    end

                    % Demap + BER
                    rxBits = qam_gray_demod(Yeq(:), M);
                    totalErr = totalErr + sum(txBits ~= rxBits);
                    totalBits = totalBits + numel(txBits);
                end

                BER(mi, mimoIdx, ci, ei) = totalErr / totalBits;
                fprintf('  Eb/N0=%2d dB -> BER=%.3e\n', EbNo_dB(ei), BER(mi,mimoIdx,ci,ei));
            end
        end
    end
end

%% Plot 12 curves
figure; hold on; grid on; set(gca,'YScale','log');

styles = ["-o","-s","-^","-d","--o","--s","--^","--d",":o",":s",":^",":d"];
names  = strings(1,12);
idx = 1;

for mi = 1:numel(mods)
    for mimoIdx = 1:2
        for ci = 1:2
            y = squeeze(BER(mi, mimoIdx, ci, :));
            plot(EbNo_dB, y, styles(idx), 'LineWidth', 1.5);
            names(idx) = sprintf("M=%d, %s, %s", mods(mi), mimoNames(mimoIdx), chanNames(ci));
            idx = idx + 1;
        end
    end
end

xlabel('E_b/N_0 (dB)');
ylabel('BER');
title('Final 12 Curves: OFDM BER vs Eb/N0 (AWGN vs Fixed Fading, SISO vs SIMO)');
legend(names, 'Location', 'southwest');
ylim([1e-5 1]);

%% -------------------- Helper Functions --------------------
function syms = qam_gray_mod(bits, M)
    % Square QAM Gray mapping (manual), normalized to theoretical unit avg energy.
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

    gI = bi2int_manual(bI);
    gQ = bi2int_manual(bQ);

    iI = gray2bin_int_manual(gI);
    iQ = gray2bin_int_manual(gQ);

    aI = 2*iI - (m-1);
    aQ = 2*iQ - (m-1);

    normFactor = sqrt(2*(m^2-1)/3);
    syms = (aI + 1i*aQ) / normFactor;  % unit avg energy
end

function bits = qam_gray_demod(syms, M)
    % Hard decision Gray demapper (manual).
    k = log2(M);
    m = round(sqrt(M));
    if m*m ~= M
        error("M must be a perfect square (4,16,64...).");
    end

    bps = k/2;

    % Undo normalization back to grid levels
    normFactor = sqrt(2*(m^2-1)/3);
    syms = syms * normFactor;

    I = real(syms);
    Q = imag(syms);

    levels = (-(m-1):2:(m-1));
    idxI = slicer_to_index_manual(I, levels);
    idxQ = slicer_to_index_manual(Q, levels);

    gI = bin2gray_int_manual(idxI);
    gQ = bin2gray_int_manual(idxQ);

    bI = int2bits_manual(gI, bps);
    bQ = int2bits_manual(gQ, bps);

    bitsMat = [bI bQ];
    bits = reshape(bitsMat.', [], 1);
end

function idx = slicer_to_index_manual(x, levels)
    idx = zeros(size(x));
    for n = 1:numel(x)
        [~, ii] = min(abs(x(n) - levels));
        idx(n) = ii - 1; % 0-based
    end
end

function v = bi2int_manual(B)
    b = size(B,2);
    weights = 2.^(b-1:-1:0);
    v = B * weights.';
end

function B = int2bits_manual(v, b)
    v = v(:);
    n = numel(v);
    B = zeros(n,b);
    for i = 1:b
        B(:,i) = bitget(v, b-i+1);
    end
end

function b = gray2bin_int_manual(g)
    g = uint32(g);
    b = g;
    while any(g)
        g = bitshift(g, -1);
        b = bitxor(b, g);
    end
    b = double(b);
end

function g = bin2gray_int_manual(b)
    b = uint32(b);
    g = bitxor(b, bitshift(b, -1));
    g = double(g);
end
