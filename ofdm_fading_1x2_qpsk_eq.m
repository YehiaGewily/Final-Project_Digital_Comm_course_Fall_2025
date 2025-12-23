%% OFDM Project - Phase 5B: SIMO 1x2 Fixed Fading + Pilot Estimation + MRC
% 1 Tx -> 2 Rx antennas (SIMO)
% Channel: Fixed multipath fading (L taps) + AWGN
% Coding: Hamming (7,4) (External functions)
% Pilots: 1st OFDM symbol is all pilots
% Combining: MRC in frequency domain

clear; clc; close all;
rng(1); % Set seed for reproducibility

%% 1. Parameters
N           = 1024;      % FFT size
L           = 50;        % Channel taps
cp_len      = 72;        % CP length (must be > L)
EbNo_dB     = 0:2:30;    % Eb/N0 sweep range
numRuns     = 200;       % Iterations per SNR
numSymRun   = 50;        % OFDM symbols per frame
numPilotSym = 1;         % First symbol is pilot
numDataSym  = numSymRun - numPilotSym;

% Modulation (QPSK)
M  = 4;
k0 = log2(M); 

% Pilot Symbol (Known to Rx)
pilotSym = (1+1i)/sqrt(2) * ones(N, 1); % All subcarriers pilot

% Channel Setup: Fixed Fading (Static for the simulation)
% Normalized so average power is 1
h1 = (randn(1,L) + 1i*randn(1,L)); 
h1 = h1 / norm(h1); 
h2 = (randn(1,L) + 1i*randn(1,L)); 
h2 = h2 / norm(h2);

BER = zeros(size(EbNo_dB));

%% 2. Main Simulation Loop
fprintf('Starting Simulation...\n');
for e = 1:length(EbNo_dB)
    totalErr  = 0;
    totalBits = 0;
    
    EbNoLin = 10^(EbNo_dB(e)/10);
    
    % coding rate for (7,4)
    codeRate = 4/7; 
    
    for run = 1:numRuns
        %% --- TRANSMITTER ---
        
        % 1. Data Bit Generation
        % Calculate exact bits needed to fill the data symbols
        nCodedBitsNeeded = N * k0 * numDataSym;
        % We need a multiple of 7 for the coded bits to fit the decoder
        nCodedBitsNeeded = floor(nCodedBitsNeeded/7) * 7;
        
        nRawBits = nCodedBitsNeeded * codeRate; 
        txDataBits = randi([0 1], nRawBits, 1);
        
        % 2. Channel Coding (Hamming 7,4) - EXTERNAL FUNCTION
        txCodedBits = encode74(txDataBits);
        
        % 3. Symbol Mapping
        dataSyms = qam_gray_mod(txCodedBits, M);
        
        % 4. Frame Assembly
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym; % Symbol 1 is Pilot
        
        % Fill remainder with data
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        % 5. IFFT & CP Addition
        x_time = ifft(X, N, 1);
        x_cp   = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        %% --- CHANNEL (SIMO 1x2) ---
        
        % Calculate Noise Power
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        sigPower = mean(abs(tx_serial).^2);
        noisePower = sigPower / SNRlin;
        
        % Apply Multipath Fading (Filter)
        rx1_serial_clean = filter(h1, 1, tx_serial);
        rx2_serial_clean = filter(h2, 1, tx_serial);
        
        % Add AWGN
        noiseScale = sqrt(noisePower/2);
        n1 = noiseScale * (randn(size(rx1_serial_clean)) + 1i*randn(size(rx1_serial_clean)));
        n2 = noiseScale * (randn(size(rx2_serial_clean)) + 1i*randn(size(rx2_serial_clean)));
        
        rx1_serial = rx1_serial_clean + n1;
        rx2_serial = rx2_serial_clean + n2;
        
        %% --- RECEIVER ---
        
        % 1. Parallelization & CP Removal
        rx1_mat = reshape(rx1_serial, N+cp_len, numSymRun);
        rx2_mat = reshape(rx2_serial, N+cp_len, numSymRun);
        
        rx1_no_cp = rx1_mat(cp_len+1:end, :);
        rx2_no_cp = rx2_mat(cp_len+1:end, :);
        
        % 2. FFT
        Y1 = fft(rx1_no_cp, N, 1);
        Y2 = fft(rx2_no_cp, N, 1);
        
        % 3. Channel Estimation (Using Pilot at Symbol 1)
        H1_est = Y1(:,1) ./ X(:,1);
        H2_est = Y2(:,1) ./ X(:,1);
        
        % Replicate estimate for all data symbols
        H1_grid = repmat(H1_est, 1, colsFilled);
        H2_grid = repmat(H2_est, 1, colsFilled);
        
        % 4. MRC Combining
        Y1_data = Y1(:, 2:1+colsFilled);
        Y2_data = Y2(:, 2:1+colsFilled);
        
        numerator = conj(H1_grid).*Y1_data + conj(H2_grid).*Y2_data;
        denominator = abs(H1_grid).^2 + abs(H2_grid).^2;
        denominator(denominator < 1e-10) = 1e-10; % Safety
        
        Y_equalized = numerator ./ denominator;
        
        % 5. Demapping
        rxCodedBits = qam_gray_demod(Y_equalized(:), M);
        
        % 6. Channel Decoding (Hamming 7,4) - EXTERNAL FUNCTION
        rxDecodedBits = decode74(rxCodedBits);
        
        % 7. BER Calculation
        len = min(length(txDataBits), length(rxDecodedBits));
        bitErrors = sum(txDataBits(1:len) ~= rxDecodedBits(1:len));
        
        totalErr  = totalErr + bitErrors;
        totalBits = totalBits + len;
    end
    
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB | BER = %.5e\n', EbNo_dB(e), BER(e));
end

%% 3. Visualization
figure;
semilogy(EbNo_dB, BER, '-bo', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
title('OFDM SIMO 1x2 with Fixed Fading & MRC');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
legend('SIMO 1x2 (Estimated Channel)');
ylim([1e-5 1]);

%% -------------------- Local Helper Functions --------------------
% (Only Modulation helpers remain here. If you moved these to external files
% as well, you can delete them from here.)

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