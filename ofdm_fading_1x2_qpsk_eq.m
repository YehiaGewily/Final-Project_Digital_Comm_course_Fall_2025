%% OFDM Project - Phase 5B: Final Corrected Code (SIMO 1x2 + Hamming + MRC)
% 1 Tx -> 2 Rx antennas (SIMO)
% Coding: Hamming (7,4) (Included as local functions)
% Channel: Fixed multipath fading + AWGN
% Equalization: MRC
clear; clc; close all;

%% 1. Parameters
rng(1); % Set global seed
N           = 1024;      % FFT size
L           = 50;        % Channel taps
cp_len      = 72;        % CP length
EbNo_dB     = 0:2:30;    % Eb/N0 sweep range
numRuns     = 200;       % Iterations per SNR
numSymRun   = 50;        % OFDM symbols per frame
numPilotSym = 1;         % First symbol is pilot
numDataSym  = numSymRun - numPilotSym;

% Modulation (QPSK)
M  = 4;
k0 = log2(M); 

% Coding Rate (Hamming 7,4)
codeRate = 4/7;

% --- FIX: Better Pilot Generation ---
% Generate a random QPSK pilot sequence (fixed seed for consistency)
prev_rng = rng;
rng(42); 
pilotSym = (randi([0 1], N, 1)*2 - 1) + 1i*(randi([0 1], N, 1)*2 - 1);
pilotSym = pilotSym / sqrt(2);
rng(prev_rng);

% Channel Setup: Fixed Fading
h1 = (randn(1,L) + 1i*randn(1,L)); h1 = h1 / norm(h1); 
h2 = (randn(1,L) + 1i*randn(1,L)); h2 = h2 / norm(h2);

BER = zeros(size(EbNo_dB));

%% 2. Main Simulation Loop
fprintf('Starting Final Simulation...\n');

for e = 1:length(EbNo_dB)
    totalErr  = 0;
    totalBits = 0;
    
    EbNoLin = 10^(EbNo_dB(e)/10);
    
    for run = 1:numRuns
        %% --- TRANSMITTER ---
        
        % 1. Data Bit Generation (The "LCM" Fix)
        % We need the total coded bits to be divisible by:
        %  - 7 (for Hamming Encoder output)
        %  - 2 (for QPSK Modulator input)
        % LCM(7, 2) = 14.
        
        totalSubcarriers = N * numDataSym;
        maxCodedBits = totalSubcarriers * k0;
        
        % Round down to nearest multiple of 14
        nCodedBits = floor(maxCodedBits / 14) * 14;
        
        % Calculate raw data bits needed to produce those coded bits
        nDataBits = nCodedBits * codeRate;
        
        txDataBits = randi([0 1], nDataBits, 1);
        
        % 2. Channel Coding (Hamming 7,4)
        txCodedBits = encode74(txDataBits);
        
        % 3. Symbol Mapping
        dataSyms = qam_gray_mod(txCodedBits, M);
        
        % 4. Frame Assembly
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym; % Symbol 1 is Pilot
        
        % Fill remainder with data
        % Note: There might be a few unused subcarriers at the very end
        % because of the rounding to 14. We fill them with zeros (padding).
        dataGridVector = [dataSyms; zeros(maxCodedBits/k0 - length(dataSyms), 1)];
        X(:, 2:end) = reshape(dataGridVector, N, numDataSym);
        
        % 5. IFFT & CP Addition
        x_time = ifft(X, N, 1);
        x_cp   = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        %% --- CHANNEL (SIMO 1x2) ---
        
        % Calculate Noise Power
        SNRlin = EbNoLin * k0 * codeRate * (N / (N + cp_len));
        sigPower = mean(abs(tx_serial).^2);
        noisePower = sigPower / SNRlin;
        
        % Apply Fading
        rx1_clean = filter(h1, 1, tx_serial);
        rx2_clean = filter(h2, 1, tx_serial);
        
        % Add AWGN
        scale = sqrt(noisePower/2);
        rx1 = rx1_clean + scale * (randn(size(rx1_clean)) + 1i*randn(size(rx1_clean)));
        rx2 = rx2_clean + scale * (randn(size(rx2_clean)) + 1i*randn(size(rx2_clean)));
        
        %% --- RECEIVER ---
        
        % 1. CP Removal & FFT
        rx1_mat = reshape(rx1, N+cp_len, numSymRun);
        rx2_mat = reshape(rx2, N+cp_len, numSymRun);
        
        Y1 = fft(rx1_mat(cp_len+1:end, :), N, 1);
        Y2 = fft(rx2_mat(cp_len+1:end, :), N, 1);
        
        % 2. Channel Estimation
        H1_est = Y1(:,1) ./ X(:,1);
        H2_est = Y2(:,1) ./ X(:,1);
        
        H1_grid = repmat(H1_est, 1, numDataSym);
        H2_grid = repmat(H2_est, 1, numDataSym);
        
        % 3. MRC Combining
        Y1_data = Y1(:, 2:end);
        Y2_data = Y2(:, 2:end);
        
        numer = conj(H1_grid).*Y1_data + conj(H2_grid).*Y2_data;
        denom = abs(H1_grid).^2 + abs(H2_grid).^2 + 1e-10;
        
        Y_eq = numer ./ denom;
        
        % 4. Demapping
        % Extract valid symbols (ignore the zero-padding at the end)
        numValidSyms = length(dataSyms);
        Y_vec = Y_eq(:);
        Y_valid = Y_vec(1:numValidSyms);
        
        rxCodedBits = qam_gray_demod(Y_valid, M);
        
        % 5. Channel Decoding
        rxDecodedBits = decode74(rxCodedBits);
        
        % 6. BER Calc
        bitErrors = sum(txDataBits ~= rxDecodedBits);
        totalErr  = totalErr + bitErrors;
        totalBits = totalBits + length(txDataBits);
    end
    
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB | BER = %.5e\n', EbNo_dB(e), BER(e));
end

%% 3. Visualization
save('results_simo_final.mat', 'EbNo_dB', 'BER');
figure;
semilogy(EbNo_dB, BER, '-bo', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
title('OFDM SIMO 1x2 with Fixed Fading & Hamming(7,4)');
xlabel('Eb/N0 (dB)');
ylabel('BER');
ylim([1e-6 1]);

%% -------------------- Local Helper Functions --------------------

% --- QPSK MOD/DEMOD ---
function syms = qam_gray_mod(bits, M)
    k = log2(M); 
    x = reshape(bits, k, []).';
    dec = x * [2; 1]; % Binary to integer (for k=2)
    % Gray Map for QPSK: 0->0, 1->1, 3->2, 2->3 (standard QPSK gray)
    % Map: 00-> -1-1i, 01-> -1+1i, 11-> 1+1i, 10-> 1-1i
    map = [(-1-1i) (-1+1i) (1-1i) (1+1i)]/sqrt(2); 
    syms = map(dec+1).';
end

function bits = qam_gray_demod(syms, M)
    % Hard decision demodulation for QPSK
    % Constellation: 1st quadrant (1+1i)->11(3), 2nd (-1+1i)->01(1)
    % 3rd (-1-1i)->00(0), 4th (1-1i)->10(2)
    k = log2(M);
    bits = zeros(length(syms)*k, 1);
    
    I = real(syms); Q = imag(syms);
    
    % Decision thresholds are at 0
    b0 = (I > 0); % MSB
    b1 = (Q > 0); % LSB
    
    % Interleave bits
    bits(1:2:end) = b0;
    bits(2:2:end) = b1;
end

% --- HAMMING (7,4) ENCODER ---
function coded = encode74(msg)
    % Standard G matrix for Hamming (7,4)
    % Structure: [P I] where P is 4x3 parity, I is 4x4 identity
    G = [1 1 0 1 0 0 0;
         0 1 1 0 1 0 0;
         1 1 1 0 0 1 0;
         1 0 1 0 0 0 1];
     
    n = 7; k = 4;
    numBlks = length(msg) / k;
    msgMat = reshape(msg, k, numBlks).'; % numBlks x 4
    
    % Modulo-2 matrix multiplication
    codedMat = mod(msgMat * G, 2);
    
    coded = reshape(codedMat.', [], 1);
end

% --- HAMMING (7,4) DECODER ---
function decoded = decode74(coded)
    % Parity Check Matrix H
    H = [1 0 0 1 0 1 1;
         0 1 0 1 1 1 0;
         0 0 1 0 1 1 1];
    
    n = 7; k = 4;
    numBlks = length(coded) / n;
    rxMat = reshape(coded, n, numBlks).'; % numBlks x 7
    
    % Calculate Syndrome: S = R * H'
    syndrome = mod(rxMat * H.', 2);
    
    % Convert syndrome binary vector to decimal index
    synDec = syndrome * [4; 2; 1]; 
    
    % Error correction logic (Syndrome points to error column in H)
    % Map syndrome value to column index in H to flip
    % H columns: [100, 010, 001, 110, 011, 111, 101] (Decimal: 4, 2, 1, 6, 3, 7, 5)
    % Map: Syn 4->Bit1, Syn 2->Bit2, Syn 1->Bit3 ...
    
    % We process row by row
    decodedMat = zeros(numBlks, k);
    
    % Indices of data bits in the codeword (based on G above): 4,5,6,7
    dataIdx = [4 5 6 7];
    
    for i = 1:numBlks
        s = synDec(i);
        r_row = rxMat(i,:);
        
        if s ~= 0
            % Find which bit corresponds to this syndrome
            % Based on H structure above:
            % S=1(001)->Bit3, S=2(010)->Bit2, S=3(011)->Bit5
            % S=4(100)->Bit1, S=5(101)->Bit7, S=6(110)->Bit4, S=7(111)->Bit6
            flipMap = [3 2 5 1 7 4 6]; 
            errBit = flipMap(s);
            r_row(errBit) = ~r_row(errBit); % Correct error
        end
        decodedMat(i,:) = r_row(dataIdx);
    end
    
    decoded = reshape(decodedMat.', [], 1);
end