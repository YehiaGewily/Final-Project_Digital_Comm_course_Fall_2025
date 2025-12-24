clear; clc; close all;
rng(1);

N           = 1024;
cp_len      = 72;
EbNo_dB     = 0:3:40;
numRuns     = 1000;
numSymRun   = 100;

M  = 16;
k0 = log2(M);
pilotSym = (1+1i)/sqrt(2) * ones(N, 1);
BER = zeros(size(EbNo_dB));

for e = 1:length(EbNo_dB)
    EbNoLin = 10^(EbNo_dB(e)/10);
    totalErr  = 0;
    totalBits = 0;
    
    codeRate = 4/7;
    
    for run = 1:numRuns
        X = zeros(N, numSymRun);
        X(:,1) = pilotSym;
        
        numDataSym = numSymRun - 1;
        
        nCodedBitsAvailable = N * k0 * numDataSym;
        nCodedBitsNeeded = floor(nCodedBitsAvailable/(7*4*N)) * (7*4*N);
        nRawBits = nCodedBitsNeeded * codeRate;
        
        txDataBits = randi([0 1], nRawBits, 1);
        
        txCodedBits = encode74(txDataBits);
        
        dataSyms = qam_gray_mod(txCodedBits, M);
        
        dataGrid = reshape(dataSyms, N, []);
        colsFilled = size(dataGrid, 2);
        X(:, 2:1+colsFilled) = dataGrid;
        
        x_time = ifft(X, N, 1);
        x_cp   = [x_time(end-cp_len+1:end, :); x_time];
        tx_serial = x_cp(:);
        
        tx_power = mean(abs(tx_serial).^2);
        tx_serial_norm = tx_serial / sqrt(tx_power);
        
        SNRlin = EbNoLin * codeRate * k0 * (N / (N + cp_len));
        noise_variance = 1 / SNRlin;
        
        noise = sqrt(noise_variance/2) * (randn(size(tx_serial_norm)) + 1i*randn(size(tx_serial_norm)));
        rx_serial = tx_serial_norm + noise;
        
        rx_mat = reshape(rx_serial, N+cp_len, numSymRun);
        rx_no_cp = rx_mat(cp_len+1:end, :);
        Yf = fft(rx_no_cp, N, 1);
        
        Ydata = Yf(:, 2:1+colsFilled);
        Ydata = Ydata ./ sqrt(N);
        
        rxCodedBits = qam_gray_demod(Ydata(:), M);
        
        rxDecodedBits = decode74(rxCodedBits);
        
        len = min(length(txDataBits), length(rxDecodedBits));
        err = sum(txDataBits(1:len) ~= rxDecodedBits(1:len));
        
        totalErr  = totalErr + err;
        totalBits = totalBits + len;
    end
    
    BER(e) = totalErr / totalBits;
    fprintf('Eb/N0 = %2d dB | BER = %.5e\n', EbNo_dB(e), BER(e));
end

figure;
semilogy(EbNo_dB, BER, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
grid on;
title('SISO AWGN (16-QAM) with Hamming (7,4) Coding');
xlabel('E_b/N_0 (dB)');
ylabel('Bit Error Rate (BER)');
legend('SISO AWGN 16-QAM Coded');
ylim([1e-5 1]);

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