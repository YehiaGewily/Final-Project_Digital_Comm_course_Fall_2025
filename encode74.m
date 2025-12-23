function txCodedBits = encode74(txDataBits)
    % (7,4) Hamming Generator Matrix
    G = [1 0 0 0 1 1 0; 
         0 1 0 0 0 1 1; 
         0 0 1 0 1 1 1; 
         0 0 0 1 1 0 1];

    % Ensure input is a multiple of 4
    nPad = mod(4 - mod(numel(txDataBits), 4), 4);
    txDataBits = [txDataBits(:); zeros(nPad, 1)];

    % Reshape to Nx4 and multiply mod 2
    bitBlocks = reshape(txDataBits, 4, []).'; 
    codedBlocks = mod(bitBlocks * G, 2); 
    txCodedBits = codedBlocks(:); 
end