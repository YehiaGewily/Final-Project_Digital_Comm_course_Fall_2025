function decodedBits = decode74(rxCodedBits)
    % Parity Check Matrix
    H = [1 0 1 1 1 0 0;
         1 1 1 0 0 1 0;
         0 1 1 1 0 0 1];

    % Syndrome-to-Error-Position Look-up Table
    % This tells us which of the 7 bits is wrong based on the syndrome
    syndromes = mod(eye(7) * H.', 2); 
    
    rxBlocks = reshape(rxCodedBits, 7, []).';
    numBlocks = size(rxBlocks, 1);
    correctedBlocks = rxBlocks;

    for i = 1:numBlocks
        s = mod(rxBlocks(i, :) * H.', 2); % Calculate syndrome
        if any(s)
            % Match syndrome to find the error bit position
            [~, pos] = ismember(s, syndromes, 'rows');
            if pos > 0
                correctedBlocks(i, pos) = 1 - correctedBlocks(i, pos); % Flip it back!
            end
        end
    end
    
    % Extract only the first 4 data bits
    dataBlocks = correctedBlocks(:, 1:4);
    decodedBits = dataBlocks(:);
end