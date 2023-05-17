%function [AO1,AO2,accBeats] = findAOs(acc, pks, M, overlap, sfootPPG, pad, peaknum)
function [AO1,AO2] = findAOs(acc, sfootPPG, pad, peaknum)

    %[accBeats, ~] = ensembleAvg3(separateBeat(acc, pks, 0, 500), M, overlap);
    
    %sigBeats = accBeats;
    sigBeats = acc;
    numFeatures = 4;
%     origIdx = zeros(size(sfootPPG,1), 1);

    origIdx = cell(numFeatures, 1);
    indices = cell(numFeatures, 1);   
    %startIdx = pks;
    AO2 = zeros(size(sfootPPG,1),1);

    % ---------------------------------------------------------------------
    % Extract Features with Peak-Counting
    % ---------------------------------------------------------------------
    for feature = 2
        
        % Find desired feature type along each signal segment
        for segment = 1:size(sigBeats, 2)

            tmp = sigBeats(pad:end,segment);
            foot = sfootPPG(segment);

            % Find all features of the given type
            [peaks{segment}, valleys{segment}, inflections{segment}] = ...
                getPeaks(tmp);

            % Replace empty vectors with NaN
            if isempty(peaks{segment}); peaks{segment} = NaN; end
            if isempty(valleys{segment}); valleys{segment} = NaN; end

            % Return the first feature within the search range
            if feature==1; indices{feature}(segment) = valleys{segment}(1)+pad-1; 
            elseif feature==3; indices{feature}(segment) = valleys{segment}(2)+pad-1;
            elseif feature==2; indices{feature}(segment) = peaks{segment}(1)+pad-1;
            elseif feature==4; indices{feature}(segment) = peaks{segment}(2)+pad-1;
            end
            
            % Find the point nearest to sfootPPG
            tmpdis = peaks{segment} - sfootPPG(segment);
            tmp = peaks{segment};
            
            % Only consider points later than sfootPPG?
            %tmpdis = tmpdis(tmpdis>0);
            %tmp = tmp(tmp>0);
            
            [~,tmpidx] = min(abs(tmpdis));            
            tmpAO = tmp(tmpidx);
            
            AO2(segment) = tmpAO+pad-1;
                                   
            % Save indices
            origIdx{feature}(segment) = indices{feature}(segment);

        end
         
    end
    
    AO1 = indices{2*peaknum}';

end

