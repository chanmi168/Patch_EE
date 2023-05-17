%{
meth 1 = find peaknum
meth 2 = 
%}

function aoPoint = findAO (beats, meth, arg3, sfootPPG)
        
    beats = beats - mean(beats);
    
    aoPoint = zeros(1, size(beats, 2));

    if meth == 1
        peakNum = arg3;
        
        meanBeat = mean(beats, 2);

        [~, pks] = findpeaks(-abs(diff(meanBeat))); %Find max peak first derivative

        allPks = zeros(length(pks), size(beats, 2));

        for i = 1:length(pks)
            for j = 1:size(beats, 2)
                [~, tempPks] = findpeaks(-abs(diff(beats(:, j))));
                [~, locPks] = min(abs(tempPks - pks(i)));
                allPks(i, j) = tempPks(locPks);
            end
        end

        aoPoint = allPks(peakNum, :)';
        
    elseif meth == 2 % Find max before foot of PPG
        
        if nargin == 4
            for i = 1:length(sfootPPG)
                [~, aoPoint(i)] = max(beats(1:sfootPPG(i), i));
            end
        else
            [~, aoPoint] = max(beats);
        end
        aoPoint = aoPoint';
        
   elseif meth == 3 % Find peaks before foot of PPG
        
        for i = 1:length(sfootPPG)
            try
                [~, locs] = findpeaks(beats(1:sfootPPG(i), i), 'MinPeakProminence' ,20);
                aoPoint(i) = locs(end);
            catch
                aoPoint(i) = 490;
            end
        end
        
        aoPoint = aoPoint';
        
        
   elseif meth == 4 % Find peaks before foot of PPG
                  
       for i = 1:length(sfootPPG)
            try
                [~, locs] = findpeaks(beats(1:sfootPPG(i), i), 'MinPeakProminence' , 5);
                
                if length(locs) == 1
                    aoPoint(i) = locs(1);
                else
                    first = 1;                   
                    if locs(1) < 10 && length(locs) > 2
                        first = 2;
                    end                 
                    aoPoint(i) = locs(first+1);
                end
                
            catch
                aoPoint(i) = 490;
            end
            
       end        
        aoPoint = aoPoint';
        
    end
           
 

end