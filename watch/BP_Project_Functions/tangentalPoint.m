
% meth = 1, finds the intersect of min and tangent to max derivative
% meth = 2, finds absolute 'max' of signal
% meth = 3, max first derivative, someonewhere on the upswing
% meth = 4, max second derivative
% meth = 5, finds local min right before max
% meth = 6, 

function [points, slope] = tangentalPoint (signals, meth, region)

    if length(signals) == 0
        points = 0;
        slope = 0;
        return;
    end
    
    start = 0;
    points = zeros(size(signals,2), 1);
    
    for int = 1:size(signals, 2)

        % finds the intersect of min and tangent to max derivative
        signal1d = diff(signals(:, int));

        bpf = ones(10, 1)/10;
        signal1d = filter(bpf, 1, signal1d);
        signal1d = signal1d(floor(length(bpf)/2):end);
        signal2d = diff(signal1d);
        
        if meth == 1
            [slope, loc_x] = max(diff(signals(1:size(signals,1)-200,int)));
            
            if nargin == 3
                [slope, loc_x] = max(diff(signals(region,int)));
            end

            loc_y = signals(loc_x, int);
            [max_y, loc_max] = max(signals(1:size(signals,1),int));

            [min_y, loc_min] = min(signals(loc_x - loc_x+1:loc_max,int));

            b = loc_y - slope * loc_x;

            points(int) = floor((min_y - b)/(slope));
            [~, points(int)] = max(signal2d);
        
            points(int) = floor((min_y - b)/(slope));
                 
        end

        % finds 'max' of signal
        if meth == 2
            [~, points(int)] = max(signals(:, int));
            points(int) = points(int) - 1;
        end

        % max first derivative, someonewhere on the upswing
        if meth == 3
            [~, points(int)] = max(signal1d);
        end

        % max second derivative
        if meth == 4
            [~, points(int)] = max(signal2d);
        end

        % finds min right before max
        if meth == 5
            [~, locMax] = max(signals(floor(length(signals(:, int))/2):end, int));
            locMax = locMax + floor(length(signals(:, int))/2) - 1;
            [~, locMin] = findpeaks(-signals(1:locMax, int));
            if isempty(locMin)
                locMin = 1;
            end
            points(int) = locMin(end);
        end

        % first point before maximum that is either a peak or zerocrossing
        % of first derivative. 
        if meth == 6
            [~, locMax] = max(diff(signals(floor(length(signals(:, int))/2):end, int)));
            locMax = locMax + floor(length(signals(:, int))/2) - 1;
            range = min([locMax-3, length(signal1d)]);
            [amp, locMin] = findpeaks(-abs(signal1d(1:range)));
            if ~isempty(locMin(amp > -.0001))
                locMin = locMin(amp > -.0001 & signals(locMin, int) < .3);
                %locMin = locMin(amp > -.0001);
            else
                [~, maxamp] = max(amp);
                locMin = locMin(maxamp);
            end

            if isempty(locMin)
                locMin = 1;
            end
            points(int) = locMin(end);
        end

        if meth == 7
            points(int) = floor((max_y - b)/(slope));
            
            if points(int) <= 0
                points(int) = 1;
            end
        end
        
        % tangental method, but uses max slope right before the max
        if meth == 8
            [~, diffPks] = findpeaks(-signals(:, int));
            diffPks = diffPks(diffPks < loc_max);
            
            try
            diffPks = [1; diffPks];
            catch
                pause(.01);
            end

            [slope, loc_x] = max(diff(signals(diffPks(end):length(signals),int)));
            loc_x = loc_x + diffPks(end);
            loc_y = signals(loc_x, int);
            
            b = loc_y - slope * loc_x;
            points(int) = floor((min_y - b)/(slope));
            
            if points(int) > 220-50
                pause(.01);
            end
        end
        
        if meth == 9
            [amp, diffPks] = findpeaks(-signals(:, int));
            diffPks = diffPks(-amp < (max(signals(:, int)) + min(signals(:, int)))/2);
            diffPks = diffPks(diffPks < loc_max);
            
            try
                diffPks = [1; diffPks];
            catch
                pause(.01);
            end

            [slope, loc_x] = max(diff(signals(diffPks(end):length(signals),int)));
            loc_x = loc_x + diffPks(end);
            loc_y = signals(loc_x, int);
            
            b = loc_y - slope * loc_x;
            try
            points(int) = floor((-amp(end) - b)/(slope));
            catch
                a = 0;
            end
            
            if points(int) <150
                pause(.01);
            end
        end
        
        if meth == 10
            [amp, diffPks] = findpeaks(-signals(:, int));
            diffPks = diffPks(-amp < (max(signals(:, int)) + min(signals(:, int)))/2);
            
            diffPks = diffPks(diffPks < loc_max);
            
            try
                diffPks = [1; diffPks];
            catch
                pause(.01);
            end
            
            [slope, loc_x] = max(diff(signals(diffPks(end):length(signals),int)));
            loc_x = loc_x + diffPks(end);
            loc_y = signals(loc_x, int);
            
            b = loc_y - slope * loc_x;
            try
            points(int) = floor((-amp(end) - b)/(slope));
            catch
                a = 0;
            end
            
            if points(int) <150
                pause(.01);
            end
        end
        
        if points(int) < 1
            points(int) = 1;
        elseif points(int) > length(signal1d)
            points(int) = length(signal1d);
        end
        
    end

end