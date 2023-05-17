% input is max of PPG to whatever

%method 1: find zero crossings and peaks in - 1stD of signal, finds max
%method 2: max of 2ndD
%method 3: same as method 1, but constraints
%method 4: finds first +1stD max

function loc = findInflection (signal, method)

signal1d = diff(signal);
signal2d = diff(signal1d);

if method == 1
    [~, diffPk] = findpeaks(-abs(signal1d));
    if isempty(diffPk)
        
    end
    diffPk = diffPk(signal2d(diffPk) < 0);
    
    [~, maxPk] = max(signal(diffPk));
    loc = diffPk(maxPk);
end

if method == 2
    [~, max2d] = max(signal2d);
    loc = max2d;
end

if method == 3
    [amp, diffPk] = findpeaks(-abs(signal1d));
    
    diffPk = diffPk(amp > -.00001);
    diffPk = diffPk(signal2d(diffPk) < 0);
    
    if isempty(diffPk)
        [amp, diffPk] = findpeaks(-abs(signal1d));
        diffPk = diffPk(signal2d(diffPk) < 0);
        if isempty(diffPk)
            [~, diffPk] = findpeaks(-signal1d);
        end
    end
    
    [~, maxPk] = max(signal(diffPk));
    loc = diffPk(maxPk);
end

if method == 4
    [~, diffPk] = findpeaks(signal1d);
    
    loc = diffPk(1);
end

if method == 5
    [~, minD] = min(signal1d(100:300));
    [minDPksA, minDPks] = findpeaks(-signal1d(150:400));
    
    minD = minDPks(max(minDPksA)==minDPksA) + 150;
    
    if isempty(minD)
        [~, minD] = min(signal1d(100:300));
        minD = minD + 100;
    end
    
    [amp, diffPk] = findpeaks(signal1d(minD:end));
    
    diffPk = diffPk(amp > -.002);
    loc = diffPk(1) + minD;

end
    
    
    