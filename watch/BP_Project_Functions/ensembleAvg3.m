function [beatsN, snr] = ensembleAvg3 (beats, n, overlap, minSNR)

%{
beats: data in LxM layout, L is length of beats, M is number of beats
n: number of beats to be averaged together
overlap: percent overlap, written in decimal form
%}

beats(:, sum(abs(beats)) > 3*mean(sum(abs(beats)))) = 0;

beatOverlap = floor(n*(1-overlap));
beatOverlap = max([beatOverlap, 1]);
overlap = 1 - beatOverlap/n;

beatsN = zeros(size(beats, 1), floor(size(beats,2)/n + (1/(1-overlap) - 1)*(size(beats,2)/n - 1)));
snr = zeros(size(beatsN, 2), 1);

for i = 1:size(beatsN, 2)

    tempBeat = beats(:, (i-1)*beatOverlap+1:(i-1)*beatOverlap + n);
    
    tempBeat(:, sum(tempBeat) == 0) = [];
    
    [snr(i), snr_beat, beatPlot] = calcSNR2(tempBeat);
    snr(i) = calcSNR2(tempBeat);
%     figure, plot(beatPlot);
    if nargin == 4
        beatsN(:, i) = mean(tempBeat(:, snr_beat > minSNR), 2);
    else
        beatsN(:, i) = mean(tempBeat, 2);
    end

end

end