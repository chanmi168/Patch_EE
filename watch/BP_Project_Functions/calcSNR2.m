function [snr, snr_beat, beats] = calcSNR2(beats)

beats = beats - (mean(beats)'*ones(1, size(beats,1)))';

ensemble = mean(beats, 2);

a = zeros(1, size(beats, 2));

for i = 1:length(a)
    a(i) = ensemble'*beats(:,i)/(ensemble'*ensemble);
end
a(a>1000) = 1;

noise = beats - ensemble*a;

nsr_beat = var(ensemble).\var(noise);
% nsr_beat = nsr_beat(nsr_beat < mean(nsr_beat) + std(nsr_beat));
% nsr_beat = nsr_beat(nsr_beat < mean(nsr_beat) + std(nsr_beat));
snr = 20*log10(1/(mean(nsr_beat)));

snr_beat = 1./(nsr_beat);
