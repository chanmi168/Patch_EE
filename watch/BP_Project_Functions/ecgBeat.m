function [pks] = ecgBeat (ecg, env, ratio, varargin)
% determines the peaks in the ecg
bPlot = 0;
% Parse optional input arguments
if ~isempty(varargin)
    for arg = 1:length(varargin)
        if strcmp(varargin{arg}, 'plot'); bPlot = true;
        end
    end
end

pksLoc = find(-diff(sign(diff(ecg))) > .2);
pks = []; 
cnt = 1;
[upper, ~] = envelope(ecg, env, 'peak'); %4000
threshold = ratio*upper; %.6
threshold(threshold < .1) = .1;
minWait = 250;
for ind = 1:length(pksLoc)
   if (ecg(pksLoc(ind)) > threshold(pksLoc(ind)))
       if  cnt == 1
           pks(cnt) = pksLoc(ind);
           cnt = cnt + 1;
       else
           if  pksLoc(ind) > (pks(cnt-1)+minWait)
               pks(cnt) = pksLoc(ind);
               cnt = cnt + 1;
           end
       end
   end
end
pksF = [];

for ind = 1:length(pks)
    tempPks = pksLoc(pksLoc > pks(ind) - 35 & pksLoc < pks(ind) + 35);
    if sum(ecg(tempPks) > .5*ecg(pks(ind))) == 1
        pksF = [pksF pks(ind)];
    end
end

pks = pksF;

if bPlot
 figure; plot(ecg); hold on; plot(pks, ecg(pks), 'r*');
 figure; plot(ecg); hold on; plot(upper);
end
