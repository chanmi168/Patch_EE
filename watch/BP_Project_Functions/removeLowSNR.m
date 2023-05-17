function [cleanPPGs, snrValues] = removeLowSNR(ppgArray, pks, cutoff)


snrValues = zeros(1, size(ppgArray, 2));
for i = 1:size(ppgArray, 2)
    
    beatsTemp = ensembleAvg3(separateBeat(ppgArray(:, i), pks, 0), 10, 0);
    snrValues(i) = calcSNR2(beatsTemp);
    
end

cleanPPGs = ppgArray(:, snrValues > max(snrValues) - 20);

