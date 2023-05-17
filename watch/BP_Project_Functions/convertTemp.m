% Convert BME Temperature
function [T, tFine] = convertTemp(tempRaw, tCal)
    var1 = ((double(tempRaw))./16384 - (double(tCal{1}))./1024) .* (double(tCal{2}));
    var2 = (((double(tempRaw))./131072 - (double(tCal{1}))./8192) .* ((double(tempRaw))./131072 - ((double(tCal{1}))./8192))) .* (double(tCal{3}));
    tFine = var1 + var2;
    T = tFine./5120;
end

