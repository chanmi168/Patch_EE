% Convert BME Pressure
function [P] = convertPres(presRaw, tFine, pCal)
    var1 = (double(tFine)./2) - 64000;
    var2 = var1 .* var1 .* (double(pCal{6})) ./ 32768;
    var2 = var2 + var1 .* (double(pCal{5})) .* 2;
    var2 = (var2/4) + ((double(pCal{4})) .* 65536);
    var1 = ((double(pCal{3})) .* var1 .* var1 ./ 524288 + (double(pCal{2})) .* var1) ./ 524288;
    var1 = (1 + var1 ./ 32768) .* (double(pCal{1}));
    if (var1 == 0)
        P = 0;
    else
        p = 1048576 - double(presRaw);
        p = (p - (var2./4096)) .* 6250 ./ var1;
        var1 = (double(pCal{9})) .* p .* p ./ 2147483648;
        var2 = p .* (double(pCal{8})) ./ 32768;
        p = p + (var1 + var2 + (double(pCal{7}))) ./ 16;
        P = p;
    end
end