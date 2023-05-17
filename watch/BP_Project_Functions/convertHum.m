% Convert BME Humidity
function [H] = convertHum(humRaw, tFine, hCal)
    varH = ((double(tFine)) - 76800);
    varH = (humRaw - ((double(hCal{4})) .* 64 + (double(hCal{5})) ./ 16384 .* varH)) ...
         .* ((double(hCal{2})) ./ 65536 .* (1 + (double(hCal{6})) ./ 67108864 .* varH ...
         .* (1 + (double(hCal{3})) ./ 67108864 .* varH)));
     varH = varH .* (1 - (double(hCal{1})) .* varH ./ 524288);
     
     if (varH > 100)
         varH = 100;
     elseif (varH < 0)
         varH = 0;
     end
     H = varH;
end