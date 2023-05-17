function pk2 = findHR (signal, pk1)
pks = [];
minpk = 1.4;
while isempty(pks)
    warning('off')
    [~, pks] = findpeaks(signal(pk1+400:end), 'MinPeakHeight', signal(pk1)/minpk);
    warning('on')
    minpk = minpk + .1;
    
    if minpk >= 2
        [~, pks] = max(signal(pk1+400:end));
    end
end

pk2 = pks(1) + pk1 + 400-1;