function filteredSig = filterSig (sig, Fstop1, Fpass1, Fstop2, Fpass2, Fs)
        
    dim = size(sig);
    
    if dim(1) == 1
        sig = sig';
        dim = size(sig);
    end
    
    bpf = bpfParametric(Fstop1, Fpass1, Fstop2, Fpass2, Fs);
    sig = filter(bpf, 1, sig);
    filteredSig = [sig(floor(length(bpf)/2):end, :); zeros(floor(length(bpf)/2) - 1, dim(2))];
    
end


function b = bpfParametric(Fstop1, Fpass1, Fstop2, Fpass2, Fs)

Dstop1 = 0.031622776602;  % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.031622776602;  % Second Stopband Attenuation
flag   = 'scale';         % Sampling Flag

% Calculate the order from the parameters using KAISERORD.
[N,Wn,BETA,TYPE] = kaiserord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 ...
                            1 0], [Dstop1 Dpass Dstop2]);
                        
% Calculate the coefficients using the FIR1 function.
b  = fir1(N, Wn, TYPE, kaiser(N+1, BETA), flag);

end


