function [selACC, selSternum1, selWrist1] = findMaxSNR(sternumPPG, wristPPG, ACCs, intervals_end, intervals_dur, ECG)


sSNR1 = [];
sSNR2 = [];
sSNR3 = [];
sSNR4 = [];

wSNR1 = [];
wSNR2 = [];
wSNR3 = [];
wSNR4 = [];

accXSNR = [];
accYSNR = [];
accZSNR = [];

offset = 10;
fs = 1000;
extract = {'BL sit', 'BL stand', 'MA', 'MA rest', 'CP', 'CP rest', 'EX rest'};

for int = 1:length(extract)
    
    M = 10;
   
    tmp_end = intervals_end(extract{int});
    tmp_start = tmp_end - intervals_dur(extract{int});
    section = (tmp_start + offset*fs):(tmp_end - offset*fs);
    
    pks = ecgBeat(ECG(section)) + section(1);
    pks = pks(pks < size(sternumPPG,1));
            
    % ---------------------------------------------------------------------------------------
    % PPG signals
    % ---------------------------------------------------------------------------------------

    % Ensemble Average
    [~, sSNR1_tmp] = ensembleAvg3 (separateBeat(sternumPPG(:, 1), pks, 0), M, 0.5);
    [~, sSNR2_tmp] = ensembleAvg3 (separateBeat(sternumPPG(:, 2), pks, 0), M, 0.5);
    [~, sSNR3_tmp] = ensembleAvg3 (separateBeat(sternumPPG(:, 3), pks, 0), M, 0.5);
    [~, sSNR4_tmp] = ensembleAvg3 (separateBeat(sternumPPG(:, 4), pks, 0), M, 0.5);
    
    [~, wSNR1_tmp] = ensembleAvg3 (separateBeat(wristPPG(:, 1), pks, 0), M, 0.5);
    [~, wSNR2_tmp] = ensembleAvg3 (separateBeat(wristPPG(:, 2), pks, 0), M, 0.5);
    [~, wSNR3_tmp] = ensembleAvg3 (separateBeat(wristPPG(:, 3), pks, 0), M, 0.5);
    [~, wSNR4_tmp] = ensembleAvg3 (separateBeat(wristPPG(:, 4), pks, 0), M, 0.5);
    
    [~, accXSNR_tmp] = ensembleAvg3(separateBeat(ACCs(:,1), pks, 0, 500), M, 0.5);
    [~, accYSNR_tmp] = ensembleAvg3(separateBeat(ACCs(:,2), pks, 0, 500), M, 0.5);
    [~, accZSNR_tmp] = ensembleAvg3(separateBeat(ACCs(:,3), pks, 0, 500), M, 0.5);
    
    % ACC
    accXSNR = [accXSNR; accXSNR_tmp];
    accYSNR = [accYSNR; accYSNR_tmp];
    accZSNR = [accZSNR; accZSNR_tmp];
    
    % Sternum PPG
    sSNR1 = [sSNR1; sSNR1_tmp];
    sSNR2 = [sSNR2; sSNR2_tmp];
    sSNR3 = [sSNR3; sSNR3_tmp];
    sSNR4 = [sSNR4; sSNR4_tmp];
    
    % Wrist PPG
    wSNR1 = [wSNR1; wSNR1_tmp];
    wSNR2 = [wSNR2; wSNR2_tmp];
    wSNR3 = [wSNR3; wSNR3_tmp];
    wSNR4 = [wSNR4; wSNR4_tmp];
    
    % ACC
    accSNR = [accXSNR, accYSNR, accZSNR];
    
    % Sternum PPG
    sSNR = [sSNR1, sSNR2, sSNR3, sSNR4];   
           
    % Wrist PPG
    wSNR = [wSNR1, wSNR2, wSNR3, wSNR4];
      
end

[~,selACC] = max(mean(accSNR,1)); % highest SNR   
[~,selSternum1] = max(mean(sSNR,1)); % highest SNR 
[~,selWrist1] = max(mean(wSNR,1)); % highest SNR   



end


