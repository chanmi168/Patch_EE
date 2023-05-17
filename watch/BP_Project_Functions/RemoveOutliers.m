function [RSel, bpSel, B, range] = RemoveOutliers(ptt,bp,threshold)


bpSel = [];

if size(ptt,1) < 20   
    [dis,range] = calcDis(ptt,bp);
    bpSel = [bpSel; [ptt , bp]];    
    [B, ~, ~, ~, stats1] = regress(bpSel(:,2), [1./bpSel(:,1) ones(size(bpSel,1), 1)]);   
else

    [dis,range] = calcDis(ptt,bp);

    percntiles = prctile(dis,threshold);
    Idx = dis < percntiles;
    bpSel = [bpSel; [ptt(Idx) , bp(Idx)]];

    [B, ~, ~, ~, stats1] = regress(bpSel(:,2), [1./bpSel(:,1) ones(size(bpSel,1), 1)]);
    
end

while B(1) < 0
    
    bpSel = [];
    
    % Remove further points
    cond = ptt>(mean(ptt)-2*std(ptt)) & ptt<(mean(ptt)+2*std(ptt)) & ptt>0;
    ptt = ptt(cond);
    bp = bp(cond);
    
    [dis,range] = calcDis(ptt,bp);

    percntiles = prctile(dis,threshold);
    Idx = dis < percntiles;
    bpSel = [bpSel; [ptt(Idx) , bp(Idx)]];

    [B, ~, ~, ~, stats1] = regress(bpSel(:,2), [1./bpSel(:,1) ones(size(bpSel,1), 1)]);
    
end

    
RSel = sqrt(stats1(1));

end

