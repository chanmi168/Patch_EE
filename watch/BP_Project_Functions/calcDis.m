function [dis,range] = calcDis(ptt, bp)

xmin = min(1./ptt);
ymin = min(bp);
xmax = max(1./ptt);
ymax = max(bp);

range = ymax-ymin;

slope = (ymax - ymin)/(xmax-xmin);
intercept = ymin - slope * xmin;

bCand = [slope;intercept];

X = [1./ptt ones(size(ptt,1), 1)];
yEst = X*bCand;
RMSE = sqrt((bp - yEst).^2);

tmp = abs(slope.*(1./ptt)-bp+intercept);
dis = tmp./sqrt(slope^2 + 1);


end