% % find augmentation index
function AIx = findAIx(ppg) 
    if(length(ppg(isnan(ppg)==0))~=0)
        [pks, locs] = findpeaks(ppg);
        if(length(pks)>1)
            %figure, plot(ppg), hold on, plot(locs(1:2), pks(1:2),'o');
            deltaP = pks(1) - pks(2);
            PP = pks(1);
            if(pks(2)<pks(1))
                AIx = abs(deltaP)/abs(PP);
            else
                AIx = 0;
            end
            %title(['AIx = ' num2str(deltaP) '/' num2str(PP) ' = ' num2str(AIx)]);
        else
            AIx = 0;
        end
    else
        AIx = 0;
    end
end