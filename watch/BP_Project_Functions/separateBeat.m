function [array, window_pks] = separateBeat(signal, window_pks, bp, minn)

while window_pks(end) > length(signal) %error with indexing line below window_pks(end) + beat_length needs to be less than signal length
    window_pks = window_pks(1:end-1);
end

ref_window_pks = zeros(length(window_pks),1);
dwpks = diff(window_pks);
k = 1;
while k <= length(dwpks)
    p = 0;
    %     while k+p <= length(dwpks) && dwpks(k+p) < 700
    %         p = p + 1;
    %     end
    [~, loc] = min(signal(window_pks(k:k+p)));
    ref_window_pks(k) = window_pks(k+loc-1);
    k = k + p + 1;
end

window_pks = ref_window_pks(ref_window_pks > 0);


%figure(a);
%hold on;
%plot(window_pks, cust_ppg(window_pks), 'r*');
beat_length = 999; %800
min(diff(window_pks));
if nargin == 4
    beat_length = minn;
end

temp = repmat((0: beat_length)', 1, length(window_pks)) + repmat(window_pks', beat_length+1, 1);

while temp(end) > length(signal) %error with indexing line below window_pks(end) + beat_length needs to be less than signal length
    window_pks = window_pks(1:end-1);
    temp = repmat((0: beat_length)', 1, length(window_pks)) + repmat(window_pks', beat_length+1, 1);
end

array = signal(repmat((0: beat_length)', 1, length(window_pks)) + repmat(window_pks', beat_length+1, 1));


%size(array,2) : length(window_pks)

if bp == 1
    i = 1;
    while i <= size(array, 2)
        beat = array(:, i);
        if (max(beat) - min(beat)) < .1 || mean(diff(beat(1:100))) > -.0001 || max(diff(beat(1:100))) > .002
            array(:, i) = 0;
        end
        i = i + 1;
    end

end
