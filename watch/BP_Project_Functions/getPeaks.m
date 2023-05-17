function [peaks, valleys, inflections] = getPeaks(signal)

% -------------------------------------------------------------------------
% This function returns peaks, valleys, and inflections in a data vector in
% order to extract time-domain features. Input arguments include:
% - dataVector:     [N x 1] Signal vector for analysis
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Extract Peaks
% -------------------------------------------------------------------------

% Differentiate signal
diffSig = diff(signal);

% Initialize return value placeholder
peaks = [];

% Find peaks
for i = 1:length(diffSig)-1
    if diffSig(i) > 0 && diffSig(i+1) < 0
        peaks = [peaks; i];
    end
end

% -------------------------------------------------------------------------
% Extract Valleys
% -------------------------------------------------------------------------

% Initialize return value placeholder
valleys = [];

% Find valleys
for i = 1:length(diffSig)-1
    if diffSig(i) < 0 && diffSig(i+1) > 0
        valleys = [valleys; i];
    end
end

% -------------------------------------------------------------------------
% Extract Inflections
% -------------------------------------------------------------------------

% Differentiate signal
diffdiffSig = diff(diffSig);

% Initialize return value placeholder
inflections_raw = [];

% Find inflections
for i = 1:length(diffdiffSig)-1
    if diffdiffSig(i) < 0 && diffdiffSig(i+1) > 0
        inflections_raw = [inflections_raw; i];
    elseif diffdiffSig(i) > 0 && diffdiffSig(i+1) < 0
        inflections_raw = [inflections_raw; i];
    end
end

% An inflection is only valid if there are no peaks in between the
% inflection and the following (or preceding) inflection (otherwise it's 
% just a normal component of an oscillating signal)
inflections = [];
for i = 1:length(inflections_raw) - 1
    % Get possible peaks in between the inflections
    if isempty(find(peaks > inflections_raw(i) & peaks < inflections_raw(i + 1), 1)) && ...
            isempty(find(valleys > inflections_raw(i) & valleys < inflections_raw(i + 1), 1))
        inflections = [inflections; inflections_raw(i)];	% Valid inflections
    end
end

end

