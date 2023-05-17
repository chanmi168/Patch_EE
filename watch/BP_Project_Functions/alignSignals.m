function [signal1_c, signal2_c, lagSamples, lagTime] = ...
    alignSignals(signal1 ,signal2, Fs, Plot, varargin)

% -------------------------------------------------------------------------
% This function aligns two signals using cross-correlation and returns the
% aligned signals. Input arguments include:
% - signal1:    First signal to be aligned
% - signal2:    Second signal to be aligned
% - Fs1:        Sampling frequency of first signal
% - Fs2:        Sampling frequency of second signal
% - Plot:       Plot results?
% - 'maxlag':   (optional) Maximum lag, in percent of signal length
% - 'method':   (optional) Alignment method ('truncate' or 'pad')
% - 'modify':   (optional) Modification method ('consistent' or 'normal')
%
% NOTE: When 'modify' is set to 'consistent', signal1 is always the
% modified signal!
% -------------------------------------------------------------------------

% Parse optional arguments:
if ~isempty(varargin)
    for arg = 1:length(varargin)
        if strcmp(varargin{arg}, 'maxlag'); maxlag = varargin{arg + 1};
        elseif strcmp(varargin{arg}, 'method'); method = varargin{arg + 1};
        elseif strcmp(varargin{arg}, 'modify'); modify = varargin{arg + 1};
        end
    end
end

% Set defaults
if ~exist('maxlag', 'var'); maxlag = 1.0; end
if ~exist('method', 'var'); method = 'truncate'; end
if ~exist('modify', 'var'); modify = 'normal'; end

% Convert maximum lag to number of samples
maxlag = floor(maxlag*size(signal1, 1));

% Normalize both signals
signal1_n = (signal1 - mean(signal1))./std(signal1);
signal2_n = (signal2 - mean(signal2))./std(signal2);

% Obtain cross-correlation of both signals
[acor, lag] = xcorr(signal1_n, signal2_n, maxlag);

% Get the index of maximum correlation
[~, I] = max(abs(acor));

% Get the lag (in indices and seconds)
lagSamples = lag(I); lagTime = lagSamples/Fs;

% INCONSISENT 
% If lagSamples is negative, that means signal2 lags signal1. Thus, signal2
% will be truncated at the beginning by the lag. In order to maintain
% constant lengths, signal1 will be truncated at the end by the lag. Or,
% signal1 will be padded at the beginning by the lag and truncated at the
% end by the lag.

% If lagSamples is positive, that means signal1 lags signal2. Thus, signal1
% will be truncated at the beginning by the lag. In order to maintain
% constant lengths, signal2 will be truncated at the end by the lag. Or,
% signal2 will be padded at the beginning by the lag and truncated at the
% end by the lag.

% CONSISTENT
% If lagSamples is negative, that means signal2 lags signal1. Thus, signal1
% will be truncated at the end by the lag and padded at the beginning.
% If lagSamples is positive, that means signal1 lags signal2. Thus, signal1
% will be truncated at the beginning by the lag and padded at the end.

% Align the two signals
if strcmp(method, 'truncate')
    if lagSamples < 0       % Signal 2 lags Signal 1
        signal2_c = signal2(abs(lagSamples)+1:end); signal1_c = signal1(1:end-abs(lagSamples));       % Raw
        signal2_cn = signal2_n(abs(lagSamples):end); signal1_cn = signal1_n(1:end-abs(lagSamples)); % Normalized
    elseif lagSamples > 0   % Signal 1 lags Signal 2
        signal1_c = signal1(lagSamples:end); signal2_c = signal2(1:end-lagSamples);         % Raw
        signal1_cn = signal1_n(lagSamples:end); signal2_cn = signal2_n(1:end-lagSamples);   % Normalized
    else
        % Handle the special case where the two signals are perfectly aligned
        signal1_c = signal1; signal2_c = signal2;       % Raw
        signal1_cn = signal1_n; signal2_cn = signal2_n; % Normalized
    end
else
    if lagSamples < 0       % Signal 2 lags Signal 1
        signal1_c = [zeros(lagSamples, 1); signal1]; signal1_c(end-(lagSamples-1):end) = [];
        signal1_cn = [zeros(lagSamples, 1); signal1_n]; signal1_cn(end-(lagSamples-1):end) = [];
        signal2_c = signal2; signal2_cn = signal2_n;
    elseif lagSamples > 0   % Signal 1 lags Signal 2
        if strcmp(modify, 'normal')
            signal2_c = [zeros(lagSamples, 1); signal2]; signal2_c(end-(lagSamples-1):end) = [];
            signal2_cn = [zeros(lagSamples, 1); signal2_n]; signal2_cn(end-(lagSamples-1):end) = [];
            signal1_c = signal1; signal1_cn = signal1_n;
        else
            signal1_c = [signal1; zeros(lagSamples, 1)]; signal1_c(1:lagSamples) = [];
            signal1_cn = [signal1_n; zeros(lagSamples, 1)]; signal1_cn(1:lagSamples) = [];
            signal2_c = signal2; signal2_cn = signal2_n;
        end
    else
        % Handle the special case where the two signals are perfectly aligned
        signal1_c = signal1; signal2_c = signal2;       % Raw
        signal1_cn = signal1_n; signal2_cn = signal2_n; % Normalized
    end
end

% Plot results
if Plot
    
    % Plot the cross-correlation function vs. offset
    figure; hold on; grid on;
    plot(lag, acor)
    xlabel('Lag'); ylabel('Autocorrelation')
    title('Cross-Correlation vs. Lag')
    
    % Plot the original signals
    figure; subplot(2, 1, 1); hold on; grid on;
    plot(signal1_n); plot(signal2_n)
    xlabel('Sample'); ylabel('Amplitude')
    title('Original Signals')
    legend('Signal 1', 'Signal 2')

    % Plot the adjusted signals
    subplot(2, 1, 2); hold on; grid on;
    plot(signal1_c); plot(signal2_c)
    xlabel('Sample'); ylabel('Amplitude')
    title('Aligned Signals')
    legend('Signal 1', 'Signal 2')
end

end
