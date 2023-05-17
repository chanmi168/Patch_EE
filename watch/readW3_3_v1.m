clear all;
%close all;
addpath('BP_Project_Functions');
plotGraphs = 0;

%% New Header Read
headerLength = 512;
%folder = '/Pilot/test/015/'; %place subject file into new folder with sujbect ID and update this to run
% folder = '\V3\test_data\008\';
% folder = '\home\mchan\Estimation_EE\data\raw\sub104\watch\';
% folder = '/watch_data/';
folder = '../../data/raw/sub104/watch/';

%%
% folder = '\001\'; %place subject file into new folder with sujbect ID and update this to run
% directory = [cd folder];
%dataSet = 'ALINE0001';
directory = folder;

files = dir([directory 'HP*.bin']);
for i = 1:length(files)
    filename = files(i).name;
    dateFile = find(filename == '-', 1, 'last');
    dateFile = filename(dateFile+1:dateFile+15);
    str = 'January 1, 1900 00:00:00';
    %     t1 = datevec(str, 'mmmm dd,s yyyy HH:MM:SS');
    %     t2 = [str2num(dateFile(1:4)) str2num(dateFile(5:6)) str2num(dateFile(7:8)) str2num(dateFile(10:11)) str2num(dateFile(12:13)) str2num(dateFile(14:15))];
    %     startTime = etime(t2, t1);
    startTime = 0;
    
    fid = fopen(strcat(directory, filename));
    header = fread(fid,headerLength,'uint8=>uint8');
    AST_SR = 16384/(2*2);
    
    % Pull out file information from header
    deviceSerial = header(1:16,:);
    HP_Serial = header(23:24,:);
    devFirmwareVersion = header(25:28,:);
    BME_Cal = header(29:61,:);
    fileTime_Year = header(62:65,:);
    fileTime_Month = header(66:69,:);
    fileTime_Day = header(74:77,:);
    
    % Convert from bytes into integerf
    deviceSerial = double(typecast(deviceSerial(:),'uint32'));
    HP_Serial = double(typecast(HP_Serial(:),'uint16'));
    fileTime_Year = double(typecast(fileTime_Year(:),'uint32'));
    fileTime_Month = double(typecast(fileTime_Month(:),'uint32'));
    fileTime_Day = double(typecast(fileTime_Day(:),'uint32'));
    
    % Move to the correct place on the SD card and Pull data
    fseek(fid,headerLength,'bof');
    A = fread(fid,'uint8=>uint8');
    fclose(fid);
    
    %% reshape data
    % Shape into packets that are the buffer length long
    A = A(1:length(A)-mod(length(A),8192));
    A = reshape(A,4096*2,floor(length(A)/4096/2)); % one column per 4096
    B = A(8121:8184,:); % Pull BME data
    V2 = reshape(B,16,size(B,2)*4); %reshape from MxN to M/13*N*4
    
    A = A(1:8120,:); % ECG and ACCEL data
    A = reshape(A,58,size(A,2)*140); % re-arrange into sample sizes
    %% Convert and Extract BME Data
    % pull BME time
    B_time = B(4:-1:1,:); %Correct byte order
    B_time = double(typecast(B_time(:),'uint32'));
    B_time = (B_time-B_time(1))/AST_SR; %seconds
    %figure, plot(B_time);
    % convert all the values
    [tCal, pCal, hCal, tempRaw, presRaw, humRaw] = extractBMEVals(BME_Cal, B);
    [B_temp, tFine]  = convertTemp(tempRaw, tCal);
    B_pres = convertPres(presRaw, tFine, pCal);
    B_humi = convertHum(humRaw, tFine, hCal);
    % Old BME Data code with conversion on board
    % B_pres = B(8:-1:5,:);
    % B_pres = double(typecast(B_pres(:),'uint32')); % Pascals
    % B_temp = B(12:-1:9,:);
    % B_temp = double(typecast(B_temp(:),'int32'))/100; % Degrees celcius
    % B_humi = B(16:-1:13,:);
    % B_humi = double(typecast(B_humi(:),'uint32'))/1024; % Realtive humidity %
    % %B_alt = -((287.053*(273.15+B_temp))/9.80665).*log(B_pres/101325);
    % %https://en.wikipedia.org/wiki/Atmospheric_pressure
    % %https://en.wikipedia.org/wiki/Vertical_pressure_variation
    % %clear B; % done with B, all data has been pushed
    %% ECG time and data
    E_time = A(4:-1:1,:); %start of sampling
    E_time = double(typecast(E_time(:),'uint32'));
    E_time = (E_time)/AST_SR + startTime;            %seconds
    %figure; plot(E_time); 
    
    E_data = A(10:-1:8,:);
    E_data = [zeros(1,size(E_data,2)); E_data];
    E_data = double(typecast(E_data(:),'int32'));
    
    % Figure the converstion factor for counts to volts
    E_data_gain = 3;
    E_data_vRef = 2.42;
    E_data_volts_per_bit = (E_data_vRef/E_data_gain)/(2^23-1);
    
    % Convert to volts
    E_data = (E_data * E_data_volts_per_bit)/(2^8); %adjust for bit shifting
    
%     %don't take data when ECG is zero (data wasnt ready)
%      zeroMask = E_data~=0;
%      E_data = E_data(zeroMask);
%      E_time = E_time(zeroMask);
    
    LOFF_RLD = bitget(A(5,:),4);
    LOFF_IN1N = bitget(A(5,:),1);
    LOFF_IN1P = bitget(A(6,:),8);
    i_LOFF = logical(LOFF_RLD|LOFF_IN1N|LOFF_IN1P);
    %i_LOFF = logical(LOFF_IN1P);
    i_LON = ~i_LOFF;
    
    E_time_2 = A(57:-1:54, :); % end sampling
    E_time_2 = double(typecast(E_time_2(:),'uint32'));
    E_time_2 = (E_time_2)/AST_SR + startTime;            %seconds
    
%     %don't take data when ECG is zero (data wasnt ready)
%      E_time_2 = E_time_2(zeroMask);

    figure; subplot(2, 1, 1);
    plot(E_time, E_time_2 - E_time,E_time(1:(end-1):end),mean(E_time_2 -E_time)*[1 1], 'r-');
    title('Sample Time'), ylim([0 .002]);
    subplot(2, 1, 2); hist(E_time_2 - E_time,linspace(0,5e-3,25));
    title('Hist Diff ECG time'); xlabel('Seconds'); ylabel('Count'); grid on;
    set(gca,'YScale','log');
    
    % There is edge case where there is no connection what-so-ever
    % throws a warning and the legend is messed up, need to deal with
    
    %% Accel data and time 
           
    %Updated:
    % shift by 256 to keep the sign
    % divide by 16 to kill the 4 bits of bad data1 at the bottom
    % divide by 256000 (bits/g)
    
    A_data_x = A(14:-1:12,:);
    A_data_x = [zeros(1,size(A_data_x,2)); A_data_x];
    A_data_x = double(typecast(A_data_x(:),'int32')) / 256 / 16 / 256000; %/2
    
    A_data_y = A(17:-1:15,:);
    A_data_y = [zeros(1,size(A_data_y,2)); A_data_y];
    A_data_y = double(typecast(A_data_y(:),'int32')) / 256 / 16 / 256000; %/2

    A_data_z = A(20:-1:18,:);
    A_data_z = [zeros(1,size(A_data_z,2)); A_data_z];
    A_data_z = double(typecast(A_data_z(:),'int32')) / 256 / 16 / 256000; %/2
 
    % generate a Mask that marks where there was NO data actually taken
    % and apply to all the data to kill the "bad" data locations
    zeroMask = A_data_x~=0 | A_data_y~=0 | A_data_z~=0;
    A_data_x = A_data_x(zeroMask);
    A_data_y = A_data_y(zeroMask);
    A_data_z = A_data_z(zeroMask);
    A_time = E_time(zeroMask);
    
    diffE_time = diff(E_time);
    diffA_time = diff(A_time);
    figure; % Compare to Seconds
    subplot(2,1,1)
    plot(E_time(1:end-1),diffE_time,E_time(1:(end-1):end),mean(diffE_time)*[1 1],'r-');
    title('Diff ECG time'); xlabel('Seconds'); ylabel('Seconds'); xlim(E_time([1 end])), ylim([0 0.002]);
    subplot(2,1,2)
    plot(A_time(1:end-1),diffA_time,A_time(1:(end-1):end),mean(diffA_time)*[1 1],'r-');
    title('Diff ACCEL time'); xlabel('Seconds'); ylabel('Seconds'); xlim(A_time([1 end])), ylim([0 0.01]);
    
    %check ecg scg alignment
    figure, plot(E_time,E_data*10), hold on, plot(A_time,A_data_z);
    
    %accFifoClear = (A(63,:)==0);
    %5igure, plot(A_data1_z,'.'); hold on, plot(accFifoClear,mean(A_data1_z)*ones(length(accFifoClear),1),'.','MarkerSize',20);
       
%%  Gyroscope data
    G_data_x = A(22:23, :);
    G_data_x = double(typecast(G_data_x(:), 'int16'));
    
    G_data_y = A(24:25, :);
    G_data_y = double(typecast(G_data_y(:), 'int16'));
    
    G_data_z = A(26:27, :);
    G_data_z = double(typecast(G_data_z(:), 'int16')); 
    
    zeroMask = G_data_x~=0 | G_data_y~=0 | G_data_z~=0;
    G_data_x = G_data_x(zeroMask);
    G_data_y = G_data_y(zeroMask);
    G_data_z = G_data_z(zeroMask);
    G_time = E_time(zeroMask);
    
    figure, subplot(2,1,1), plot(G_data_x), title('Gyroscope');
    diffG_time = diff(G_time);
    subplot(2,1,2), plot(G_time(1:end-1),diffG_time,G_time(1:(end-1):end),mean(diffG_time)*[1 1],'r-');
    title('Diff Gyro time'); xlabel('Seconds'); ylabel('Seconds'); xlim(G_time([1 end])), ylim([0 0.01]);
    
    %save([dataSet 'mat'], 'A_time', 'E_time', 'B_time', 'A_data_x', 'A_data_y', 'A_data_z', 'E_data', 'B_temp');
    
%% Sternum PPG
    % PPGW data and time
    % Bytes 32 to 40 | Data is formatted as follows:
    
    % Note: There are 3 PPGWS and Three Wavelengths each (Red, IR, Green)
    % Each PPGW has 3 bytes of data for each color so 9 bytes total
    % 3 PPGWs -> 27 bytes; The first 5 bits of the MSB for each color
    % represent the ppgS tag code (i.e., tells you what color it is, but
    % they should be coming in order anyway (which is red-ir-green for this
    % iteration)
    
    %     They all come in when the first one (A(31,:) = PPGW_toggle = 1) does,
    %     so align in time. NOTE: Due to toggling in succession, theoretically
    %     they are coming in miliseconds apart due to sample conversion time so
    %     its neglible given the PPGW signal bandwidth.
    %     Essentially make ppgS_2/3 time = ppgS_1 time for the wavelength
    
    % During continuous mode the green LED though normally LED 3 (tag = 3)
    % is placed in the LEDC1 slot therefore its tag becomes 1
    % updated firmware such that it waits to switch between modes only
    % after all three are read
    %when user is done though last ones might not have been registered if
    %promptly plugged in? truncate end at start or just ppg data?
    if diff(diff([size(find(A(31,:)==1),2) size(find(A(31,:)==2),2) size(find(A(31,:)==3),2)]))~=0
        temp = find(A(31,:)==1);
        A(31:40,temp(end):end) = 0;
    end
    
    %find the location of the first toggle and then ensure 2 and 3 follow
    
   %Grab continuous mode data, and change tag markers (or could not use
   %tags...)
    continuousMode_index = intersect(find(A(32,:)==0),find(A(38,:)~=0));
    A(32,continuousMode_index) = 8;
    A(35,continuousMode_index) = 16;
    A(38,continuousMode_index) = 24;
    
    ppg_toggle = double(A(31,:));
    figure, plot(ppg_toggle,'.'), title('PPG Sternum 1 Toggle');
    
    %   PPGW 1  
    ppgS_1_mask = A(31,:) == 1;
    
    PPGW_1_tag_order = [bitand(A(32, ppgS_1_mask), 248)/2^3; bitand(A(35, ppgS_1_mask), 248)/2^3; bitand(A(38, ppgS_1_mask), 248)/2^3];
    PPGW_1_tag_order = reshape((PPGW_1_tag_order), 1, numel(PPGW_1_tag_order));
     
    ppgS_1_toggle = double(A(31, ppgS_1_mask));
    figure, plot(diff(ppgS_1_toggle),'.'), title('PPG Sternum 1 Toggle');
    
    ppgS_1_current_led1 = double(A(28,ppgS_1_mask))*0.24; %62mA range
    ppgS_1_current_led2 = double(A(29,ppgS_1_mask))*0.24;
    ppgS_1_current_led3 = double(A(30,ppgS_1_mask))*0.24;
    figure, subplot(3,1,1), plot(ppgS_1_current_led1,'.'), title('PPG Sternum 1 Current LED1'), subplot(3,1,2), plot(ppgS_1_current_led2,'.'), title('PPG Sternum 1 Current LED2'), subplot(3,1,3), plot(ppgS_1_current_led3,'.'), title('PPG Sternum 1 Current LED3');
    
    ppgS_1 = [A(34:-1:32, ppgS_1_mask); zeros(1,size(A(34:-1:32, ppgS_1_mask),2)); A(37:-1:35, ppgS_1_mask); zeros(1,size(A(37:-1:35, ppgS_1_mask),2)); A(40:-1:38, ppgS_1_mask); zeros(1,size(A(40:-1:38, ppgS_1_mask),2));];
    ppgS_1(3, :) = bitand(ppgS_1(3, :), 7);
    ppgS_1(7, :) = bitand(ppgS_1(7, :), 7);
    ppgS_1(11, :) = bitand(ppgS_1(11, :), 7);
    ppgS_1 = double(typecast(ppgS_1(:), 'int32'));
    
    ppgS_1_time = repelem(E_time(ppgS_1_mask), 3); %three tags.
    
    ppgS_time = ppgS_1_time(PPGW_1_tag_order == 1);
    ppgS_R1 =  ppgS_1(PPGW_1_tag_order == 1);
    ppgS_I1 = ppgS_1(PPGW_1_tag_order == 2);
    ppgS_G1 =  ppgS_1(PPGW_1_tag_order == 3);

    if 0
        figure; plot(ppgS_time, ppgS_R1); hold on, plot(ppgS_time,ppgS_I1), plot(ppgS_time,ppgS_G1); title('PPG Sternum 1'); legend({'Red PPG Sternum','IR PPG Sternum', 'Green PPG Sternum'});
        
        diffppgS_R1_time = diff(ppgS_time); diffppgS_I1_time = diff(ppgS_time); diffppgS_G1_time = diff(ppgS_time);
        figure, ax1 = subplot(3,1,1), plot(ppgS_time(1:end-1),diffppgS_R1_time,ppgS_time(1:(end-1):end),mean(diffppgS_R1_time)*[1 1],'r-'), ylim([0.001 0.01]), title('Diff PPG Sternum 1 time');
        ax2 = subplot(3,1,2), plot(ppgS_time(1:end-1),diffppgS_I1_time,ppgS_time(1:(end-1):end),mean(diffppgS_I1_time)*[1 1],'r-'), ylim([0.001 0.01]);
        ax3 = subplot(3,1,3), plot(ppgS_time(1:end-1),diffppgS_G1_time,ppgS_time(1:(end-1):end),mean(diffppgS_G1_time)*[1 1],'r-'), ylim([0.001 0.01]);
        linkaxes([ax1 ax2 ax3],'x');
    end
    
 %  PPGW 2  
    ppgS_2_mask = A(31,:) == 2;
    
    ppgS_2_tag_order = [bitand(A(32, ppgS_2_mask), 248)/2^3; bitand(A(35, ppgS_2_mask), 248)/2^3; bitand(A(38, ppgS_2_mask), 248)/2^3];
    ppgS_2_tag_order = reshape((ppgS_2_tag_order), 1, numel(ppgS_2_tag_order));
     
    ppgS_2_toggle = double(A(31, ppgS_2_mask));
    figure, plot(ppgS_2_toggle,'.'), title('PPG Sternum 2 Toggle');
    
    ppgS_2_current_led1 = double(A(28,ppgS_2_mask))*0.24; %62mA range
    ppgS_2_current_led2 = double(A(29,ppgS_2_mask))*0.24;
    ppgS_2_current_led3 = double(A(30,ppgS_2_mask))*0.24;
    figure, subplot(3,1,1), plot(ppgS_2_current_led1,'.'), title('PPG Sternum 2 Current LED1'), subplot(3,1,2), plot(ppgS_2_current_led2,'.'), title('PPG Sternum 2 Current LED2'), subplot(3,1,3), plot(ppgS_2_current_led3,'.'), title('PPG Sternum 2 Current LED3');
    
    ppgS_2 = [A(34:-1:32, ppgS_2_mask); zeros(1,size(A(34:-1:32, ppgS_2_mask),2)); A(37:-1:35, ppgS_2_mask); zeros(1,size(A(37:-1:35, ppgS_2_mask),2)); A(40:-1:38, ppgS_2_mask); zeros(1,size(A(40:-1:38, ppgS_2_mask),2));];
    ppgS_2(3, :) = bitand(ppgS_2(3, :), 7);
    ppgS_2(7, :) = bitand(ppgS_2(7, :), 7);
    ppgS_2(11, :) = bitand(ppgS_2(11, :), 7);
    ppgS_2 = double(typecast(ppgS_2(:), 'int32'));
    
    ppgS_R2 =  ppgS_2(ppgS_2_tag_order == 1);
    ppgS_I2 = ppgS_2(ppgS_2_tag_order == 2);
    ppgS_G2 =  ppgS_2(ppgS_2_tag_order == 3);
    
    if 0
        figure; plot(ppgS_time, ppgS_R2); hold on, plot(ppgS_time,ppgS_I2), plot(ppgS_time,ppgS_G2); title('PPG Sternum 2'); legend({'Red PPG Sternum','IR PPG Sternum', 'Green PPG Sternum'});
       
        diffppgS_R2_time = diff(ppgS_time); diffppgS_I2_time = diff(ppgS_time); diffppgS_G2_time = diff(ppgS_time);
        figure, ax1 = subplot(3,1,1), plot(ppgS_time(1:end-1),diffppgS_R2_time,ppgS_time(1:(end-1):end),mean(diffppgS_R2_time)*[1 1],'r-'), ylim([0.001 0.01]), title('Diff PPG Sternum 2 time');
        ax2 = subplot(3,1,2), plot(ppgS_time(1:end-1),diffppgS_I2_time,ppgS_time(1:(end-1):end),mean(diffppgS_I2_time)*[1 1],'r-'), ylim([0.001 0.01]);
        ax3 = subplot(3,1,3), plot(ppgS_time(1:end-1),diffppgS_G2_time,ppgS_time(1:(end-1):end),mean(diffppgS_G2_time)*[1 1],'r-'), ylim([0.001 0.01]);
        linkaxes([ax1 ax2 ax3],'x');
    end
    
  %  PPGW 3
    ppgS_3_mask = A(31,:) == 3;
    
    ppgS_3_tag_order = [bitand(A(32, ppgS_3_mask), 248)/2^3; bitand(A(35, ppgS_3_mask), 248)/2^3; bitand(A(38, ppgS_3_mask), 248)/2^3];
    ppgS_3_tag_order = reshape((ppgS_3_tag_order), 1, numel(ppgS_3_tag_order));
     
    ppgS_3_toggle = double(A(31, ppgS_3_mask));
    figure, plot(ppgS_3_toggle,'.'), title('PPG Sternum 3 Toggle');
    
    ppgS_3_current_led1 = double(A(28,ppgS_3_mask))*0.24; %62mA range
    ppgS_3_current_led2 = double(A(29,ppgS_3_mask))*0.24;
    ppgS_3_current_led3 = double(A(30,ppgS_3_mask))*0.24;
    figure, subplot(3,1,1), plot(ppgS_3_current_led1,'.'), title('PPG Sternum 3 Current LED1'), subplot(3,1,2), plot(ppgS_3_current_led2,'.'), title('PPG Sternum 3 Current LED2'), subplot(3,1,3), plot(ppgS_3_current_led3,'.'), title('PPG Sternum 3 Current LED3');
    
    ppgS_3 = [A(34:-1:32, ppgS_3_mask); zeros(1,size(A(34:-1:32, ppgS_3_mask),2)); A(37:-1:35, ppgS_3_mask); zeros(1,size(A(37:-1:35, ppgS_3_mask),2)); A(40:-1:38, ppgS_3_mask); zeros(1,size(A(40:-1:38, ppgS_3_mask),2));];
    ppgS_3(3, :) = bitand(ppgS_3(3, :), 7);
    ppgS_3(7, :) = bitand(ppgS_3(7, :), 7);
    ppgS_3(11, :) = bitand(ppgS_3(11, :), 7);
    ppgS_3 = double(typecast(ppgS_3(:), 'int32'));
    
    ppgS_R3 =  ppgS_3(ppgS_3_tag_order == 1);
    ppgS_I3 = ppgS_3(ppgS_3_tag_order == 2);
    ppgS_G3 =  ppgS_3(ppgS_3_tag_order == 3);

    if 0
        figure; plot(ppgS_time, ppgS_R3); hold on, plot(ppgS_time,ppgS_I3), plot(ppgS_time,ppgS_G3); title('PPG Sternum 3'); legend({'Red PPG Sternum','IR PPG Sternum', 'Green PPG Sternum'});
        diffppgS_R3_time = diff(ppgS_time); diffppgS_I3_time = diff(ppgS_time); diffppgS_G3_time = diff(ppgS_time);
        
        figure, ax1 = subplot(3,1,1), plot(ppgS_time(1:end-1),diffppgS_R3_time,ppgS_time(1:(end-1):end),mean(diffppgS_R3_time)*[1 1],'r-'), ylim([0.001 0.01]), title('Diff PPG Sternum 3 time');
        ax2 = subplot(3,1,2), plot(ppgS_time(1:end-1),diffppgS_I3_time,ppgS_time(1:(end-1):end),mean(diffppgS_I3_time)*[1 1],'r-'), ylim([0.001 0.01]);
        ax3 = subplot(3,1,3), plot(ppgS_time(1:end-1),diffppgS_G3_time,ppgS_time(1:(end-1):end),mean(diffppgS_G3_time)*[1 1],'r-'), ylim([0.001 0.01]);
        linkaxes([ax1 ax2 ax3],'x');
    end
%% Finger PPG 
    % PPGS data and time
    % Bytes 42 to 50 | Data is formatted as follows:
    
    % Note: There are 3 PPGSS and Three Wavelengths each (Red, IR, Green)
    % Each PPGS has 3 bytes of data for each color so 9 bytes total
    % 3 PPGSs -> 27 bytes; The first 5 bits of the MSB for each color
    % represent the ppgF tag code (i.e., tells you what color it is, but
    % they should be coming in order anyway (which is red-ir-green for this
    % iteration)
    
    %     They all come in when the first one (A(41,:) = PPGS_toggle = 1) does,
    %     so align in time. NOTE: Due to toggling in succession, theoretically
    %     they are coming in miliseconds apart due to sample conversion time so
    %     its neglible given the PPGS signal bandwidth.
    %     Essentially make ppgF_2/3 time = ppgF_1 time for the wavelength
    %     wait to switch modes (ECG or accel data might be bad)
    %when user is done though last ones might not have been registered if
    %promptly plugged in? truncate end at start or just ppg data?
    if diff(diff([size(find(A(41,:)==1),2) size(find(A(41,:)==2),2) size(find(A(41,:)==3),2)]))~=0
        temp = find(A(41,:)==1);
        A(41:50,temp(end):end) = 0;
    end
    
%   PPGS 1  
    ppgF_1_mask = A(41,:) == 1;
    
    PPGS_1_tag_order = [bitand(A(42, ppgF_1_mask), 248)/2^3; bitand(A(45, ppgF_1_mask), 248)/2^3; bitand(A(48, ppgF_1_mask), 248)/2^3];
    PPGS_1_tag_order = reshape((PPGS_1_tag_order), 1, numel(PPGS_1_tag_order));
     
    ppgF_1_toggle = double(A(41, ppgF_1_mask));
    figure, plot(ppgF_1_toggle,'.'), title('PPG Finger 1 Toggle');
    
    ppgF_1_current_led1 = double(A(51,ppgF_1_mask))*0.24; %62mA range
    ppgF_1_current_led2 = double(A(52,ppgF_1_mask))*0.24;
    ppgF_1_current_led3 = double(A(53,ppgF_1_mask))*0.24;
    figure, subplot(3,1,1), plot(ppgF_1_current_led1,'.'), title('PPG Finger 1 Current LED1'), subplot(3,1,2), plot(ppgF_1_current_led2,'.'), title('PPG Finger 1 Current LED2'), subplot(3,1,3), plot(ppgF_1_current_led3,'.'), title('PPG Finger 1 Current LED3');
    
    ppgF_1 = [A(44:-1:42, ppgF_1_mask); zeros(1,size(A(44:-1:42, ppgF_1_mask),2)); A(47:-1:45, ppgF_1_mask); zeros(1,size(A(47:-1:45, ppgF_1_mask),2)); A(50:-1:48, ppgF_1_mask); zeros(1,size(A(50:-1:48, ppgF_1_mask),2));];
    ppgF_1(3, :) = bitand(ppgF_1(3, :), 7);
    ppgF_1(7, :) = bitand(ppgF_1(7, :), 7);
    ppgF_1(11, :) = bitand(ppgF_1(11, :), 7);
    ppgF_1 = double(typecast(ppgF_1(:), 'int32'));
    
    PPGS_1_time = repelem(E_time(ppgF_1_mask), 3); %three tags.
    
    ppgF_time = PPGS_1_time(PPGS_1_tag_order == 1);
    ppgF_I1 =  ppgF_1(PPGS_1_tag_order == 1);
    ppgF_R1 = ppgF_1(PPGS_1_tag_order == 2);
    ppgF_G1 =  ppgF_1(PPGS_1_tag_order == 3);

    if 0
        figure; plot(ppgF_time, ppgF_R1); hold on, plot(ppgF_time,ppgF_I1), plot(ppgF_time,ppgF_G1); title('PPG Finger 1'); legend({'Red PPG Finger','IR PPG Finger', 'Green PPG Finger'});
        diffppgF_R1_time = diff(ppgF_time); diffppgF_I1_time = diff(ppgF_time); diffppgF_G1_time = diff(ppgF_time);
        
        figure, ax1 = subplot(3,1,1), plot(ppgF_time(1:end-1),diffppgF_R1_time,ppgF_time(1:(end-1):end),mean(diffppgF_R1_time)*[1 1],'r-'), ylim([0.001 0.01]), title('Diff PPG Finger 1 time');
        ax2 = subplot(3,1,2), plot(ppgF_time(1:end-1),diffppgF_I1_time,ppgF_time(1:(end-1):end),mean(diffppgF_I1_time)*[1 1],'r-'), ylim([0.001 0.01]);
        ax3 = subplot(3,1,3), plot(ppgF_time(1:end-1),diffppgF_G1_time,ppgF_time(1:(end-1):end),mean(diffppgF_G1_time)*[1 1],'r-'), ylim([0.001 0.01]);
        linkaxes([ax1 ax2 ax3],'x');
    end
    
 %  PPGS 2  
    ppgF_2_mask = A(41,:) == 2;
    
    ppgF_2_tag_order = [bitand(A(42, ppgF_2_mask), 248)/2^3; bitand(A(45, ppgF_2_mask), 248)/2^3; bitand(A(48, ppgF_2_mask), 248)/2^3];
    ppgF_2_tag_order = reshape((ppgF_2_tag_order), 1, numel(ppgF_2_tag_order));
     
    ppgF_2_toggle = double(A(41, ppgF_2_mask));
    figure, plot(ppgF_2_toggle,'.'), title('PPG Finger 2 Toggle');
    
    ppgF_2_current_led1 = double(A(51,ppgF_2_mask))*0.24; %62mA range
    ppgF_2_current_led2 = double(A(52,ppgF_2_mask))*0.24;
    ppgF_2_current_led3 = double(A(53,ppgF_2_mask))*0.24;
    figure, subplot(3,1,1), plot(ppgF_2_current_led1,'.'), title('PPG Finger 2 Current LED1'), subplot(3,1,2), plot(ppgF_2_current_led2,'.'), title('PPG Finger 2 Current LED2'), subplot(3,1,3), plot(ppgF_2_current_led3,'.'), title('PPG Finger 2 Current LED3');
    
    ppgF_2 = [A(44:-1:42, ppgF_2_mask); zeros(1,size(A(44:-1:42, ppgF_2_mask),2)); A(47:-1:45, ppgF_2_mask); zeros(1,size(A(47:-1:45, ppgF_2_mask),2)); A(50:-1:48, ppgF_2_mask); zeros(1,size(A(50:-1:48, ppgF_2_mask),2));];
    ppgF_2(3, :) = bitand(ppgF_2(3, :), 7);
    ppgF_2(7, :) = bitand(ppgF_2(7, :), 7);
    ppgF_2(11, :) = bitand(ppgF_2(11, :), 7);
    ppgF_2 = double(typecast(ppgF_2(:), 'int32'));
    
    ppgF_I2 =  ppgF_2(ppgF_2_tag_order == 1);
    ppgF_R2 = ppgF_2(ppgF_2_tag_order == 2);
    ppgF_G2 =  ppgF_2(ppgF_2_tag_order == 3);
    
    if 0
        figure; plot(ppgF_time, ppgF_R2); hold on, plot(ppgF_time,ppgF_I2), plot(ppgF_time,ppgF_G2); title('PPG Finger 2'); legend({'Red PPG Finger','IR PPG Finger', 'Green PPG Finger'});
        diffppgF_R2_time = diff(ppgF_time); diffppgF_I2_time = diff(ppgF_time); diffppgF_G2_time = diff(ppgF_time);
        
        figure, ax1 = subplot(3,1,1), plot(ppgF_time(1:end-1),diffppgF_R2_time,ppgF_time(1:(end-1):end),mean(diffppgF_R2_time)*[1 1],'r-'), ylim([0.001 0.01]), title('Diff PPG Finger 2 time');
        ax2 = subplot(3,1,2), plot(ppgF_time(1:end-1),diffppgF_I2_time,ppgF_time(1:(end-1):end),mean(diffppgF_I2_time)*[1 1],'r-'), ylim([0.001 0.01]);
        ax3 = subplot(3,1,3), plot(ppgF_time(1:end-1),diffppgF_G2_time,ppgF_time(1:(end-1):end),mean(diffppgF_G2_time)*[1 1],'r-'), ylim([0.001 0.01]);
        linkaxes([ax1 ax2 ax3],'x');
    end
    
  %  PPGS 3
    ppgF_3_mask = A(41,:) == 3;
    
    ppgF_3_tag_order = [bitand(A(42, ppgF_3_mask), 248)/2^3; bitand(A(45, ppgF_3_mask), 248)/2^3; bitand(A(48, ppgF_3_mask), 248)/2^3];
    ppgF_3_tag_order = reshape((ppgF_3_tag_order), 1, numel(ppgF_3_tag_order));
     
    ppgF_3_toggle = double(A(41, ppgF_3_mask));
    figure, plot(ppgF_3_toggle,'.'), title('PPG Finger 3 Toggle');
    
    ppgF_3_current_led1 = double(A(51,ppgF_3_mask))*0.24; %62mA range
    ppgF_3_current_led2 = double(A(52,ppgF_3_mask))*0.24;
    ppgF_3_current_led3 = double(A(53,ppgF_3_mask))*0.24;
    figure, subplot(3,1,1), plot(ppgF_3_current_led1,'.'), title('PPG Finger 3 Current LED1'), subplot(3,1,2), plot(ppgF_3_current_led2,'.'), title('PPG Finger 3 Current LED2'), subplot(3,1,3), plot(ppgF_3_current_led3,'.'), title('PPG Finger 3 Current LED3');
    
    ppgF_3 = [A(44:-1:42, ppgF_3_mask); zeros(1,size(A(44:-1:42, ppgF_3_mask),2)); A(47:-1:45, ppgF_3_mask); zeros(1,size(A(47:-1:45, ppgF_3_mask),2)); A(50:-1:48, ppgF_3_mask); zeros(1,size(A(50:-1:48, ppgF_3_mask),2));];
    ppgF_3(3, :) = bitand(ppgF_3(3, :), 7);
    ppgF_3(7, :) = bitand(ppgF_3(7, :), 7);
    ppgF_3(11, :) = bitand(ppgF_3(11, :), 7);
    ppgF_3 = double(typecast(ppgF_3(:), 'int32'));
    

    ppgF_I3 =  ppgF_3(ppgF_3_tag_order == 1);
    ppgF_R3 = ppgF_3(ppgF_3_tag_order == 2);
    ppgF_G3 =  ppgF_3(ppgF_3_tag_order == 3);
    
    if 0
        figure; plot(ppgF_time, ppgF_R3); hold on, plot(ppgF_time,ppgF_I3), plot(ppgF_time,ppgF_G3); title('PPG Finger 3'); legend({'Red PPG Finger','IR PPG Finger', 'Green PPG Finger'});
        diffppgF_R3_time = diff(ppgF_time); diffppgF_I3_time = diff(ppgF_time); diffppgF_G3_time = diff(ppgF_time);
        
        figure, ax1 = subplot(3,1,1), plot(ppgF_time(1:end-1),diffppgF_R3_time,ppgF_time(1:(end-1):end),mean(diffppgF_R3_time)*[1 1],'r-'), ylim([0.001 0.01]), title('Diff PPG Finger 3 time');
        ax2 = subplot(3,1,2), plot(ppgF_time(1:end-1),diffppgF_I3_time,ppgF_time(1:(end-1):end),mean(diffppgF_I3_time)*[1 1],'r-'), ylim([0.001 0.01]);
        ax3 = subplot(3,1,3), plot(ppgF_time(1:end-1),diffppgF_G3_time,ppgF_time(1:(end-1):end),mean(diffppgF_G3_time)*[1 1],'r-'), ylim([0.001 0.01]);
        linkaxes([ax1 ax2 ax3],'x');
    end
    %% Plots
    
    figure;
    subplot(3,1,1)
    plot(B_time,B_pres,'k-'); title('Pressure'); xlabel('Seconds'); ylabel('Pascals (Pa)'); xlim(B_time([1 end]));
    subplot(3,1,2)
    plot(B_time,B_temp,'r-'); title('Temperature'); xlabel('Seconds'); ylabel('\circC'); xlim(B_time([1 end]));
    subplot(3,1,3)
    plot(B_time,B_humi,'b-'); title('Relative Humidity'); xlabel('Seconds'); ylabel('%RH'); xlim(B_time([1 end]));
    
    if plotGraphs
        figure;
        subplot(3,1,1)
        plot(A_time,A_data_x,'b-',A_time,A_data_y,'r-',A_time,A_data_z,'k-'); xlim(A_time([1 end]));
        title('Acceleration'); xlabel('Seconds'); ylabel('g');
        legend ('accelX (g)', 'accelY (g)', 'accelZ (g)');
        
        %         subplot(3,1,2)
        %         plot(G_time,G_data_x,'b-',G_time,G_data_y,'r-',G_time,G_data_z,'k-'); xlim(G_time([1 end]));
        %         title('Gyroscope'); xlabel('Seconds'); ylabel('g');
        %         legend ('X', 'Y', 'Z');
        %Plot ECG
        subplot(3,1,3)
        
        
        plot(E_time(i_LON),E_data(i_LON),'b.', ...
            E_time(i_LOFF),E_data(i_LOFF),'r.',...
            'markers',3);
        title('ECG'); xlabel('Seconds'); ylabel('mV'); xlim(E_time([1 end]));
        legend ('ALL ON', 'LEAD OFF');
        
        figure;
        subplot(2, 1, 1);
        plot(A_time, A_time_2 - A_time, A_time(1:(end-1):end), mean(A_time_2 - A_time)*[1 1], 'r-');
        title('Sample Time');
        subplot(2, 1, 2);
        hist(A_time_2 - A_time,linspace(0,5e-3,25)); title('Hist Diff ECG time'); xlabel('Seconds'); ylabel('Count'); grid on;
        set(gca,'YScale','log')
        
        
        diffE_time = diff(E_time);
        diffA_time = diff(A_time);
        diffB_time = diff(B_time);
        diffG_time = diff(G_time);
        figure; % Compare to Seconds
        subplot(4,1,1)
        plot(E_time(1:end-1),diffE_time,E_time(1:(end-1):end),mean(diffE_time)*[1 1],'r-');
        title('Diff ECG time'); xlabel('Seconds'); ylabel('Seconds'); xlim(E_time([1 end]));
        subplot(4,1,2)
        plot(G_time(1:end-1),diffG_time,G_time(1:(end-1):end),mean(diffG_time)*[1 1],'r-');
        title('Diff GYRO time'); xlabel('Seconds'); ylabel('Seconds'); xlim(G_time([1 end]));
        subplot(4,1,3)
        plot(A_time(1:end-1),diffA_time,A_time(1:(end-1):end),mean(diffA_time)*[1 1],'r-');
        title('Diff ACCEL time'); xlabel('Seconds'); ylabel('Seconds'); xlim(A_time([1 end]));
        subplot(4,1,4)
        plot(B_time(1:end-1),diffB_time,B_time(1:(end-1):end),mean(diffB_time)*[1 1],'r-');
        title('Diff BME time'); xlabel('Seconds'); ylabel('Seconds'); xlim(B_time([1 end]));
        
        figure; % Compare to Samples
        subplot(4,1,1)
        plot(diff(E_time)); title('Diff ECG time'); xlabel('Sample'); ylabel('Seconds'); grid on;
        subplot(4, 1, 2);
        plot(diff(G_time)); title('Diff GYRO time'); xlabel('Sample'); ylabel('Seconds');
        subplot(4,1,3)
        plot(diff(A_time)); title('Diff ACCEL time'); xlabel('Sample'); ylabel('Seconds');
        subplot(4,1,4)
        plot(diff(B_time)); title('Diff BME time'); xlabel('Sample'); ylabel('Seconds');
        
        
        %% diff time Accel vs temp
        % figure;
        % plot(1:length(A_data_x),A_data_x,'b-',1:length(A_data_x),A_data_y,'r-',1:length(A_data_x),A_data_z,'k-');
        % title('Acceleration'); xlabel('Seconds'); ylabel('g');
        % legend ('accelX (g)', 'accelY (g)', 'accelZ (g)');
        figure;
        [ha, ~, ~] = plotyy(A_time(1:end-1),diff(A_time), B_time,B_temp);
        title('Diff ACCEL time and TEMP over time'); xlabel('Seconds');
        set(get(ha(1),'ylabel'),'string','Seconds');
        set(get(ha(2),'ylabel'),'string','\circC');
        set(ha(1),'xlim',A_time([1 end]));
        set(ha(2),'xlim',A_time([1 end]));
        
        %% Histograms of each diff Time
        figure; % Compare to Samples
        subplot(3,1,1)
        hist(diff(E_time),linspace(2e-4,2e-3,25)); title('Hist Diff ECG time'); xlabel('Seconds'); ylabel('Count'); grid on;
        set(gca,'YScale','log')
        subplot(3,1,2)
        hist(diff(A_time),linspace(2e-4,8e-3,25)); title('Hist Diff Accel time'); xlabel('Seconds'); ylabel('Count'); grid on;
        set(gca,'YScale','log')
        subplot(3,1,3)
        hist(diff(B_time),linspace(48e-3,52e-3,25)); title('Hist Diff BME time'); xlabel('Seconds'); ylabel('Count'); grid on;
        set(gca,'YScale','log')
    end
    
    %     save(strcat(directory, 'dataSet'), 'A_data_x', 'A_data_y', 'A_data_z', 'A_time',...
    %         'B_humi', 'B_pres', 'B_temp', 'E_data', 'E_time', 'G_data_x', 'G_data_y', 'G_data_z',...
    %         'G_time','ppg_G1_time', 'ppg_G1', 'ppg_G2_time', 'ppg_G2', 'ppg_G3_time', 'ppg_G3',...
    %         'ppg_R1_time', 'ppg_R1', 'ppg_R2_time', 'ppg_R2', 'ppg_R3_time', 'ppg_R3',...
    %         'ppg_I1_time', 'ppg_I1', 'ppg_I2_time', 'ppg_I2', 'ppg_I3_time', 'ppg_I3'...
    %         );
%     save(strcat(directory, 'dataSet'), 'A_data_x', 'A_data_y', 'A_data_z', 'A_time',...
%         'E_data', 'E_time');
    
end


%figure; plot(mod(E_time, 86400)/60/60, E_data) At_Home
% figure, plot((E_time-E_time(1))/60,E_data)
ECG = filterSig(E_data, 5, 10, 35, 30, 1000); figure, plot(ECG), title('Filtered ECG');
SCGy = filterSig(A_data_y, 0.8, 1, 30, 25, 1000); 
SCGz = filterSig(A_data_z, 1, 2, 30, 25, 125); figure, plot(SCGz), title('Filtered SCG');
PPG_r = filterSig(ppgS_R2, 0.7, 1, 10, 8, 200); figure, plot(PPG_r), title('Filtered PPG 2');
PPG_i = filterSig(ppgS_I2, 0.7, 1, 10, 8, 200); hold on, plot(PPG_i);
PPG_g = filterSig(ppgS_G2, 0.7, 1, 10, 8, 200); hold on, plot(PPG_g),
legend({'Red PPG','IR PPG', 'Green PPG'});


PPG1_i = filterSig(ppgS_I1, 0.7, 1, 10, 8, 200);
PPG2_i = filterSig(ppgS_I2, 0.7, 1, 10, 8, 200);
PPG3_i = filterSig(ppgS_I3, 0.7, 1, 10, 8, 200);
figure, ax1 = subplot(2,1,1), plot(-PPG1_i), hold on, plot(-PPG2_i), plot(-PPG3_i), title('Sternum IR PPG');

PPGf1_i = filterSig(ppgF_I1, 0.7, 1, 10, 8, 200);
PPGf2_i = filterSig(ppgF_I2, 0.7, 1, 10, 8, 200);
PPGf3_i = filterSig(ppgF_I3, 0.7, 1, 10, 8, 200);
ax2 = subplot(2,1,2), plot(-PPGf1_i), hold on, plot(-PPGf2_i), plot(-PPGf3_i), linkaxes([ax1 ax2], 'x'), title('Finger IR PPG');

PPG1_r = filterSig(ppgS_R1, 0.7, 1, 10, 8, 200);
PPG2_r = filterSig(ppgS_R2, 0.7, 1, 10, 8, 200);
PPG3_r = filterSig(ppgS_R3, 0.7, 1, 10, 8, 200);
figure, ax1 = subplot(2,1,1), plot(-PPG1_r), hold on, plot(-PPG2_r), plot(-PPG3_r), title('Sternum Red PPG');

PPGf1_r = filterSig(ppgF_R1, 0.7, 1, 10, 8, 200);
PPGf2_r = filterSig(ppgF_R2, 0.7, 1, 10, 8, 200);
PPGf3_r = filterSig(ppgF_R3, 0.7, 1, 10, 8, 200);
ax2 = subplot(2,1,2), plot(-PPGf1_r), hold on, plot(-PPGf2_r), plot(-PPGf3_r), linkaxes([ax1 ax2], 'x'), title('Finger Red PPG');

PPG1_g = filterSig(ppgS_G1, 0.7, 1, 10, 8, 200);
PPG2_g = filterSig(ppgS_G2, 0.7, 1, 10, 8, 200);
PPG3_g = filterSig(ppgS_G3, 0.7, 1, 10, 8, 200);
figure, ax1 = subplot(2,1,1), plot(-PPG1_g), hold on, plot(-PPG2_g), plot(-PPG3_g), title('Sternum Green PPG');

PPGf1_g = filterSig(ppgF_G1, 0.7, 1, 10, 8, 200);
PPGf2_g = filterSig(ppgF_G2, 0.7, 1, 10, 8, 200);
PPGf3_g = filterSig(ppgF_G3, 0.7, 1, 10, 8, 200);
ax2 = subplot(2,1,2), plot(-PPGf1_g), hold on, plot(-PPGf2_g), plot(-PPGf3_g), linkaxes([ax1 ax2], 'x'), title('Finger Green PPG');

    save(strcat(directory, 'patchdataSet'), 'A_data_x', 'A_data_y', 'A_data_z', 'A_time',...
        'E_data', 'E_time','ppgF_time','ppgF_I1','ppgF_I2','ppgF_I3','ppgF_R1','ppgF_R2','ppgF_R3',...
        'ppgF_G1','ppgF_G2','ppgF_G3','ppgS_time', 'ppgS_I1', 'ppgS_I2', 'ppgS_I3', 'ppgS_R1', 'ppgS_R2', 'ppgS_R3',...
        'ppgS_G1', 'ppgS_G2', 'ppgS_G3');
% % % files = dir([directory 'subdataSet*.mat']);
% % %
% % % a = load([files(1).folder '/' files(1).name]);
% % % for i = 2:size(files)
% % %     b = load([files(i).folder '/' files(i).name]);
% % %
% % %     vrs = fieldnames(a);
% % %
% % %     for k = 1:length(vrs)
% % %         a.(vrs{k}) = [a.(vrs{k}); b.(vrs{k})];
% % %     end
% % % end
% % %
% % % save([directory '/' 'dataSet.mat'] ,'-struct','a')
%1/(max(diff(btime))-min(diff(btime)))
% 1/(max(diff(p_time))-min(diff(p_time)))
