function [tCal, pCal, hCal, tempRaw, presRaw, humRaw] = extractBMEVals(BME_Cal, B)
    % pull all 9 bytes of Raw BME data make them uint32_t
    cmdResp = B(5,:); %response
    presMSB = [B(6,:); zeros(3,size(B,2))]; presMSB = typecast(presMSB(:), 'uint32'); presLSB = [B(7,:); zeros(3,size(B,2))]; presLSB = typecast(presLSB(:), 'uint32'); presXLSB = [bitshift(bitand(B(8,:),240),-4); zeros(3,size(B,2))]; presXLSB = typecast(presXLSB(:), 'uint32'); %4 msbits (7:4), bit (3:0) = 0
    tempMSB = [B(9,:); zeros(3,size(B,2))]; tempMSB = typecast(tempMSB(:), 'uint32'); tempLSB = [B(10,:); zeros(3,size(B,2))]; tempLSB = typecast(tempLSB(:), 'uint32'); tempXLSB = [bitshift(bitand(B(11,:),240),-4); zeros(3,size(B,2))]; tempXLSB = typecast(tempXLSB(:), 'uint32'); %4 msbits (7:4), bit (3:0) = 0
    humMSB = [B(12,:); zeros(3,size(B,2))]; humMSB = typecast(humMSB(:), 'uint32'); humLSB = [B(13,:); zeros(3,size(B,2))]; humLSB = typecast(humLSB(:), 'uint32'); 
    % combine into adc values
    presRaw = typecast( bitor( bitor( bitshift(presMSB, 12), bitshift(presLSB, 4), 'uint32'), presXLSB, 'uint32' ), 'int32');
    tempRaw = typecast( bitor( bitor( bitshift(tempMSB, 12), bitshift(tempLSB, 4), 'uint32'), tempXLSB, 'uint32' ), 'int32'); %tempRaw = typecast(tempRaw(:), 'int32');
    humRaw = double(typecast(bitor(bitshift(humMSB, 8), humLSB, 'uint32'), 'int32'));
    
    % pull all calibration values 
    T1 = typecast(BME_Cal(1:2),'uint16');  T2 = typecast(BME_Cal(3:4),'int16'); T3 = typecast(BME_Cal(5:6),'int16');
    P1 = typecast(BME_Cal(7:8),'uint16'); P2 = typecast(BME_Cal(9:10),'int16'); P3 = typecast(BME_Cal(11:12),'int16'); P4 = typecast(BME_Cal(13:14),'int16'); P5 = typecast(BME_Cal(15:16),'int16'); P6 = typecast(BME_Cal(17:18),'int16'); P7 = typecast(BME_Cal(19:20),'int16'); P8 = typecast(BME_Cal(21:22),'int16'); P9 = typecast(BME_Cal(23:24),'int16'); 
    H1 = BME_Cal(25); H2 = typecast(BME_Cal(26:27),'int16'); H3 = BME_Cal(28); H4 = typecast(BME_Cal(29:30),'int16'); H5 = typecast(BME_Cal(31:32),'int16'); H6 = typecast(BME_Cal(33),'int8');
    tCal = {T1; T2; T3};
    pCal = {P1; P2; P3; P4; P5; P6; P7; P8; P9};
    hCal = {H1; H2; H3; H4; H5; H6};
end

