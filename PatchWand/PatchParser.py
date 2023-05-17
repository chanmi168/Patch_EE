import numpy as np
import sys
import time
import struct
import datetime


# TODO: need better documentation

# [1. get sensitivty of the PD for 950nm] TBD
# 1uA reverse light current corresponds to 31.45mW/cm2 for 950nm light on Figure 3 of the VEMD8080 datasheet
# To get the sensitivity, divide the curret by the irradiance. In other words,
# R950 A/W = 
# 1uA / ( 31.48 irradiance (mW/cm2) * 4.5mm2 ) = 
# 1uA * (1e-6A/1uA) / ( 31.48mW/cm2 * (1e-3W/mW) * (1e-2cm2/1mm2) * 4.5mm2 )
R_950 = 1 * 1e-6 / (31.48 * 1e-3 * 1e-2 * 4.5) # Edited 12/29/2021: changed from 4.55 (mistake) to 4.5
# = 0.0007059155725 A/W (= uA/uW)

# [1. get sensitivty of the PD for 950nm] # edited on 3/13/2023
# 1uA reverse light current corresponds to 31.45uW/cm2 for 950nm light on Figure 3 of the VEMD8080 datasheet
# To get the sensitivity, divide the curret by the irradiance. In other words,
# uA to uW
# Y uA * [(31.48 uW/cm2 / uA) x (1e-2 cm2/mm2) x 4.5mm2]
# = Y uA / 1/[(31.48 uW/cm2 / uA) x (1e-2 cm2/mm2) x 4.5mm2]
# = Y uA / R_950
# => R_950 = 1/[31.48 x 1e-2 x 4.5] = 0.7059 uA/uW
# R_950 = 1 / (31.48 * 1e-2 * 4.5) # Edited 3/13/2023



# [2. Now we know the sensitivity of 950nm light. We can use the following relative values to get sensitivty in other wavelegnths]
# these relative values are found from Figure 5 of the VEMD8080 datasheet
# 950nm is not 1 beacuse the highest sensitivity is at ~840nm
R_lamda_relative = {
    'green': 0.622, # 526nm
    'red': 0.828, # 660nm
    'infrared': 0.825, # 950 nm
}

# [3. Store the sensitivity of different wavelengths in a dictionary. Again, the unit is in A/W]
R_lamda = {} # the unit is A/W
for key in R_lamda_relative:
    R_lamda[key] = R_lamda_relative[key] * R_950 / R_lamda_relative['infrared']

def PPGunit_conversion(ppg, firmware_id, wavelength):
    """
    PPGunit_conversion converts ppg (of a wavelength) data exported from the patch to the right unit (microWatt)

    ppg: the ppg data from the patch
    firmware_id: different firmwares have different PD bias (in the SpO2 study, we use firmware_id=1)

    wavelength: wavelength of the ppg
    return: ppg_uW (ppg data in MicroWatt)
    """ 
    
    # the latest patch firmware does not have c_PDbias (firmware_id=0)
    if firmware_id == 0:
        c_PDbias = 8 # uA
    else:
        c_PDbias = 0 # uA
    
    c_bits2uA = 61e-6 # uA/bits
    
    ppg_uA = ((ppg*c_bits2uA)-c_PDbias) # uA
    ppg_uW = ppg_uA / R_lamda[wavelength] # uW = uA / (A/W)
    return ppg_uW

# TBD:  12/12/2021
# def PPGunit_conversion(ppg, wavelength, c_PDbias):

#     c_bits2uA = 61e-6 # uA/bits
#     ppg_uA = ppg*c_bits2uA-c_PDbias/1000 # (b/c nA = 1e-3 uA)
#     ppg_uW = ppg_uA / R_lamda[wavelength] # uW = uA / (A/W)

#     return ppg_uW

# Edited: 12/12/2021, confirmed with Venu
def ECGunit_conversion(ecg):
    """
    ECGunit_conversion converts ecg data exported from the patch to the right unit (mV)

    ecg: the ecg data from the patch
    return: ecg (ecg data in mV)
    Note: see pg 28 of ads1291 dataset for reference of data format
    """ 

    ecg_gain = 8
    ecg_vRef = 2.42

    ecg_volts_per_bit = (ecg_vRef/ecg_gain)/(2**23-1) # (V/bit)
    # Convert to mV
    ecg = (ecg * ecg_volts_per_bit)*1000 # mV
    return ecg

# Edited: 12/12/2021, confirmed with Venu
def ACCunit_conversion(acc):
    """
    ACCunit_conversion converts acc data exported from the patch to the right unit (g)

    acc: the acc data from the patch
    return: acc (acc data in g)
    Note: see pg 4 of adxl355  dataset for reference of data format (find sensitiivity)
    256000 has the unit of LSB/g
    """ 
    acc = acc / 256000 # g
    return acc


def bytetoint(byte):
    if byte > 127:
        return byte - 256
    else:
        return byte
    
def bytestoint(byte):
    data = byte[0]*256 + byte[1]
    if data > 32767:
        return data - 65536
    else:
        return data
   
    #return int.from_bytes(byte + b'\x00\x00', "big",signed=True) >> 16

def parse(data, header = True, print_updates = False, isConvUnits = True):
    """
    This is the parsing script provided by Cardiosense 
    Author: Michael Chan (mchan81@gatech.edu)
    Cardiosense rep: Venu Ganti (venu@cardiosense.com)
    
    TODO: This code is still not very well organized. Please package this script better.
    Versions 1.1.0
    
    Input:
        fileContent - bytes read from the file
        header - bytes read from the file
        print_updates - print parsing information (e.g., speed, data size, etc.)
        isConvUnits - if True, converts acceleration binary code to accleration units (g)

    Usage:
        file = open(fileName, mode='rb') # open the .bin file exported from the patch using the Cardiosense software
        fileContent = file.read() # specified number of bytes from the file
        patch_dict = parse(fileContent, header = True, print_updates = True, isConvUnits = True)

    Release Notes:
        Version 1.0.0 (2022): Initial release
        Version 1.1.0 (2023/02/25): Fixed an issue, Red and Green PPG are swapped. In your own dataset, please check the perfusion index (PI) of your subjects during baseline. For your reference: Range of PI for red ≈ [0.05, 0.07]. Range of PI for green ≈ [0.56, 0.6]. Range of PI for infrared ≈ [0.08, 0.10]. All of the changes can be found by searching the key word "Edited on 2023/02/25"
    """

    time1 = time.time()

    AST_SR = 16384/(2*2)
       
    ts = int.from_bytes(data[50:58], "big")
    dt = datetime.datetime(1601, 1, 1, 0, 0, 0) + datetime.timedelta(seconds = ts/1e7)
    dt.ctime()

    if(header and not(dt < datetime.datetime.utcnow() < dt + datetime.timedelta(hours = 3*365*24))):
        data = data[512:]
       
        ts = int.from_bytes(data[50:58], "big") # try with offset
        dt = datetime.datetime(1601, 1, 1, 0, 0, 0) + datetime.timedelta(seconds = ts/1e7)
        dt.ctime()

   
    accel_size = int.from_bytes(data[58:62], "little")
    ppg_size = int.from_bytes(data[62:66], "little")
    ecg_size = int.from_bytes(data[66:70], "little")
    env_size = int.from_bytes(data[70:74], "little")
    ast_time = int.from_bytes(data[85:89], "little")
   
    devName = data[74:84].decode('latin1').split("\x00")[0]
   
    ecg = np.zeros(ecg_size) #[None] * 10000000
    ecg_time = np.zeros(ecg_size) #[None] * 10000000
    ecg_count = 0;

    accel_x = np.zeros(accel_size) #[None] * 10000000
    accel_y = np.zeros(accel_size) #[None] * 10000000
    accel_z = np.zeros(accel_size) #[None] * 10000000
    accel_time = np.zeros(accel_size) #[None] * 10000000
    accel_count = 0;

    ppg_ir_1_tag = np.zeros(ppg_size)
    ppg_g_1 = np.zeros(ppg_size) #[None] * 10000000
    ppg_r_1 = np.zeros(ppg_size) #[None] * 10000000
    ppg_ir_1 = np.zeros(ppg_size) #[None] * 10000000
    ppg_g_1_current = np.zeros(ppg_size)
    ppg_r_1_current = np.zeros(ppg_size)
    ppg_ir_1_current = np.zeros(ppg_size)
    ppg_time = np.zeros(ppg_size) #[None] * 10000000
    ppg_count = 0
   
    ppg_g_2 = np.zeros(ppg_size) #[None] * 10000000
    ppg_r_2 = np.zeros(ppg_size) #[None] * 10000000
    ppg_ir_2 = np.zeros(ppg_size) #[None] * 10000000
    ppg_g_2_current = np.zeros(ppg_size)
    ppg_r_2_current = np.zeros(ppg_size)
    ppg_ir_2_current = np.zeros(ppg_size)
   
    temp_skin = np.zeros(env_size) #[None] * 10000000
    pres = np.zeros(env_size)
    temp_internal = np.zeros(env_size)
    env_time = np.zeros(env_size)
   
    if print_updates:
        sys.stdout.write("Device name: " + devName + "\n")
        sys.stdout.write("Last sync time: " + dt.ctime() +" UTC\n")
        sys.stdout.write("Accel Size: " + str(accel_size) +"\n")
        sys.stdout.write("PPG Size: " + str(ppg_size) +"\n")
        sys.stdout.write("ECG Size: " + str(ecg_size) +"\n")
        sys.stdout.write("Environmental Size: " + str(env_size) +"\n")
        sys.stdout.write("AST Time Length: " + str(ast_time/AST_SR) +"\n")
   
    dataSize = len(data) - 512
    ind = 512
    indCount = 1000000;
    indStep = 1000000;
    time_temp = 0;
    compressCount = 0;
    ecgCompressCount = 0;
    accelCompressCount = 0;
    astCompressCount = 0;
    envCount = 0
    while accel_count < accel_size - 80:  
        flag = data[ind]
        flag_ind = ind
        ind += 1
        if flag & 0x80:      
            time_temp += data[ind]/AST_SR
            ind += 1
            astCompressCount += 1
        else:
            time_temp = int.from_bytes(data[ind:ind+4], "little")/AST_SR
            ind += 4
        if flag & 0x01:
            if flag & 0x10:
                ecg[ecg_count] = ecg[ecg_count-1] + bytetoint(data[ind+1])
                ind += 2
                compressCount += 2
                ecgCompressCount += 1
            else:
                ecg[ecg_count] = int.from_bytes(b'\x00' + data[ind+1:ind+4], "little",signed=True) >> 8
                ind += 4
            ecg_time[ecg_count] = time_temp
            ecg_count += 1
        if flag & 0x02:
            if flag & 0x40:
                accel_x[accel_count] = accel_x[accel_count-1] + bytetoint(data[ind])
                ind += 1
                accel_y[accel_count] = accel_y[accel_count-1] + bytetoint(data[ind])
                ind += 1
                accel_z[accel_count] = accel_z[accel_count-1] + bytetoint(data[ind])
                ind += 1
                accelCompressCount += 2
                compressCount += 6
            elif flag & 0x20:
                accel_x[accel_count] = accel_x[accel_count-1] + bytestoint(data[ind:ind+2])
                ind += 2
                accel_y[accel_count] = accel_y[accel_count-1] + bytestoint(data[ind:ind+2])
                ind += 2
                accel_z[accel_count] = accel_z[accel_count-1] + bytestoint(data[ind:ind+2])
                ind += 2
                accelCompressCount += 1
                compressCount += 3
            else:
                accel_x[accel_count] = int.from_bytes(b'\x00' + data[ind:ind+3], "little",signed=True) >> 8
                ind += 3
                accel_y[accel_count] = int.from_bytes(b'\x00' + data[ind:ind+3], "little",signed=True) >> 8
                ind += 3
                accel_z[accel_count] = int.from_bytes(b'\x00' + data[ind:ind+3], "little",signed=True) >> 8
                ind += 3
            accel_time[accel_count] = time_temp
            accel_count += 1
        if flag & 0x04:
            ppg_ir_1[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            if ppg_ir_1[ppg_count] > 524288:
                ppg_ir_1[ppg_count] -= 1048576
            ind += 3
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_r_1[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            if ppg_r_1[ppg_count] > 524288:
                ppg_r_1[ppg_count] -= 1048576
            ind += 3
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_g_1[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            if ppg_g_1[ppg_count] > 524288:
                ppg_g_1[ppg_count] -= 1048576
            # ind += 3
            # ppg_g_1[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            # if ppg_g_1[ppg_count] > 524288:
            #     ppg_g_1[ppg_count] -= 1048576
            # ind += 3
            # ppg_r_1[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            # if ppg_r_1[ppg_count] > 524288:
            #     ppg_r_1[ppg_count] -= 1048576
                
                
            ind += 3
            ppg_ir_2[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            if ppg_ir_2[ppg_count] > 524288:
                ppg_ir_2[ppg_count] -= 1048576
            ind += 3
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_r_2[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            if ppg_r_2[ppg_count] > 524288:
                ppg_r_2[ppg_count] -= 1048576
            ind += 3
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_g_2[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            if ppg_g_2[ppg_count] > 524288:
                ppg_g_2[ppg_count] -= 1048576
            # ind += 3
            # ppg_g_2[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            # if ppg_g_2[ppg_count] > 524288:
            #     ppg_g_2[ppg_count] -= 1048576
            # ind += 3
            # ppg_r_2[ppg_count] = int.from_bytes(bytes([data[ind] & 0x0F])+data[ind+1:ind+3], "big")
            # if ppg_r_2[ppg_count] > 524288:
            #     ppg_r_2[ppg_count] -= 1048576
            ind += 3
            ppg_time[ppg_count] = time_temp
            ppg_ir_1_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            ind += 1
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_r_1_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            # ppg_g_1_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            ind += 1
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_g_1_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            # ppg_r_1_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            ind += 1
            ppg_ir_2_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            ind += 1
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_r_2_current[ppg_count] = data[ind]
            # ppg_g_2_current[ppg_count] = data[ind]
            ind += 1
            # Red and Green were incorrectly labeled. Edited on 2023/02/25
            ppg_g_2_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            # ppg_r_2_current[ppg_count] = int.from_bytes(data[ind:ind+1], "little")
            ind += 1
            ppg_count += 1
        if flag & 0x08:
            temp_skin[envCount] = int.from_bytes(data[ind:ind+2], "big") * 0.005
            env_time[envCount] = time_temp
            ind += 2
            pres[envCount] = int.from_bytes(data[ind:ind+3], "little") / 4096
            ind += 3
            temp_internal[envCount] = bytestoint(data[ind+1:ind-1:-1]) / 100
            ind += 2
            envCount += 1
       
        if print_updates:
            if ind > indCount:
                indCount = indCount + indStep
                sys.stdout.write(str(ind) + " " + str(round(ind/len(data)*100, 2)) + "% done                      \r")
   
    ppg_time = ppg_time[0:ppg_count]
    ppg_g_1 = ppg_g_1[0:ppg_count]
    ppg_g_2 = ppg_g_2[0:ppg_count]
    ppg_r_1 = ppg_r_1[0:ppg_count]
    ppg_r_2 = ppg_r_2[0:ppg_count]
    ppg_ir_1 = ppg_ir_1[0:ppg_count]
    ppg_ir_2 = ppg_ir_2[0:ppg_count]

    # convert binary code to ppg units (uW)
    if (isConvUnits):
        ppg_g_1_current = ppg_g_1_current[0:ppg_count]
        ppg_g_2_current = ppg_g_2_current[0:ppg_count]
        ppg_r_1_current = ppg_r_1_current[0:ppg_count]
        ppg_r_2_current = ppg_r_2_current[0:ppg_count]
        ppg_ir_1_current = ppg_ir_1_current[0:ppg_count]
        ppg_ir_2_current = ppg_ir_2_current[0:ppg_count]

        # Edited: 12/12/2021, confirmed with Venu
        firmware_id = 1 # 1 for the latest version of the patch
        ppg_g_1 = PPGunit_conversion(ppg_g_1, firmware_id, 'green')
        ppg_g_2 = PPGunit_conversion(ppg_g_2, firmware_id, 'green')
        ppg_r_1 = PPGunit_conversion(ppg_r_1, firmware_id, 'red')
        ppg_r_2 = PPGunit_conversion(ppg_r_2, firmware_id, 'red')
        ppg_ir_1 = PPGunit_conversion(ppg_ir_1, firmware_id, 'infrared')
        ppg_ir_2 = PPGunit_conversion(ppg_ir_2, firmware_id, 'infrared')

   
    accel_x = accel_x[0:accel_count]
    accel_y = accel_y[0:accel_count]
    accel_z = accel_z[0:accel_count]
    accel_time = accel_time[0:accel_count]

    # convert binary code to accleration units (g)
    if (isConvUnits):
        accel_x = ACCunit_conversion(accel_x)
        accel_y = ACCunit_conversion(accel_y)
        accel_z = ACCunit_conversion(accel_z)
   
    ecg = ecg[0:ecg_count]
    ecg_time = ecg_time[0:ecg_count]

    # convert binary code to ecg units (V)
    if (isConvUnits):
        ecg = ECGunit_conversion(ecg)

    env_time = env_time[0:envCount]
    temp_skin = temp_skin[0:envCount]
    temp_internal = temp_internal[0:envCount]
    pres = pres[0:envCount]

    if print_updates:
        sys.stdout.write(str(ind) + " " + '%.1f' % round(ind/len(data)*100, 1) + "% done\n")
       
        sys.stdout.write("\nSTATS:")
        sys.stdout.write("\nMeasurement Length: " + str(round(accel_time[-1] - accel_time[0], 2)) + " seconds")
        sys.stdout.write("\nFile Size: " + str(dataSize/1000) + " kbytes")
        sys.stdout.write("\nAverage data rate: " + str(round(dataSize/(accel_time[-1] - accel_time[0]), 2)) + " bytes/second")
        sys.stdout.write("\nTheoretical uncompressed data rate: " + str(round((ppg_size*18 + accel_size*4 + accel_size*10 + ecg_size*4)/(accel_time[-1] - accel_time[0]), 2)) + " bytes/second")
        sys.stdout.write("\nPPG is " + str(round(100 * ppg_size*18/dataSize, 2)) + "% of the data")
        sys.stdout.write("\nECG is " + str(round(100 * (ecg_size*4-ecgCompressCount*2)/dataSize, 2)) + "% of the data")
        sys.stdout.write("\nAccel is " + str(round(100 * (accel_size*9-accelCompressCount*3)/dataSize, 2)) + "% of the data")
        sys.stdout.write("\nEnvironmental is " + str(round(100 * (env_size*7)/dataSize, 2)) + "% of the data")
        sys.stdout.write("\nAST is " + str(round(100 * (accel_size*4-astCompressCount*3)/dataSize, 2)) + "% of the data")
        sys.stdout.write("\nData Flag is " + str(round(100 * (accel_size)/dataSize, 2)) + "% of the data")
        sys.stdout.write("\ntime ellapsed: " + '%.3f' % (time.time() - time1) + " seconds\n")
        sys.stdout.write("parse speed: " + '%.3f' % (dataSize/(time.time() - time1)) + " bytes/second\n")
        sys.stdout.write("parse speed: " + '%.3f' % ((accel_time[-1] - accel_time[0])/(time.time() - time1)) + " dataSeconds/second\n")
    return {"ecg_time": ecg_time, "ecg": ecg,
        "accel_time": accel_time, "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
        "ppg_time": ppg_time, "ppg_g_1": ppg_g_1, "ppg_r_1": ppg_r_1, "ppg_ir_1": ppg_ir_1,
        "ppg_g_2": ppg_g_2, "ppg_r_2": ppg_r_2, "ppg_ir_2": ppg_ir_2,
        "ppg_g_1_current": ppg_g_1_current, "ppg_r_1_current": ppg_r_1_current, "ppg_ir_1_current": ppg_ir_1_current,
        "ppg_g_2_current": ppg_g_2_current, "ppg_r_2_current": ppg_r_2_current, "ppg_ir_2_current": ppg_ir_2_current,
        "temp_skin": temp_skin, "temp_internal": temp_internal, "pres": pres, "env_time": env_time}

# Edited: 12/12/2021, confirmed with Venu
# unit of the patch data if unit conversion code has been used
unit_dict = {
    'accel': 'g',
    'ppg': 'uW',
    'temp': 'Celsius',
    'pres': 'mbar', # or hPa
    'ecg': 'mV',    
}