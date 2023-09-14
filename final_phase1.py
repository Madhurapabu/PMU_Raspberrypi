import numpy as np
import serial
import websockets
import asyncio
from datetime import datetime
from datetime import timezone
import datetime
import math
np.set_printoptions(threshold=np.inf)

def Smart_DFT(fs, vi, ndata, base_frequency, n, shift):
    '''
    Smart DFT is the precise version of DFT in the case of frequency's variation can't be ignored,
    it use numerical solution to calculate the frequency , phase angel , and magnitude of the signal(voltage)
    that you give to.
    This function can use the vi array to calculate to 10 sets of frequency, angle, and magnitude of the signal(voltage)
    , base on the base_frequency.

    fs : sampling rate, fs = base_frequency*n, fs must be the multiples of base_frequency, or the frequency,
        phase angle ,and magnitude of the signal(voltage) you want to calculate will be wrong.
    vi : the samples of voltage.
    ndata : the number of data which every time we take from array vi in "for count" loop,
        ndata = fs/data_output_per_second.
    base_frequency : frequency base of country, area or you want it to be.
    n : the number of voltage data that is used to calculate to a DFT value.
    shift : shift points of vi array.
    '''
    m = ndata - n - 2
    v = np.array(vi[shift:ndata+shift])
    _Xn = 0
    _Xna = [None] * (ndata - N+1)
    
    for k in range (0,n):
        _Xn = _Xn + v[k] * np.exp(complex(0,-1) * 2 * np.pi / n * k)  # Discrete Fourier Transform(DFT)
    _Xna[0] = _Xn / n * 2  # every array which doing one time DFT will change to a complex number

    for r in range (2,(ndata - n+2)):
        # create many new Xna(DFT value)
        _Xn = (_Xn - v[r - 2]) * np.exp(complex(0, 1) * 2 * np.pi / n) + v[r + n - 2] * np.exp(
            complex(0, -1) * 2 * np.pi / n * (n - 1))
        _Xna[r - 1] = _Xn / n * 2
        if r > m + 2:
            # do only one time in "for r" loop
            #r - M = 3
            ##### if you want to use the Harmonic_analysis
            fr, _Ar = harmonic_analysis(_Xna, r, m, fs)
            ##### if you don't want to use the Harmonic_analysis
            # _X = (_Xna[0]+_Xna[2])/(2*_Xna[1])
            # o = np.arccos(_X.real)
            # a = np.exp(complex(0, 1) * o)
            # _Ar1 = (_Xna[1]*a - _Xna[0])/(a*a-1)
            # fr1 = o*fs/(2*np.pi)
            # pr1_abs = abs(_Ar1)*n*np.sin(np.pi*(fr1 - base_frequency)/fs)/np.sin(
            #    np.pi * (fr1 - base_frequency) / base_frequency) / np.sqrt(2)  # RMS value of voltage
            # pr1_theta = np.angle(_Ar1) - np.pi / (base_frequency * n) *\
            #    ((fr1 - base_frequency) * (n - 1))  # angle of voltage in radians
            # # #print (fr1, pr1_abs, pr1_theta)

    fr_abs = np.abs(fr)  # absolute value
    pr_abs = np.abs(_Ar) * n * np.sin(np.pi * (fr_abs - base_frequency) / (base_frequency * n)) / np.sin(
        np.pi * (fr_abs - base_frequency) / base_frequency) / np.sqrt(2)  # RMS value of voltage
    pr_theta = np.angle(_Ar) - np.pi / (base_frequency * n) * ((
        fr_abs - base_frequency) * (n - 1))  # angle of voltage in radians
    return fr_abs[0], pr_abs[0], pr_theta[0]
   
   
   
def harmonic_analysis(Xna, r, m, fs_):
    '''
    harmonic_analysis is being used in the condition that voltage data is received by Analog to Digital Converter,
    so the signal(voltage) data will combine with some noises or harmonics appearing in wire or in power system witch
    are much smaller than the main signal(smaller than 10% of main signal).

    Xna : the DFT values.
    r : count parameter, r = m+3.
    m : count parameter, m = r-3.
    '''
    Xna = np.array(Xna)
    x1 = Xna[r - m - 3]
    x2 = Xna[r - m - 2]
    _A1 = [None] * m  # the array of values of DFT_r+1 (X_hat_r+1)
    _B1 = [None] * m  # the array of values of DFT_r + DFT_r+2 (X_hat_r + X_hat_r+2)
    for k in range(0, m):
        _A1[k] = 2 * Xna[r-m+k-1]  # k=0 => r-M+k-1 = 2
        _B1[k] = Xna[r-m+k-2] + Xna[r-m+k]  # k=0 => r-M+k-2 = 1, r-M+k = 3

    _A1 = np.array([_A1])  # plus [] to let A1 be a column vector
    _B1 = np.array(_B1)
    _X = np.dot(_A1.conj(), _B1) / np.dot(_A1, _A1.conj().transpose())  # minimum variance of error
    # X will be "a" number ,not a matrix or array
    # A1.conj().transpose() is conjugate transpose
    if _X.real >= 1:
        # in python 1 may = 1.000000000000001 that arccos can't deal with it
        o = 0
    elif _X.real <= -1:
        # in python -1 may = -1.000000000000001 that arccos can't deal with it
        o = np.pi
    else:
        o = np.arccos(_X.real)

    a = np.exp(complex(0, 1) * o)  # if o = 0 => a =1

    return (o*fs_)/(2*np.pi), (x2*a-x1)/(a*a-1)  # frequency and Ar

def find_zero_crossing(data):
    zero_crossing = []

    for i in range(1, len(data)):
        if ((data[i-1] > 0 and data[i] <= 0) or (data[i-1] < 0 and data[i] >= 0)):
            zero_crossing.append(i)
    return zero_crossing

def find_zero_crossing_P_N(data):
    zero_crossing = []

    for i in range(1, len(data)):
        if ((data[i-1] > 0 and data[i] <= 0)):
            zero_crossing.append(i)
    return zero_crossing



################################ main code #####################################
fs = 200  # sampling rate, must > base_frequency * 2
theta = 20  # the angle of signal(voltage) you want to create(Degree)
frequency_siganl = 59  # the frequency of the signal(voltage) you create (Hz)
#vi = make_signal(fs, frequency_siganl, theta)  # making a wave using the parameter you give
#data_per_set = int(fs * 0.1)  # 10% of signal(voltage) data, must > fs/base_frequency
data_per_set = 20
base_frequency = 50  # frequency base of country , area or you want ot it be (Hz)
N = int(fs/base_frequency)  # N is the number of signal(voltage) data that is used to calculate a DFT value
shift = 0  # shift points of vi array

ser = serial.Serial('/dev/ttyACM0', 230400, timeout=1)  # Adjust the port if needed
ser.reset_input_buffer()

async def connect():
    uri = "ws://34.125.45.1:4000"
    async with websockets.connect(uri) as websocket:
        print("Connect to WS server")
        
        while True:
            message = await websocket.recv()
            print("Message from server:",message)
            
            vi_1 = []
            vi_2 = []
            vi_3 = []
            
            ci_1 = []
            ci_2 = []
            ci_3 = []
            
            time_stamp_arr = []
            
            raw_value = 0.00
            
            #Mag calculate parameters
            lowPassFilteredMagnitude_ph1 = 0.0
            highPassFilteredMagnitude_ph1 = 0.0 
            highPassAlpha = 0.5
            lowPassAlpha = 0.5
            prev = 0.0
            timestamp = 0.0
            prv_time = 0.0
            count = 0
            flag = 0
            flag_2 = 0
            mag_value = 0.0
            
            #Mag calculate parameters phase_2
            lowPassFilteredMagnitude_ph2 = 0.0
            highPassFilteredMagnitude_ph2 = 0.0 
            highPassAlpha_ph2 = 0.5
            lowPassAlpha_ph2 = 0.5
            prev_ph2 = 0.0
            timestamp = 0.0
            prv_time_ph2 = 0.0
            count_ph2 = 0
            flag_ph2 = 0
            mag_value_ph2 = 0.0
            
            #Mag calculate parameters phase_3
            lowPassFilteredMagnitude_ph3 = 0.0
            highPassFilteredMagnitude_ph3 = 0.0 
            highPassAlpha_ph3 = 0.5
            lowPassAlpha_ph3 = 0.5
            prev_ph3 = 0.0
            prv_time_ph3 = 0.0
            count_ph3 = 0
            flag_ph3 = 0
            mag_value_ph3 = 0.0
            
            time_dif = 0.0
            time_dif_ph2 = 0.0
########################################################################################################################            
            #Current Mag calculate parameters
            lowPassFilteredCurrent_ph1 = 0.0
            highPassFilteredCurrent_ph1 = 0.0 
            highPassAlpha_current = 0.1
            lowPassAlpha_current = 0.1
            prev_current = 0.0
            timestamp = 0.0
            prv_time_current = 0.0
            count_current = 0
            flag_current = 0
            flag_2 = 0
            mag_value_current = 0.0
            
            #Current Mag calculate parameters
            lowPassFilteredCurrent_ph2 = 0.0
            highPassFilteredCurrent_ph2 = 0.0 
            highPassAlpha_current_ph2 = 0.1
            lowPassAlpha_current_ph2 = 0.1
            prev_current_ph2 = 0.0
            prv_time_current_ph2 = 0.0
            count_current_ph2 = 0
            flag_current_ph2 = 0
            mag_value_current_ph2 = 0.0
            
            lowPassFilteredCurrent_ph3 = 0.0
            highPassFilteredCurrent_ph3 = 0.0 
            highPassAlpha_current_ph3 = 0.04
            lowPassAlpha_current_ph3 = 0.04
            prev_current_ph3 = 0.0
            prv_time_current_ph3 = 0.0
            count_current_ph3 = 0
            flag_current_ph3 = 0
            mag_value_current_ph3 = 0.0
            
            
            
            #Fre calculate parameters
            lowPassFilteredFreq = 0.0
            highPassFilteredFreq = 0.0 
            highPassAlpha_freq = 0.4
            lowPassAlpha_freq = 0.4
            prev_freq = 0.0
            timestamp_freq = 0.0
            prv_time_freq = 0.0
            count_freq = 0
            flag = 0
            flag_2 = 0
            time_gap = 1

            
            #Fre calculate parameters phase 2
            lowPassFilteredFreq_ph2 = 0.0
            highPassFilteredFreq_ph2 = 0.0 
            highPassAlpha_freq_ph2 = 0.4
            lowPassAlpha_freq_ph2 = 0.4
            prev_freq_ph2 = 0.0
            timestamp_freq_ph2 = 0.0
            prv_time_freq_ph2 = 0.0
            count_freq_ph2 = 0
            flag_ph2 = 0
            time_gap__ph2 = 1
            
            #Fre calculate parameters phase 3
            lowPassFilteredFreq_ph3 = 0.0
            highPassFilteredFreq_ph3 = 0.0 
            highPassAlpha_freq_ph3 = 0.04
            lowPassAlpha_freq_ph3 = 0.04
            prev_freq_ph3 = 0.0
            timestamp_freq_ph3 = 0.0
            prv_time_freq_ph3 = 0.0
            count_freq_ph3 = 0
            flag_ph3 = 0
            time_gap__ph3 = 1
            
            #phasor estimation parameter
            cos_fun = []
            clock_time = 0
            highPassFilteredPhase = 0.0 
            highPassAlpha_phase= 0.0001
            phase_val_new = 0.0
            
            highPassFilteredPhase_ph2 = 0.0 
            phase_val_new_ph2 = 0.0
            phase_val_new_ph3 = 0.0
            
            #ROOCOF Calculation
            calculate_freq = 0.0
            roocof = 0.0
            
            #ROOCOF Calculation phase 2
            calculate_freq_ph2 = 0.0
            roocof_ph2 = 0.0
            
            #ROOCOF Calculation phase 2
            calculate_freq_ph3 = 0.0
            roocof_ph3 = 0.0
            
            
            phase_old = 0.0
            phase_old_2 = 0.0 
            phase_old_3 = 0.0
            
            phase_old_current_1 = 0.0
            phase_old_current_2 = 0.0 
            phase_old_current_3 = 0.0
            
            
            while True:
                try:
                    data = ser.readline().decode().strip()  # Read data from Arduino
                    values = data.split(",")
                    if len(values) == 6:
                        vi_1.append(float(values[0]))
                        vi_2.append(float(values[1]))
                        vi_3.append(float(values[2]))
                        
                        ci_1.append(float(values[3]))
                        ci_2.append(float(values[4]))
                        ci_3.append(float(values[5]))
                        
                    dt_1 = datetime.datetime.now(timezone.utc)
                    utc_time_1 = dt_1.replace(tzinfo=timezone.utc)
                    time_stamp_new = utc_time_1.timestamp() + 19800
                    time_stamp_arr.append(time_stamp_new)
    
                    cos_fun.append(230 * np.cos(2*np.pi*50*time_stamp_new))
                    clock_time = clock_time + 0.002
                    
                except ValueError:
                    print("error")
                
                if len(vi_1) == (data_per_set/2):
                    dt = datetime.datetime.now(timezone.utc)
                    utc_time = dt.replace(tzinfo=timezone.utc)
                    timestamp = utc_time.timestamp() + 19800

                if ((len(ci_2) == (data_per_set)) and (len(ci_3) == data_per_set)):
                    freq_1, Vrms_1, theta_1 = Smart_DFT(fs, vi_1, data_per_set, base_frequency, N, shift) 
                    freq_2, Vrms_2, theta_2 = Smart_DFT(fs, vi_2, data_per_set, base_frequency, N, shift) 
                    freq_3, Vrms_3, theta_3 = Smart_DFT(fs, vi_3, data_per_set, base_frequency, N, shift)
                    
                    freq_c_1, Crms_1, theta_c1 = Smart_DFT(fs, ci_1, data_per_set, base_frequency, N, shift) 
                    freq_c_2, Crms_2, theta_c2 = Smart_DFT(fs, ci_1, data_per_set, base_frequency, N, shift)
                    freq_c_3, Crms_3, theta_c3 = Smart_DFT(fs, ci_1, data_per_set, base_frequency, N, shift)
                    
                    all_zero_crossings_indices_1 = find_zero_crossing(vi_1)
                    all_zero_crossings_indices_2 = find_zero_crossing(vi_2)
                    all_zero_crossings_indices_3 = find_zero_crossing(vi_3)   
                                        
                    all_zero_crossings_indices_PN = find_zero_crossing_P_N(vi_1)
                    all_zero_crossings_indices_ref = find_zero_crossing_P_N(cos_fun)  
                    
                    all_zero_crossings_indices_PN_ph2= find_zero_crossing_P_N(vi_2)
                    all_zero_crossings_indices_PN_ph3= find_zero_crossing_P_N(vi_3)
                    
                    # print(vi_1)
                    # print(cos_fun,"Ref signal")
                           
                    new_rms_ph1 = Vrms_1[0]
                    new_rms_ph2 = Vrms_2[0]
                    new_rms_ph3 = Vrms_3[0]
                    
                    new_crms_ph1 = Crms_1[0]
                    new_crms_ph2 = Crms_2[0]
                    new_crms_ph3 = Crms_3[0]
                    
                    
                    vi_1 = []
                    vi_2 = []
                    vi_3 = []
                    
                    cos_fun = []
                    ci_1 = []
                    ci_2 = []
                    ci_3 = []
                    
                    #Current Magnitude Value Calculation
                    lowPassFilteredCurrent_ph1 = lowPassAlpha_current * new_crms_ph1 + \
                                    (1 - lowPassAlpha_current) * lowPassFilteredCurrent_ph1
                                    
                    highPassFilteredCurrent_ph1 = highPassAlpha_current * \
                                    (lowPassFilteredCurrent_ph1 - highPassFilteredCurrent_ph1) + \
                                    highPassFilteredCurrent_ph1
                                    
                    mag_value_current = round((highPassFilteredCurrent_ph1), 3)

                    if (((highPassFilteredCurrent_ph1 - prev_current)/(timestamp - prv_time_current) < 0.25) and (count_current > 3))  :
                        highPassAlpha_current = 0.008
                        lowPassAlpha_current = 0.008
                        flag_current = 1
                    
                    if ((abs((highPassFilteredCurrent_ph1 - prev_current)/(timestamp - prv_time_current)) > 0.2) and (flag_current == 1)):
                        highPassAlpha_current = 0.1
                        lowPassAlpha_current = 0.1
                    count_current  = count_current + 1 
                    prev_current = highPassFilteredCurrent_ph1
                    prev_current = timestamp
                    
                    
                    ####Voltage mag calculation phase 1
                    lowPassFilteredMagnitude_ph1 = lowPassAlpha * new_rms_ph1 + \
                                    (1 - lowPassAlpha) * lowPassFilteredMagnitude_ph1
                                    
                    highPassFilteredMagnitude_ph1 = highPassAlpha * \
                                    (lowPassFilteredMagnitude_ph1 - highPassFilteredMagnitude_ph1) + \
                                    highPassFilteredMagnitude_ph1
                                    
                    mag_value = round((highPassFilteredMagnitude_ph1- 17.1), 3)

                    if (((highPassFilteredMagnitude_ph1 - prev)/(timestamp - prv_time) < 0.25) and (count > 3))  :
                        highPassAlpha = 0.008
                        lowPassAlpha = 0.008
                        flag = 1
                    
                    if ((abs((highPassFilteredMagnitude_ph1 - prev)/(timestamp - prv_time)) > 0.9) and (flag == 1)):
                        highPassAlpha = 0.1
                        lowPassAlpha = 0.1
                    count  = count + 1 
                    prev = highPassFilteredMagnitude_ph1
                    prv_time = timestamp
                    
                    #print(all_zero_crossings_indices_1)
                    #Frequency Calcualtion phase 1
                    if len(all_zero_crossings_indices_1) ==  6:
                        time_gap = time_stamp_arr[all_zero_crossings_indices_1[3]] - time_stamp_arr[all_zero_crossings_indices_1[1]]
                    
                    freq_new = 1/time_gap
                    
                    lowPassFilteredFreq = lowPassAlpha_freq * freq_new + \
                                    (1 - lowPassAlpha_freq) * lowPassFilteredFreq
                                    
                    highPassFilteredFreq = highPassAlpha_freq * \
                                    (lowPassFilteredFreq - highPassFilteredFreq) + \
                                    highPassFilteredFreq
                    
                    
                    freq_val = round((highPassFilteredFreq -25), 2)
                    
                    
                    if (((highPassFilteredFreq - prev_freq)/(timestamp - prv_time_freq) < 0.0009) and (count > 3))  :
                        highPassAlpha_freq = 0.0003
                        lowPassAlpha_freq = 0.0003
                        flag = 1
                    
                    if ((abs((highPassFilteredFreq - prev_freq)/(timestamp - prv_time_freq)) > 0.0005) and (flag == 1)):
                        highPassAlpha_freq = 0.009
                        lowPassAlpha_freq = 0.009
                    
                    #ROOCOF Calculation
                    roocof = (freq_val - calculate_freq) / (timestamp - prv_time_freq)
                    calculate_freq = freq_val  
                    
                    count_freq  = count_freq + 1
                    prev_freq = highPassFilteredFreq                 
                    prv_time_freq = timestamp
                    
            
#################################################################################################################################################################################################
                    #Magnitude Value Calculation_ph2
                    lowPassFilteredMagnitude_ph2 = lowPassAlpha_ph2 * new_rms_ph2 + (1 - lowPassAlpha_ph2) * lowPassFilteredMagnitude_ph2
                                    
                    highPassFilteredMagnitude_ph2 = highPassAlpha_ph2 * (lowPassFilteredMagnitude_ph2 - highPassFilteredMagnitude_ph2) + highPassFilteredMagnitude_ph2
                                    
                    mag_value_ph2 = round((highPassFilteredMagnitude_ph2 - 31), 3)

                    if (((highPassFilteredMagnitude_ph2 - prev_ph2)/(timestamp - prv_time_ph2) < 0.25) and (count_ph2 > 3))  :
                        highPassAlpha_ph2 = 0.008
                        lowPassAlpha_ph2 = 0.008
                        flag_ph2 = 1
                    
                    if ((abs((highPassFilteredMagnitude_ph2 - prev_ph2)/(timestamp - prv_time_ph2)) > 0.3) and (flag_ph2 == 1)):
                        highPassAlpha_ph2 = 0.1
                        lowPassAlpha_ph2 = 0.1
                    count_ph2  = count_ph2 + 1 
                    prev_ph2 = highPassFilteredMagnitude_ph2
                    prv_time_ph2 = timestamp
                    
                    
                    #Current Magnitude Value Calculation
                    lowPassFilteredCurrent_ph2 = lowPassAlpha_current_ph2 * new_crms_ph2 + \
                                    (1 - lowPassAlpha_current_ph2) * lowPassFilteredCurrent_ph2
                                    
                    highPassFilteredCurrent_ph2 = highPassAlpha_current_ph2 * \
                                    (lowPassFilteredCurrent_ph2 - highPassFilteredCurrent_ph2) + \
                                    highPassFilteredCurrent_ph2
                                    
                    mag_value_current_ph2 = round((highPassFilteredCurrent_ph2), 3)

                    if (((highPassFilteredCurrent_ph2 - prev_current_ph2)/(timestamp - prv_time_current_ph2) < 0.25) and (count_current_ph2 > 3))  :
                        highPassAlpha_current_ph2 = 0.008
                        lowPassAlpha_current_ph2 = 0.008
                        flag_current_ph2 = 1
                    
                    if ((abs((highPassFilteredCurrent_ph2 - prev_current_ph2)/(timestamp - prv_time_current_ph2)) > 0.2) and (flag_current_ph2 == 1)):
                        highPassAlpha_current_ph2 = 0.1
                        lowPassAlpha_current_ph2 = 0.1
                    count_current_ph2  = count_current_ph2 + 1 
                    prev_current_ph2 = highPassFilteredCurrent_ph2
                    prev_current_ph2 = timestamp
                    
                    
                    #Frequency Calcualtion phase 2
                    if len(all_zero_crossings_indices_2) ==  6:
                        time_gap__ph2 = time_stamp_arr[all_zero_crossings_indices_2[3]] - time_stamp_arr[all_zero_crossings_indices_2[1]]

                    freq_new_ph2 = 1/time_gap__ph2
                    
                    lowPassFilteredFreq_ph2 = lowPassAlpha_freq_ph2 * freq_new_ph2 + \
                                    (1 - lowPassAlpha_freq_ph2) * lowPassFilteredFreq_ph2
                                    
                    highPassFilteredFreq_ph2 = highPassAlpha_freq_ph2 * \
                                    (lowPassFilteredFreq_ph2 - highPassFilteredFreq_ph2) + \
                                    highPassFilteredFreq_ph2
                    
                    
                    freq_val_ph2 = round((highPassFilteredFreq_ph2-0.5), 2)
                    
                    
                    if (((highPassFilteredFreq_ph2 - prev_freq_ph2)/(timestamp - prv_time_freq_ph2) < 0.009) and (count_freq_ph2 > 3))  :
                        highPassAlpha_freq_ph2 = 0.0003
                        lowPassAlpha_freq_ph2 = 0.0003
                        flag_ph2 = 1
                    
                    if ((abs((highPassFilteredFreq_ph2 - prev_freq_ph2)/(timestamp - prv_time_freq_ph2)) > 0.0005) and (flag_ph2 == 1)):
                        highPassAlpha_freq_ph2 = 0.009
                        lowPassAlpha_freq_ph2 = 0.009
                    
                    #ROOCOF Calculation
                    roocof_ph2 = (freq_val_ph2 - calculate_freq_ph2) / (timestamp - prv_time_freq_ph2)
                    calculate_freq_ph2 = freq_val_ph2  
                    
                    count_freq_ph2  = count_freq_ph2 + 1
                    prev_freq_ph2 = highPassFilteredFreq_ph2                 
                    prv_time_freq_ph2 = timestamp
                    
                    #Phasor Calculation phase 2
                    if((len(all_zero_crossings_indices_PN_ph2) ==  3) and (len(all_zero_crossings_indices_ref)== 3)):
                        time_dif_ph2 = (time_stamp_arr[all_zero_crossings_indices_ref[0]] - time_stamp_arr[all_zero_crossings_indices_PN[0]])
                        time_dif_1 = (time_stamp_arr[all_zero_crossings_indices_ref[1]] - time_stamp_arr[all_zero_crossings_indices_PN[1]])
                    
                        avg = (time_dif+time_dif_1)/2

                    highPassFilteredPhase_ph2 = highPassAlpha_phase * (time_dif_ph2 - highPassFilteredPhase_ph2) + highPassFilteredPhase_ph2
         
                    phase_val_new_ph2 = (time_dif_ph2*360*50)
                    
                    if(phase_val_new >= 360):
                        phase_val_new_ph2 = phase_val_new_ph2 - 360
                            
                    if(phase_val_new <= -360):
                        phase_val_new_ph2 = phase_val_new_ph2 + 360

#################################################################################################################################################################################################
                    #Magnitude Value Calculation_ph3
                    lowPassFilteredMagnitude_ph3 = lowPassAlpha_ph3 * new_rms_ph3 + (1 - lowPassAlpha_ph3) * lowPassFilteredMagnitude_ph3
                                    
                    highPassFilteredMagnitude_ph3 = highPassAlpha_ph3 * (lowPassFilteredMagnitude_ph3 - highPassFilteredMagnitude_ph3) + highPassFilteredMagnitude_ph3
                                    
                    mag_value_ph3 = round((highPassFilteredMagnitude_ph3 - 74), 3)

                    if (((highPassFilteredMagnitude_ph3 - prev_ph3)/(timestamp - prv_time_ph3) < 1) and (count_ph3 > 3))  :
                        highPassAlpha_ph3 = 0.005
                        lowPassAlpha_ph3 = 0.005
                        flag_ph3 = 1
                    
                    if ((abs((highPassFilteredMagnitude_ph3 - prev_ph3)/(timestamp - prv_time_ph3)) > 0.3) and (flag_ph3 == 1)):
                        highPassAlpha_ph3 = 0.1
                        lowPassAlpha_ph3 = 0.1
                    count_ph3  = count_ph3 + 1 
                    prev_ph3 = highPassFilteredMagnitude_ph3
                    prv_time_ph3 = timestamp
                    
                    #Current Magnitude Value Calculation
                    lowPassFilteredCurrent_ph3 = lowPassAlpha_current_ph3 * new_crms_ph3 + \
                                    (1 - lowPassAlpha_current_ph3) * lowPassFilteredCurrent_ph3
                                    
                    highPassFilteredCurrent_ph3 = highPassAlpha_current_ph3 * \
                                    (lowPassFilteredCurrent_ph3 - highPassFilteredCurrent_ph3) + \
                                    highPassFilteredCurrent_ph3
                                    
                    mag_value_current_ph3 = round((highPassFilteredCurrent_ph3), 3)

                    if (((highPassFilteredCurrent_ph3 - prev_current_ph3)/(timestamp - prv_time_current_ph3) < 0.25) and (count_current_ph3 > 3))  :
                        highPassAlpha_current_ph3 = 0.008
                        lowPassAlpha_current_ph3 = 0.008
                        flag_current_ph3 = 1
                    
                    if ((abs((highPassFilteredCurrent_ph3 - prev_current_ph3)/(timestamp - prv_time_current_ph3)) > 0.2) and (flag_current_ph3 == 1)):
                        highPassAlpha_current_ph3 = 0.1
                        lowPassAlpha_current_ph3 = 0.1
                    count_current_ph3  = count_current_ph3 + 1 
                    prev_current_ph3 = highPassFilteredCurrent_ph3
                    prev_current_ph3 = timestamp
                    
                    
                    #Frequency Calcualtion
                    if len(all_zero_crossings_indices_3) ==  6:
                        time_gap__ph3 = time_stamp_arr[all_zero_crossings_indices_3[3]] - time_stamp_arr[all_zero_crossings_indices_3[1]]

                    freq_new_ph3 = 1/time_gap__ph3
                    
                    lowPassFilteredFreq_ph3 = lowPassAlpha_freq_ph3 * freq_new_ph3 + \
                                    (1 - lowPassAlpha_freq_ph3) * lowPassFilteredFreq_ph3
                                    
                    highPassFilteredFreq_ph3 = highPassAlpha_freq_ph3 * \
                                    (lowPassFilteredFreq_ph3 - highPassFilteredFreq_ph3) + \
                                    highPassFilteredFreq_ph3
                    
                    
                    freq_val_ph3 = round((highPassFilteredFreq_ph3 - 0.8), 2)
                    # print(freq_val_ph3)
                    
                    if (((highPassFilteredFreq_ph3 - prev_freq_ph3)/(timestamp - prv_time_freq_ph3) < 0.009) and (count_freq_ph3 > 3))  :
                        highPassAlpha_freq_ph3 = 0.0003
                        lowPassAlpha_freq_ph3 = 0.0003
                        flag_ph3 = 1
                    
                    if ((abs((highPassFilteredFreq_ph3 - prev_freq_ph3)/(timestamp - prv_time_freq_ph3)) > 0.0005) and (flag_ph3 == 1)):
                        highPassAlpha_freq_ph3 = 0.009
                        lowPassAlpha_freq_ph3 = 0.009
                    
                    #ROOCOF Calculation phase 3
                    roocof_ph3 = (freq_val_ph3 - calculate_freq_ph3) / (timestamp - prv_time_freq_ph3)
                    calculate_freq_ph3 = freq_val_ph3  
                    
                    count_freq_ph3  = count_freq_ph3 + 1
                    prev_freq_ph3 = highPassFilteredFreq_ph3                 
                    prv_time_freq_ph3 = timestamp
                    
                    #Phasor Calculation
                    # if((len(all_zero_crossings_indices_PN) ==  2) and (len(all_zero_crossings_indices_ref)== 2)):
                    #     time_dif = (time_stamp_arr[all_zero_crossings_indices_ref[0]] - time_stamp_arr[all_zero_crossings_indices_PN[0]])
                    #     time_dif_1 = (time_stamp_arr[all_zero_crossings_indices_ref[1]] - time_stamp_arr[all_zero_crossings_indices_PN[1]])
                    
                    #     avg = (time_dif+time_dif_1)/2

                    # highPassFilteredPhase = highPassAlpha_phase * (time_dif - highPassFilteredPhase) + highPassFilteredPhase
         
                    # phase_val_new = (highPassFilteredPhase*360*50)
                    
                    # if(phase_val_new >= 360):
                    #     phase_val_new = phase_val_new - 360
                            
                    # if(phase_val_new <= -360):
                    #     phase_val_new = phase_val_new + 360
                    phase_val_new = np.rad2deg((theta_3[0]) - theta_1[0])
                    phase_val_new_ph2 = np.rad2deg((theta_1[0]) - theta_2[0]) + 120
                    phase_val_new_ph3 = np.rad2deg((theta_2[0]) - theta_3[0]) - 120
                    
                    # phase_val_Current_new_ph1 = np.rad2deg((theta_c3[0]) - theta_c1[0])
                    # phase_val_Current_new_ph2 = np.rad2deg((theta_c1[0]) - theta_c2[0]) 
                    # phase_val_Current_new_ph3 = np.rad2deg((theta_c2[0]) - theta_c3[0]) 
                    
                    phase_val_Current_new_ph1 = np.rad2deg(theta_c1[0])
                    phase_val_Current_new_ph2 = np.rad2deg(theta_c2[0])
                    phase_val_Current_new_ph3 = np.rad2deg(theta_c3[0])
                    
                    print(phase_val_Current_new_ph1,",",phase_val_Current_new_ph2,",",phase_val_Current_new_ph3)
                    
                    if (phase_val_new > 200):
                        phase_val_new = phase_old
                    if phase_val_new_ph2 > 300:
                        phase_val_new_ph2 = phase_old_2
                    if phase_val_new_ph3 < -200:
                        phase_val_new_ph3 = phase_old_3
                    
                    
                    highPassFilteredPhase_ph2 = 0.0001 * phase_val_new_ph2 + (1 - highPassFilteredPhase_ph2) * highPassFilteredPhase_ph2
                    
                    phase_old = phase_val_new
                    phase_old_2 = phase_val_new_ph2
                    phase_old_3 = phase_val_new_ph3

                    time_stamp_arr = []
                    combined_message = f"v1,{mag_value},{mag_value_ph2},{mag_value_ph3},{phase_val_new},{phase_val_new_ph2},{phase_val_new_ph3},{freq_val},{freq_val_ph2},{freq_val_ph3},{roocof},{roocof_ph2},{roocof_ph3},{mag_value_current},{mag_value_current_ph2},{mag_value_current_ph3},{2.3},{2.4},{2.5},{timestamp}"
                    #print(mag_value_current)
  

                        
                    
                    # print(np.rad2deg(theta_1[0]),",",np.rad2deg(theta_3[0]),",",np.rad2deg((theta_1[0])-theta_3[0]),",",phase_val_new)
      
                          
                    await websocket.send(combined_message)
asyncio.get_event_loop().run_until_complete(connect())