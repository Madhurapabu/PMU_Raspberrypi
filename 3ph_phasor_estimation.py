
import asyncio
import websockets
import math
import time
import serial
import datetime
import ntplib

numSamples = 20


async def applyHammingWindow_ph1(data):
    for i in range(numSamples):
      
        window = 0.56 - 0.42 * math.cos(2 * math.pi * i / (numSamples - 1))
        data[i] *= window


async def applyHammingWindow_ph2(data):
    for i in range(numSamples):
        window = 0.56 - 0.42 * math.cos(2 * math.pi * i / (numSamples - 1))
        data[i] *= window


async def applyHammingWindow_ph3(data):
    for i in range(numSamples):
        window = 1.5 - 0.2 * math.cos(2 * math.pi * i / (numSamples - 1))
        data[i] *= window


def get_ntp_time():
    try:
        # Connect to NTP server
        ntp_client = ntplib.NTPClient()
        response = ntp_client.request('pool.ntp.org', version=3, timeout=5)

        # Extract and return the timestamp
        ntp_time = datetime.datetime.fromtimestamp(response.tx_time)
        return ntp_time

    except (ntplib.NTPException, ConnectionError) as e:
        print('Error retrieving NTP time:', e)


if __name__ == '__main__':
    start_time = time.time()  # Record the start time

    ser = serial.Serial('/dev/ttyACM0', 230400, timeout=1)
    ser.reset_input_buffer()

    async def connect():
        uri = "ws://34.125.45.1:4000"
        async with websockets.connect(uri) as websocket:
            print("Connected to WS Server")

            while True:
                message = await websocket.recv()
                print("Message from server:", message)

                numSamples = 20
                prev_value = 0

                lowPassAlpha = 0.1
                lowPassFilteredMagnitude_ph1 = 0.00
                lowPassFilteredMagnitude_ph2 = 0.0
                lowPassFilteredMagnitude_ph3 = 0.0

                highPassAlpha = 0.1
                highPassFilteredMagnitude_ph1 = 0.00
                highPassFilteredMagnitude_ph2 = 0.0
                highPassFilteredMagnitude_ph3 = 0.0

                rawValue_ph1 = 0.0
                rawValue_ph2 = 0.0
                rawValue_ph3 = 0.0
                
                
                prev_value_ph1 = 0.0
                prev_value_ph2 = 0.0
                prev_value_ph3 = 0.0

                frequency = 0.0
                frequency_2 = 0.0
                frequency_3 = 0.0

                f_time_3 = 0.0
                
                count_3 = 0
                count = 0
                count_3 = 0

                lowPassAlpha_fre = 0.001
                lowPassFrq = 0.0
                lowPassFrq_3 = 0.0

                highPassAlpha_freq = 0.01
                highPassFrq = 0.0
                highPassFrq_3 = 0.0
                
                prev_value_ph1= 0.0
                prev_value_ph2= 0.0
                prev_value_ph3= 0.0

                f_time = 0.0
                f_time_3 = 0.0
                
                prv_lst = 0.0
    
                # Sending a message from the client
                while True:
                    realPart_ph1 = [0.0] * numSamples
                    imagPart_ph1 = [0.0] * numSamples

                    realPart_ph2 = [0.0] * numSamples
                    imagPart_ph2 = [0.0] * numSamples

                    realPart_ph3 = [0.0] * numSamples
                    imagPart_ph3 = [0.0] * numSamples

                    for i in range(numSamples):
                        try:
                            # Read data from Arduino
                            line = ser.readline().decode().strip()
                            # print(line)
                            if line.startswith("Ph1:"):
                                rawValue_ph1 = float(line.split(":")[1])
                            elif line.startswith("Ph2:"):
                                rawValue_ph2 = float(line.split(":")[1])
                                # print(rawValue_ph2)
                            elif line.startswith("Ph3:"):
                                rawValue_ph3 = float(line.split(":")[1])
                            elif line.startswith("Data2:"):
                                frequency = line.split(":")[1]
                            elif line.startswith("Data3:"):
                                frequency_2 = line.split(":")[1]
                                #print(frequency_2)
                            elif line.startswith("Data4:"):
                                frequency_3 = line.split(":")[1]
                                
                                # print("Data from Serial Print 2:", frequency)
                                # print("Data from Serial Print 2:", value)
                                # rawValue = float(ser.readline().decode().strip())
                            # if (rawValue_ph2 >= 0 and prev_value < 0) or (rawValue_ph2 < 0 and prev_value >= 0):
                            #     count = count + 1

                            # if (rawValue_ph3 >= 0 and prev_value_ph3 < 0) or (rawValue_ph3 < 0 and prev_value_ph3 >= 0):
                            #     count_3 = count_3 + 1

                            prev_value_ph1 = rawValue_ph1
                            prev_value_ph2 = rawValue_ph2
                            prev_value_ph3 = rawValue_ph3

                            realPart_ph1[i] = rawValue_ph1 * \
                                math.cos(2 * math.pi * i / numSamples)
                            imagPart_ph1[i] = rawValue_ph1 * \
                                math.sin(2 * math.pi * i / numSamples)

                            realPart_ph2[i] = rawValue_ph2 * \
                                math.cos(2 * math.pi * i / numSamples)
                            imagPart_ph2[i] = rawValue_ph2 * \
                                math.sin(2 * math.pi * i / numSamples)

                            realPart_ph3[i] = rawValue_ph3 * \
                                math.cos(2 * math.pi * i / numSamples)
                            imagPart_ph3[i] = rawValue_ph3 * \
                                math.sin(2 * math.pi * i / numSamples)

                            if (i == 10):
                                try:
                                    # Connect to NTP server
                                    ntp_client = ntplib.NTPClient()
                                    response = ntp_client.request(
                                        'pool.ntp.org', version=3, timeout=1)
                                    ntp_time = datetime.datetime.fromtimestamp(
                                        response.tx_time)
                                    time_only = ntp_time.time()
                                    adjusted_time = (datetime.datetime.combine(datetime.datetime.today(), time_only) +
                                                     datetime.timedelta(hours=4, minutes=30)).time()
                                    # print(adjusted_time)

                                except (ntplib.NTPException, ConnectionError) as e:
                                    print('Error retrieving NTP time:', e)

                            # if count_3 == 2:
                            #     f_time_3 = time.time() - f_time_3
                            #     frequency_3 = 1/(f_time_3)
                            #     f_time_3 = time.time()
                            #     count_3 = 0

                            #     lowPassFrq_3 = lowPassAlpha_fre * frequency_3 + \
                            #         (1 - lowPassAlpha_fre) * lowPassFrq_3

                            #     highPassFrq_3 = highPassAlpha_freq * \
                            #         (lowPassFrq_3 - highPassFrq_3) + \
                            #         highPassFrq_3

                            # combined_message_3 = f"f3,{highPassFrq_3}"
                            # await websocket.send(combined_message_3)

                            # if count == 10:
                            #     f_time = time.time() - f_time
                            #     frequency_2 = 1/(f_time)
                            #     f_time = time.time()
                            #     count = 0

                            #     lowPassFrq = lowPassAlpha_fre * frequency_2 + \
                            #         (1 - lowPassAlpha_fre) * lowPassFrq

                            #     highPassFrq = highPassAlpha_freq * \
                            #         (lowPassFrq - highPassFrq) + \
                            #         highPassFrq

                        except ValueError:
                            print("Error: Could not convert data to integer")
                            realPart_ph1[i] = prev_value_ph1 * \
                                math.cos(2 * math.pi * i / numSamples)
                            imagPart_ph1[i] = prev_value_ph1 * \
                                math.sin(2 * math.pi * i / numSamples)

                            realPart_ph2[i] = prev_value_ph2 * \
                                math.cos(2 * math.pi * i / numSamples)
                            imagPart_ph2[i] = prev_value_ph2 * \
                                math.sin(2 * math.pi * i / numSamples)

                            realPart_ph3[i] = prev_value_ph3 * \
                                math.cos(2 * math.pi * i / numSamples)
                            imagPart_ph3[i] = prev_value_ph3 * \
                                math.sin(2 * math.pi * i / numSamples)

                    # Apply Hamming Window to real Part and Imag part of three phases
                    applyHammingWindow_ph1(realPart_ph1)
                    applyHammingWindow_ph1(imagPart_ph1)

                    applyHammingWindow_ph2(realPart_ph1)
                    applyHammingWindow_ph2(imagPart_ph2)

                    applyHammingWindow_ph3(realPart_ph3)
                    applyHammingWindow_ph3(realPart_ph3)

                    # Getting sum of the real part and img part
                    sumReal_ph1 = sum(realPart_ph1)
                    sumImag_ph1 = sum(imagPart_ph1)

                    sumReal_ph2 = sum(realPart_ph2)
                    sumImag_ph2 = sum(imagPart_ph2)

                    sumReal_ph3 = sum(realPart_ph3)
                    sumImag_ph3 = sum(imagPart_ph3)

                    sumReal_ph1 *= math.sqrt(2) / numSamples
                    sumImag_ph1 *= math.sqrt(2) / numSamples

                    sumReal_ph2 *= math.sqrt(2) / numSamples
                    sumImag_ph2 *= math.sqrt(2) / numSamples

                    sumReal_ph3 *= math.sqrt(2) / numSamples
                    sumImag_ph3 *= math.sqrt(2) / numSamples

                    magnitude_ph1 = math.sqrt(
                        sumReal_ph1 ** 2 + sumImag_ph1 ** 2)

                    magnitude_ph2 = math.sqrt(
                        sumReal_ph2 ** 2 + sumImag_ph2 ** 2)

                    magnitude_ph3 = math.sqrt(
                        sumReal_ph3 ** 2 + sumImag_ph3 ** 2)


                    # Filter the amplitude value
                    lowPassFilteredMagnitude_ph1 = lowPassAlpha * magnitude_ph1 + \
                        (1 - lowPassAlpha) * lowPassFilteredMagnitude_ph1
                    lowPassFilteredMagnitude_ph2 = lowPassAlpha * magnitude_ph2 + \
                        (1 - lowPassAlpha) * lowPassFilteredMagnitude_ph2
                    lowPassFilteredMagnitude_ph3 = lowPassAlpha * magnitude_ph3 + \
                        (1 - lowPassAlpha) * lowPassFilteredMagnitude_ph3

                    highPassFilteredMagnitude_ph1 = highPassAlpha * \
                        (lowPassFilteredMagnitude_ph1 - highPassFilteredMagnitude_ph1) + \
                        highPassFilteredMagnitude_ph1

                    highPassFilteredMagnitude_ph2 = highPassAlpha * \
                        (lowPassFilteredMagnitude_ph2 - highPassFilteredMagnitude_ph2) + \
                        highPassFilteredMagnitude_ph2

                    highPassFilteredMagnitude_ph3 = highPassAlpha * \
                        (lowPassFilteredMagnitude_ph3 - highPassFilteredMagnitude_ph3) + \
                        highPassFilteredMagnitude_ph3
                    
                        

                    # if lowPassFilteredMagnitude_ph2 >= 20:
                    #     lowPassAlpha = 0.002
                    # if highPassFilteredMagnitude_ph2 >= 20:
                    #     highPassFilteredMagnitude_ph1 = 0.002
                    # if abs(prv_lst - highPassFilteredMagnitude_ph2) >= 1 :
                    #     lowPassAlpha = 0.1
                    #     highPassAlpha = 0.1

                         

                    scale_mag_ph1 = -3.7719 + (1.3029*highPassFilteredMagnitude_ph1)
                    
                    scale_mag_ph2 = -3.7719 + (1.2539*highPassFilteredMagnitude_ph2)
                    #o = (-0.009920634920635088*highPassFilteredMagnitude_ph2**2) + (4.454365079365151*highPassFilteredMagnitude_ph2)  -258.0877976190549
                    #y=-3.7719+1.2649x
                    
                    scale_mag_ph3 =  -3.7719 + (1.2139*highPassFilteredMagnitude_ph3)
                    
                    if scale_mag_ph2 >= 220 :
                        
                        lowPassAlpha = 0.01
                        highPassAlpha = 0.1
                        
                    
                    if scale_mag_ph3 >= 220 :
                        
                        lowPassAlpha = 0.01
                        highPassAlpha = 0.1
                        
                    

                    #print("Scaled Magnitude Ph1", scale_mag_ph1)
                    #print("Scaled Magnitude Ph2:", scale_mag_ph2)
                    #print("Scaled Magnitude Ph3:", scale_mag_ph3)
                    #print("Scaled Freq", frequency)

                    combined_message = f"v,{scale_mag_ph1},{scale_mag_ph2},{scale_mag_ph3},{frequency},{frequency_2},{frequency_3},{adjusted_time}"
                    # await websocket.send(str(scale_mag),str(frequency))  # Convert count to a string
                    print(combined_message)
                    await websocket.send(combined_message)
    asyncio.get_event_loop().run_until_complete(connect())
