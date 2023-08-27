import math
import time
import serial
# import ntplib
import datetime
import asyncio
import websockets
import ntplib


lowPassAlpha_fre = 0.008
lowPassFrq = 0.0


highPassAlpha_freq = 0.01
highPassFrq = 0.0


f_time = 0.0
frequency = 0.0
count = 0
prev_value = 0

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

            f_time = time.time()

            rawValue = 0
            prev_value = 0
            adjusted_time = 0
            
            while True:
                message = await websocket.recv()
                print("Message from server:", message)
                
                lowPassAlpha_fre = 0.008
                lowPassFrq = 0.0
                lowPassFrq_3 = 0.0


                highPassAlpha_freq = 0.01
                highPassFrq = 0.0
                highPassFrq_3 = 0.0


                f_time = 0.0
                frequency = 0.0
                count = 0
                
                
                f_time_3 = 0.0
                frequency_3 = 0.0
                count_3 = 0
                rawValue_ph3 = 0
                prev_value_ph3 = 0
                
                
                

                while True:
                    try:
                        line = ser.readline().decode().strip()
                        
                        if line.startswith("Ph2:"):
                            rawValue = float(line.split(":")[1])
                            
                        if line.startswith("Ph3:"):
                            rawValue_ph3 = float(line.split(":")[1])
                    
                    except ValueError:
                        print("Error: Could not convert data to integer")
                        
                    if (rawValue >= 0 and prev_value < 0) or (rawValue < 0 and prev_value >= 0):
                        count = count + 1
                        
                    if (rawValue_ph3 >= 0 and prev_value_ph3 < 0) or (rawValue_ph3 < 0 and prev_value_ph3 >= 0):
                        count_3 = count_3 + 1

                    prev_value = rawValue
                    prev_value_ph3 = rawValue_ph3

                    if count_3 == 2:
                        f_time_3 = time.time() - f_time_3
                        frequency_3 = 1/(f_time_3)
                        f_time_3 = time.time()
                        count_3 = 0
                        
                        
                        lowPassFrq_3 = lowPassAlpha_fre * frequency_3 + \
                    (1 - lowPassAlpha_fre) * lowPassFrq_3

                        highPassFrq_3 = highPassAlpha_freq * \
                            (lowPassFrq_3 - highPassFrq_3) + \
                            highPassFrq_3
                        
                        
                        combined_message = f"f3,{highPassFrq_3}"
                        await websocket.send(combined_message)
                    
                    if count == 2:
                        f_time = time.time() - f_time
                        frequency = 1/(f_time)
                        f_time = time.time()
                        count = 0
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
                        
                        
                        lowPassFrq = lowPassAlpha_fre * frequency + \
                    (1 - lowPassAlpha_fre) * lowPassFrq

                        highPassFrq = highPassAlpha_freq * \
                            (lowPassFrq - highPassFrq) + \
                            highPassFrq

                        print("fre:", highPassFrq)
                        
                        
                        combined_message = f"f2,{highPassFrq},{adjusted_time}"
                        await websocket.send(combined_message)
        
    asyncio.get_event_loop().run_until_complete(connect())

        
