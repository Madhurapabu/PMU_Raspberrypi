
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
        window = 1.5 - 0.2 * math.cos(2 * math.pi * i / (numSamples - 1))
        data[i] *= window


async def applyHammingWindow_ph2(data):
    for i in range(numSamples):
        window = 1.5 - 0.2 * math.cos(2 * math.pi * i / (numSamples - 1))
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

                lowPassAlpha = 0.05
                lowPassFilteredMagnitude_ph1 = 0.00
                lowPassFilteredMagnitude_ph2 = 0.0
                lowPassFilteredMagnitude_ph3 = 0.0

                highPassAlpha = 0.02
                highPassFilteredMagnitude_ph1 = 0.00
                highPassFilteredMagnitude_ph2 = 0.0
                highPassFilteredMagnitude_ph3 = 0.0

                rawValue_ph1 = 0.0
                rawValue_ph2 = 0.0
                rawValue_ph3 = 0.0

                frequency = 0.0

                time = 0.0

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
                                #print(rawValue_ph2)
                            elif line.startswith("Ph3:"):
                                rawValue_ph3 = float(line.split(":")[1])
                            elif line.startswith("Data2:"):
                                frequency = line.split(":")[1]
                                # print("Data from Serial Print 2:", frequency)
                                # print("Data from Serial Print 2:", value)
                                # rawValue = float(ser.readline().decode().strip())

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

                    scale_mag_ph1 = 1.21 * highPassFilteredMagnitude_ph1 - 0.86
                    scale_mag_ph2 = 1.23 * highPassFilteredMagnitude_ph2 - 0.86
                    scale_mag_ph3 = 1.25 *  highPassFilteredMagnitude_ph3 - 0.86

                    print("Scaled Magnitude Ph1", scale_mag_ph1)
                    print("Scaled Magnitude Ph2:", scale_mag_ph2)
                    print("Scaled Magnitude Ph3:", scale_mag_ph3)
                    print("Scaled Freq", frequency)

                    combined_message = f"v,{scale_mag_ph1},{scale_mag_ph2},{scale_mag_ph3},{frequency},{adjusted_time}"
                    # await websocket.send(str(scale_mag),str(frequency))  # Convert count to a string
                    await websocket.send(combined_message)
    asyncio.get_event_loop().run_until_complete(connect())
