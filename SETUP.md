# Complete Setup Instructions for Raspberry Pi 4 Model B with Multiple Sensors

## Prerequisites

1. Raspberry Pi 4 Model B with Raspberry Pi OS installed
2. MicroSD card with at least 16GB storage
3. Power supply for Raspberry Pi (5V/3A)
4. Breadboard for initial testing
5. Jumper wires (male-to-female and male-to-male)
6. Basic tools (small screwdriver, wire cutters/strippers)
7. MCP3008 ADC chip (for analog sensors)
8. 10kΩ resistors (3x)

## 1. Enabling Required Interfaces on Raspberry Pi

Start by enabling the required interfaces:

```bash
# Update your Raspberry Pi
sudo apt update
sudo apt upgrade -y

# Install needed packages
sudo apt install -y python3-pip i2c-tools python3-dev

# Install SPI and I2C Python libraries
sudo pip3 install spidev adafruit-blinka
```

Enable I2C and SPI interfaces via raspi-config:

```bash
sudo raspi-config
```

Navigate to "Interface Options" and enable both "I2C" and "SPI". Then reboot:

```bash
sudo reboot
```

## 2. Connecting Adafruit MCP9808 Temperature Sensor

The MCP9808 uses I2C protocol to communicate with the Raspberry Pi.

### Wiring Instructions:

1. **Power off your Raspberry Pi** before connecting any hardware
2. Connect the MCP9808 to the Raspberry Pi as follows:
   - MCP9808 VDD → Raspberry Pi 3.3V (Pin 1)
   - MCP9808 GND → Raspberry Pi GND (Pin 6)
   - MCP9808 SCL → Raspberry Pi SCL (Pin 5, GPIO 3)
   - MCP9808 SDA → Raspberry Pi SDA (Pin 3, GPIO 2)
   - Leave A0, A1, A2, and ALERT pins disconnected (unless you want to use the alert function)

### Verification:

Power on the Raspberry Pi and verify the sensor is detected:

```bash
sudo i2cdetect -y 1
```

You should see a device at address 0x18 in the grid output (the default I2C address for MCP9808).

### Install Required Libraries:

```bash
sudo pip3 install adafruit-circuitpython-mcp9808
```

### Test the Sensor:

Create a test script named `test_mcp9808.py`:

```python
import time
import board
import busio
import adafruit_mcp9808

# Create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create sensor instance
sensor = adafruit_mcp9808.MCP9808(i2c)

# Read and print temperature
for i in range(10):
    temp_c = sensor.temperature
    print(f"Temperature: {temp_c:.2f}°C / {(temp_c * 9/5) + 32:.2f}°F")
    time.sleep(1)
```

Run the test:
```bash
python3 test_mcp9808.py
```

## 3. Connecting MQ Gas Sensors (MQ135, MQ7, MQ2)

All three gas sensors (MQ135, MQ7, and MQ2) are analog sensors, so they require an Analog-to-Digital Converter (ADC) since the Raspberry Pi doesn't have analog inputs. We'll use the MCP3008 ADC for this purpose.

### Wiring the MCP3008 ADC:

1. First, connect the MCP3008 to the Raspberry Pi:
   - MCP3008 VDD → Raspberry Pi 3.3V (Pin 1)
   - MCP3008 VREF → Raspberry Pi 3.3V (Pin 1)
   - MCP3008 AGND → Raspberry Pi GND (Pin 9)
   - MCP3008 DGND → Raspberry Pi GND (Pin 9)
   - MCP3008 CLK → Raspberry Pi SCLK (Pin 23, GPIO 11)
   - MCP3008 DOUT → Raspberry Pi MISO (Pin 21, GPIO 9)
   - MCP3008 DIN → Raspberry Pi MOSI (Pin 19, GPIO 10)
   - MCP3008 CS → Raspberry Pi CE0 (Pin 24, GPIO 8)

### Wiring the MQ135 Gas Sensor (Air Quality):

1. Connect the MQ135 to the power and MCP3008:
   - MQ135 VCC → Raspberry Pi 5V (Pin 2)
   - MQ135 GND → Raspberry Pi GND (Pin 6)
   - MQ135 AOUT → MCP3008 CH0 (Channel 0)
   - Connect a 10kΩ resistor between VCC and AOUT (load resistor)

### Wiring the MQ7 Carbon Monoxide Sensor:

1. Connect the MQ7 to the power and MCP3008:
   - MQ7 VCC → Raspberry Pi 5V (Pin 4)
   - MQ7 GND → Raspberry Pi GND (Pin 14)
   - MQ7 AOUT → MCP3008 CH1 (Channel 1)
   - Connect a 10kΩ resistor between VCC and AOUT (load resistor)

### Wiring the MQ2 Smoke Detector:

1. Connect the MQ2 to the power and MCP3008:
   - MQ2 VCC → Raspberry Pi 5V (Pin 4)
   - MQ2 GND → Raspberry Pi GND (Pin 20)
   - MQ2 AOUT → MCP3008 CH2 (Channel 2)
   - Connect a 10kΩ resistor between VCC and AOUT (load resistor)

### Install Required Libraries:

```bash
sudo pip3 install spidev
```

### Testing the MQ Sensors:

Create a test script named `test_mq_sensors.py`:

```python
import time
import spidev

# Setup SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, device 0
spi.max_speed_hz = 1000000  # 1MHz

# Function to read SPI data from MCP3008
def read_adc(channel):
    if channel < 0 or channel > 7:
        return -1
        
    # MCP3008 command format
    # First byte: Start bit (1), single/diff (1), channel bits (3), don't-care bits (3)
    # Second byte: All don't-care bits
    r = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((r[1] & 3) << 8) + r[2]
    return data

try:
    print("MQ Sensor Test - Press CTRL+C to exit")
    print("Wait for sensors to warm up (30 seconds)...")
    
    # Wait for warm-up (the sensors need time to heat up)
    for i in range(30, 0, -1):
        print(f"Warming up: {i} seconds remaining", end="\r")
        time.sleep(1)
    
    print("\nSensors ready! Reading values...")
    
    # Main loop
    while True:
        # Read sensor values
        mq135_value = read_adc(0)
        mq7_value = read_adc(1)
        mq2_value = read_adc(2)
        
        # Print values
        print("\nRaw Sensor Values:")
        print(f"MQ135 (Air Quality): {mq135_value} (0-1023)")
        print(f"MQ7 (Carbon Monoxide): {mq7_value} (0-1023)")
        print(f"MQ2 (Smoke): {mq2_value} (0-1023)")
        
        # Simple approximation to PPM
        # Note: Proper calibration would require specific gas concentrations
        mq135_ppm = mq135_value / 4  # Simple approximation
        mq7_ppm = mq7_value / 5      # Simple approximation
        mq2_ppm = mq2_value / 3      # Simple approximation
        
        print("\nApproximate PPM Values (uncalibrated):")
        print(f"MQ135 (Air Quality): {mq135_ppm:.2f} ppm")
        print(f"MQ7 (Carbon Monoxide): {mq7_ppm:.2f} ppm")
        print(f"MQ2 (Smoke): {mq2_ppm:.2f} ppm")
        
        print("-" * 50)
        time.sleep(3)
        
except KeyboardInterrupt:
    print("\nTest stopped by user")
finally:
    spi.close()
```

Run the test:
```bash
python3 test_mq_sensors.py
```

## Important Notes About MQ Sensors

1. **Warm-up Period**: MQ sensors require a warm-up period (typically 24-48 hours for full calibration, but 1-2 minutes for basic functionality). During this time, they heat up their sensing element.

2. **Power Requirements**: MQ sensors draw significant current (around 150mA each) and can get quite hot during operation. This is normal but make sure your power supply can handle the load.

3. **Digital Outputs**: Some MQ sensors also have a digital output pin (usually marked as DOUT). This is a simple threshold-based on/off signal that you can connect directly to a GPIO pin if you want simple detection rather than analog readings.

4. **Calibration**: For accurate PPM readings, calibration with known gas concentrations is necessary. The simple approximations in the test script are just for demonstration.

5. **Environmental Factors**: Temperature and humidity can affect readings. For high-precision applications, consider environmental compensation.

## Complete System Test

Now let's create a script that reads from all sensors and prints the values:

```python
import time
import board
import busio
import adafruit_mcp9808
import spidev

# Setup I2C for MCP9808
i2c = busio.I2C(board.SCL, board.SDA)
temp_sensor = adafruit_mcp9808.MCP9808(i2c)

# Setup SPI for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

# Function to read SPI data from MCP3008
def read_adc(channel):
    if channel < 0 or channel > 7:
        return -1
    r = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((r[1] & 3) << 8) + r[2]
    return data

try:
    print("SeekLiyab Sensor Test - Press CTRL+C to exit")
    
    while True:
        # Read temperature
        temp_c = temp_sensor.temperature
        
        # Read gas sensors
        mq135_value = read_adc(0)
        mq7_value = read_adc(1)
        mq2_value = read_adc(2)
        
        # Print values
        print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Temperature: {temp_c:.2f}°C / {(temp_c * 9/5) + 32:.2f}°F")
        print(f"MQ135 (Air Quality): {mq135_value} raw")
        print(f"MQ7 (Carbon Monoxide): {mq7_value} raw")
        print(f"MQ2 (Smoke): {mq2_value} raw")
        print("-" * 50)
        
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\nTest stopped by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    spi.close()
```

## Making Permanent Connections

Once you've verified everything works on a breadboard, you can make the connections permanent:

1. **Soldering**: Solder the connections to a perfboard or stripboard
2. **Terminals**: Use screw terminals for sensors that might need replacement
3. **Enclosure**: Place the circuit in an appropriate enclosure
4. **Labeling**: Label all wires and connections for future maintenance

## Troubleshooting Common Issues

1. **I2C Device Not Found**:
   - Check wiring connections
   - Verify I2C is enabled (`sudo raspi-config`)
   - Try a different I2C address (`sudo i2cdetect -y 1`)

2. **ADC Readings Are Unstable**:
   - Add decoupling capacitors (0.1μF) near the MCP3008 power pins
   - Shorten wire lengths to minimize noise
   - Ensure good ground connections

3. **MQ Sensors Not Responding**:
   - Check power connections (MQ sensors need 5V)
   - Verify the heating coil is working (should feel warm)
   - Ensure load resistors are connected correctly

4. **Temperature Readings Inaccurate**:
   - Keep the MCP9808 away from heat sources (including MQ sensors)
   - Allow for proper air circulation around the sensor

5. **System Crashes or Reboots**:
   - Check power supply rating (MQ sensors draw significant current)
   - Add a larger power supply if necessary

## Going Further

Once your hardware is properly set up, you can integrate it with the data collection and API scripts provided in earlier responses to create your complete SeekLiyab environmental monitoring system.
## Features

- Data Transformation: Converts raw attendance data into an easily understandable and manageable format.
- Master List Integration: Merges attendance data with the master employee list for comprehensive management.
- Schedule Processing: Incorporates employee schedules to accurately apply attendance codes.
- Custom Filters: Offers customizable filters for employee name, LOB, shift, site, leader, and employer for targeted data analysis.
- Data Visualization: Visualizes multiple logs and missed punches with progress columns for easy understanding.
- Downloadable Reports: Allows for the downloading of processed attendance data in Excel format, complete with styled cells for better readability.

## User Guide
- Upload Files: Upload the required Excel files for attendance raw data, master list, and schedule through the provided file uploaders.
- Data Processing: Once all files are uploaded, the application processes the data, merging the attendance with the master list and applying the necessary codes based on the employee schedule.
- Customization: Use the sidebar to filter the displayed data according to specific criteria such as employee name, LOB, shift, among others.
- Visualization: Analyze the processed data through the visualization of multiple logs and missed punches. The application uses progress columns for a clear representation of the data.
- Download: Finally, download the processed and styled attendance data in Excel format for offline analysis or record-keeping.




## About the Developer

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jpcurada/)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Feedback

If you have any feedback, please reach out to us at fake@fake.com


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


# Project Title

A brief description of what this project does and who it's for

