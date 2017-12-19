import mindwave
import time

def main():
    headset = mindwave.Headset('/dev/rfcomm0')
    time.sleep(2)
    headset.connect()
    
    while True:
        # poor_signal, attention, meditation, blink, raw_value, status
        attention = headset.attention
        print "Attention: ", attention

if __name__ == '__main__':
    # Find MAC address(XX:XX:XX:XX:XX:XX) of MindWave Mobile
    # $ hcitool scan
    # Connect to MindWave Mobile
    # $ sudo rfcomm connect /dev/rfcomm0 MAC_ADDRESS
    # Check connection(on another terminal)
    # $ hexdump /dev/rfcomm0

    # Sample eeg acquisition code using
    # https://github.com/BarkleyUS/mindwave-python
    main()
