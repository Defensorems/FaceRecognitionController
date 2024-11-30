import wiringpi

wiringpi.wiringPiSetup()

def open_lock(pin):
    wiringpi.pinMode(pin, 1)
    wiringpi.digitalWrite(pin, 1)

def close_lock(pin):
    wiringpi.pinMode(pin, 1)
    wiringpi.digitalWrite(pin, 0)


