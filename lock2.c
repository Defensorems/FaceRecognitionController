#include <wiringPi.h>
int main(void) {
	wiringPiSetup();
	pinMode(4, OUTPUT);
	digitalWrite(4, HIGH);
	delay(5000);
	digitalWrite(4, LOW);
	return 0;
}