#include <wiringPi.h>
int main (void) {
  wiringPiSetup();
  pinMode(3, OUTPUT);
  digitalWrite(3, HIGH);
  delay(5000);
  digitalWrite(3, LOW);
  return 0;
}