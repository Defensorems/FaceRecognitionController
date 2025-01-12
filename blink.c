#include <wiringPi.h>
int main (void) {
  wiringPiSetup();
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  digitalWrite(3, HIGH);
  delay(5000);
  digitalWrite(3, LOW);
  delay(5000);
  digitalWrite(4, HIGH);
  delay(5000);
  digitalWrite(4, LOW);
  return 0;
}