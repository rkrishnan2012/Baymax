#include <Servo.h>
#include <SoftwareSerial.h>


Servo esc[4];
int motors[4] = { 5, 6, 9, 10 };
int calibrations[4] = { 179, 179, 179, 179 };

int CMD_FLAP = 10;
int CMD_CALIBRATE = 11;
int CMD_INDIVIDUAL = 12;

int MAX_MILLIS_TO_WAIT = 1000;

int motorValues[4];
SoftwareSerial PhoneSerial(12, 13); // RX, TX

void setup() {
  Serial.begin(9600);
  PhoneSerial.begin(9600);
  PhoneSerial.println("Hello Baymax :)");
 
  pinMode(A5, INPUT);
  for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
    esc[i].attach(motors[i]);  
  }
}

void waitForBatteryConnect(){
  Serial.println("Plug the battery in.");
  PhoneSerial.println("Plug the battery in.");
  while(battVoltage() < 1020){ }
  Serial.println("Found battery. Turning off motors.");
  PhoneSerial.println("Found battery. Turning off motors.");
  for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
    esc[i].write(10);  
  }
  delay(3000);
}

int battVoltage(){
  delay(300);
  analogRead(A5);
  int batt = analogRead(A5);
  return batt;
}

void individual() {
  Serial.println("Starting individual motor test.");
  PhoneSerial.println("Starting individual motor test.");
  waitForBatteryConnect();
  delay(4000);
  for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
      Serial.print("Running motor ");
      Serial.println(i);
      PhoneSerial.print("Running motor ");
      PhoneSerial.println(i);
      for(int j = 0; j < 100; j+=10){
        int speed = j * (179 - 10) / 100;
        esc[i].write(speed);  
        delay(160);
      }
      for(int j = 100; j >= 0; j-=10){
        int speed = j * (179 - 10) / 100;
        esc[i].write(speed);  
        delay(160);
      }
    }
    delay(4000);
}

void flap(){  
  Serial.println("Starting flap test.");
  PhoneSerial.println("Starting flap test.");
  waitForBatteryConnect();
  delay(4000);
  for(int k = 0; k < 5; k++){
    for(int j = 0; j < 100; j+=10){
      for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
        int speed = j * (179 - 10) / 100;
        Serial.println(speed);
        PhoneSerial.println(speed);
        esc[i].write(speed);  
      }
      delay(40);
    }
    for(int j = 100; j >= 0; j-=10){
      for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
        int speed = j * (179 - 10) / 100;
        Serial.println(speed);
        PhoneSerial.println(speed);
        esc[i].write(speed);  
      }
      delay(40);
    }
  }
}

void calibrate(){
  Serial.println("ENTERING CALIBRATION MODE!");
  Serial.println("Unplug the battery. Wait.");
  while(battVoltage() > 10){ }
  delay(3000);
  for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
    esc[i].write(calibrations[i]);  
  }
  delay(5000);
  waitForBatteryConnect();
  delay(16000);
  Serial.println("  Writing low.");
  for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
    esc[i].write(0);  
  }
  delay(2000);
  Serial.println("Unplug the battery NOW!.");
  while(battVoltage() > 100){ }
  Serial.println("Good job, now wait.");
  delay(3000);
  for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
    esc[i].write(10);  
  }
  delay(5000);
  Serial.println("Re-Plug the battery in.");
  while(battVoltage() < 100){ }
  Serial.println("Found the battery.");
  delay(3000);
  Serial.println("Ready.");
}

void loop() {
  /*while (Serial.available() == 0);
  int val = Serial.parseInt(); //read int or parseFloat for ..float...
  Serial.println(val);
  PhoneSerial.println(val);
  for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
    esc[i].write(val);  
  }
  }*/
  long starttime = millis();
  while(PhoneSerial.available() < 4 && ((millis() - starttime) < MAX_MILLIS_TO_WAIT)){}
  if(PhoneSerial.available() >= 4){
    uint32_t val[4];
    for(int n = 0; n < 4; n++){
      val[n] = PhoneSerial.read();
    }
    int x = ((val[0] & 0xFF) << 24) | ((val[1] & 0xFF) << 16)
        | ((val[2] & 0xFF) << 8) | (val[3] & 0xFF);
        
    if(x >= 1000 && x < 2000){
      motorValues[0] = x - 1000;
    } else if(x >= 2000 && x < 3000){
      motorValues[1] = x - 2000;
    } else if(x >= 3000 && x < 4000){
      motorValues[2] = x - 3000;
    } else if(x >= 4000 && x < 5000){
      motorValues[3] = x - 4000;
    } else {
      Serial.println(x);
      Serial.println("===");
      if(x == CMD_FLAP){
        flap();
      } else if(x == CMD_INDIVIDUAL){
        individual();
      }
    }
    for(int i = 0; i < sizeof(motors) / sizeof(int); i++){
      esc[i].write(motorValues[i]);  
    }
    delay(100);
  }
} 
