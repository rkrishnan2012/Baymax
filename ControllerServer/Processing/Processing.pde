// Learning Processing
// Daniel Shiffman
// http://www.learningprocessing.com11

// Example 19-1: Simple therapy server

// Import the net libraries
import processing.net.*;

// Declare a server
Server server;

// Used to indicate a new message has arrived
float newMessageColor = 255;
PFont f;
String incomingMessage = "";

void setup() {
  size(600,600);
  // Create the Server on port 8080
  server = new Server(this, 8080); 
  f = createFont("Arial",16,true);
}

int xPos = 0;
int throttle = 0;
Client client;

void draw() {
  // The most recent incoming message is displayed in the window.
  //text(incomingMessage,width/2,height/2); 
  // If a client is available, we will find out
  // If there is no client, it will be"null"
  client = server.available();
  // We should only proceed if the client is not null
  if (client!= null) {
    // Receive the message
    // The message is read using readString().
    incomingMessage = client.readString(); 
    // The trim() function is used to remove the extra line break that comes in with the message.
    incomingMessage = incomingMessage.trim();
    //println(incomingMessage);
    // Write message back out (note this goes to ALL clients)
    //server.write( "How does " + incomingMessage + " make you feel?\n" ); // A reply is sent using write().
    float[] nums = float(split(incomingMessage, ','));
    if(nums.length > 2){
      stroke(255, 0, 0);
      line(xPos, (3 * height / 4), xPos, (3 * height / 4) - ((20*nums[0])));
      stroke(0, 255, 0);
      line(xPos, height / 4, xPos, (height / 4) - ((3*nums[1]) ));
      if (xPos >= width) {
        xPos = 0;
        background(0);
      } else {
        // increment the horizontal position:
        xPos++;
      }
    }
    // Reset newMessageColor to black
    newMessageColor = 0;
  }
}

void keyPressed() {
  if (keyCode == UP) {
    throttle++;
  } else if (keyCode == DOWN) {
    throttle--;
  } 
  println("Changing throttle to " + throttle);
  server.write(throttle + "\n");
}

// The serverEvent function is called whenever a new client connects.
void serverEvent(Server server, Client client) {
  incomingMessage = "A new client has connected: " + client.ip();
  //println(incomingMessage);
  // Reset newMessageColor to black
  newMessageColor = 0;
}