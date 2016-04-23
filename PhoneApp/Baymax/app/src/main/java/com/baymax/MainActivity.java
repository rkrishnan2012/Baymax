package com.baymax;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialPort;
import com.hoho.android.usbserial.driver.UsbSerialProber;
import com.hoho.android.usbserial.util.SerialInputOutputManager;

import org.pid4j.pid.DefaultPid;
import org.pid4j.pid.Pid;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.InetAddress;
import java.net.Socket;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements View.OnClickListener,
        SeekBar.OnSeekBarChangeListener, SensorEventListener {

    private UsbManager mUsbManager;
    private SerialInputOutputManager mSerialIoManager;
    UsbSerialPort mPort;

    private TextView mStatus;
    private TextView mDumpText;
    private ScrollView mScrollView;

    private static final int TIMEOUT = 500;
    private static final int CMD_FLAP = 10;
    private static final int CMD_RESET = 11;
    private static final int CMD_INDIVIDUAL = 12;

    private static final int REAR_RIGHT = 1;
    private static final int FRONT_LEFT = 2;
    private static final int FRONT_RIGHT = 3;
    private static final int REAR_LEFT = 4;

    private static final int[] MOTOR_MINS = {0, 0, 12, 25};

    boolean connected = false;

    int throttle = 30;

    private static final String TAG = "Baymax";
    private static final String ACTION_USB_PERMISSION = "com.hoho.android.usbserial.examples.USB_PERMISSION";

    private final ExecutorService mExecutor = Executors.newSingleThreadExecutor();

    private static SensorManager mSensorManager;
    private static Sensor mAccelerometer;
    private static Sensor mMagnetic;
    private static Sensor mGyroscope;

    long lastUpdateAccelerometer = 0;
    PrintWriter sensorWriter;
    PrintWriter motorWriter;


    private void updateReceivedData(final byte[] data) {
        mDumpText.post(new Runnable() {
            @Override
            public void run() {
                mDumpText.append(new String(data));
            }
        });
    }

    private void onConnected(){
        connected = true;
        mStatus.post(new Runnable() {
            @Override
            public void run() {
                mStatus.setText("Connected to " + mPort.getDriver().getDevice().getDeviceName());
            }
        });
    }

    private void onErrorConnecting(final String error){
        mStatus.post(new Runnable() {
            @Override
            public void run() {
                mStatus.setText("Error connecting. " + error);
            }
        });
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mStatus = (TextView) findViewById(R.id.txtStatus);
        mDumpText = (TextView) findViewById(R.id.consoleText);
        mScrollView = (ScrollView)findViewById(R.id.scroller);

        findViewById(R.id.flap).setOnClickListener(this);
        findViewById(R.id.reset).setOnClickListener(this);
        findViewById(R.id.individual).setOnClickListener(this);

        ((SeekBar)findViewById((R.id.motor1))).setOnSeekBarChangeListener(this);
        ((SeekBar)findViewById((R.id.motor2))).setOnSeekBarChangeListener(this);
        ((SeekBar)findViewById((R.id.motor3))).setOnSeekBarChangeListener(this);
        ((SeekBar)findViewById((R.id.motor4))).setOnSeekBarChangeListener(this);

        mUsbManager = (UsbManager) getSystemService(Context.USB_SERVICE);

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mMagnetic = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, mAccelerometer , SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mMagnetic , SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mGyroscope , SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
        if(mUsbReceiver != null && connected){
            unregisterReceiver(mUsbReceiver);
        }
        stopIoManager();
        if (mPort != null) {
            try {
                mPort.close();
            } catch (IOException ignored) {}
            mPort = null;
        }
        finish();
    }

    @Override
    protected void onResume(){
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mMagnetic, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_GAME);
        connectToBaymax();
    }

    protected void connectToBaymax() {
        final List<UsbSerialDriver> drivers =
                UsbSerialProber.getDefaultProber().findAllDrivers(mUsbManager);
        if (drivers.size() > 0) {
            final List<UsbSerialPort> ports = drivers.get(0).getPorts();
            if (ports.size() > 0) {
                mPort = ports.get(0);
                mStatus.setText("Found device at " + mPort.getDriver().getDevice().getDeviceName());
                PendingIntent mPermissionIntent;
                mPermissionIntent = PendingIntent.getBroadcast(this, 0, new Intent(ACTION_USB_PERMISSION), 0);
                IntentFilter filter = new IntentFilter(ACTION_USB_PERMISSION);
                registerReceiver(mUsbReceiver, filter);
                mUsbManager.requestPermission(mPort.getDriver().getDevice(), mPermissionIntent);
            } else {
                mStatus.setText("Baymax disconnected.");
            }
        } else {
            mStatus.setText("Baymax disconnected.");
        }
    }

    private void stopIoManager() {
        if (mSerialIoManager != null) {
            Log.i(TAG, "Stopping io manager ..");
            mSerialIoManager.stop();
            mSerialIoManager = null;
        }
    }

    private void startIoManager() {
        if (mPort != null) {
            Log.i(TAG, "Starting io manager ..");
            mSerialIoManager = new SerialInputOutputManager(mPort, mListener);
            mExecutor.submit(mSerialIoManager);
        }
    }

    private void onDeviceStateChange() {
        stopIoManager();
        startIoManager();
    }

    private final SerialInputOutputManager.Listener mListener =
            new SerialInputOutputManager.Listener() {

                @Override
                public void onRunError(Exception e) {
                    Log.d(TAG, "Runner stopped.");
                }

                @Override
                public void onNewData(final byte[] data) {
                    MainActivity.this.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            MainActivity.this.updateReceivedData(data);
                        }
                    });
                }
            };

    private final BroadcastReceiver mUsbReceiver = new BroadcastReceiver() {
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (ACTION_USB_PERMISSION.equals(action)) {
                synchronized (this) {
                    UsbDevice device = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE);
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        if(device != null){
                            UsbDeviceConnection connection = mUsbManager.openDevice(mPort.getDriver().getDevice());
                            if (connection == null) {
                                onErrorConnecting("Device not found.");
                                return;
                            }

                            try {
                                mPort.setRTS(false);
                                mPort.setDTR(false);
                                mPort.open(connection);
                                mPort.setParameters(9600, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE);
                            } catch (IOException e) {
                                Log.e(TAG, "Error setting up device: " + e.getMessage(), e);
                                onErrorConnecting(e.getMessage());
                                try {
                                    mPort.close();
                                } catch (IOException ignored) {}
                                mPort = null;
                                return;
                            }
                            onConnected();
                        }
                    }
                    else {
                        Log.e(TAG, "permission denied for device " + device);
                    }
                    onDeviceStateChange();
                }
            }
        }
    };

    void sendMotor(int motor, int power){
        if(power < 0) power = 0;
        if(power > 100) power = 100;
        //  Compensate for all motors not starting at same time.
        if(power > 0){
            power = (int)
                    (((power / 100f) * (100f - MOTOR_MINS[motor - 1])) + (MOTOR_MINS[motor - 1]));
        }
        int value = (motor * 1000) + (180 * power / 100);
        byte[] ary = new byte[4];
        ary[0] = (byte) ((value >> 24) & 0xFF);
        ary[1] = (byte) ((value >> 16) & 0xFF);
        ary[2] = (byte) ((value >> 8) & 0xFF);
        ary[3] = (byte) (value & 0xFF);
        try {
            mPort.write(ary, TIMEOUT);
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onClick(View v) {
        try{
            int value = 0;
            switch(v.getId()){
                case R.id.flap:{
                    throttle = 0;
                    CommunicationThread.start();
                    AcroThread.start();
                    break;
                }
                case R.id.reset: {
                    value = CMD_RESET;
                    ((SeekBar)(findViewById(R.id.motor1))).setProgress(0);
                    ((SeekBar)(findViewById(R.id.motor2))).setProgress(0);
                    ((SeekBar)(findViewById(R.id.motor3))).setProgress(0);
                    ((SeekBar)(findViewById(R.id.motor4))).setProgress(0);
                    sendMotor(1, 0);
                    sendMotor(2, 0);
                    sendMotor(3, 0);
                    sendMotor(4, 0);
                    break;
                }
                case R.id.individual: {
                    value = CMD_INDIVIDUAL;
                    byte[] ary = new byte[4];
                    ary[0] = (byte) ((value >> 24) & 0xFF);
                    ary[1] = (byte) ((value >> 16) & 0xFF);
                    ary[2] = (byte) ((value >> 8) & 0xFF);
                    ary[3] = (byte) (value & 0xFF);
                    mPort.write(ary, TIMEOUT);
                    break;
                }
            }

        } catch (IOException ignored){}
    }
    long lastTime = 0;

    @Override
    public synchronized void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        try{
            int value = 0;
            switch(seekBar.getId()){
                case R.id.motor1:
                    value = 1000 + (180 * progress / 100);
                    break;
                case R.id.motor2:
                    value = 2000 + (180 * progress / 100);
                    break;
                case R.id.motor3:
                    value = 3000 + (180 * progress / 100);
                    break;
                case R.id.motor4:
                    value = 4000 + (180 * progress / 100);
                    break;

            }
            byte[] ary = new byte[4];
            ary[0] = (byte) ((value >> 24) & 0xFF);
            ary[1] = (byte) ((value >> 16) & 0xFF);
            ary[2] = (byte) ((value >> 8) & 0xFF);
            ary[3] = (byte) (value & 0xFF);
            if(System.currentTimeMillis() - lastTime > 100){
                mPort.write(ary, TIMEOUT);
                lastTime = System.currentTimeMillis();
                final int finalVal = value;
                mDumpText.post(new Runnable() {
                    @Override
                    public void run() {
                        //mDumpText.append("Sending " + finalVal + ".\n");
                    }
                });
            }
        } catch (IOException ignored){}
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {}

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {}

    float[] mGravity;
    float[] mGeomagnetic;
    float[] mAccelRotation;
    float[] mOrientation;

    long startTime = -1;

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
            mGravity = event.values;
        if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD)
            mGeomagnetic = event.values;
        if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE)
            mAccelRotation = event.values;
        if (mGravity != null && mGeomagnetic != null) {
            float R[] = new float[9];
            float I[] = new float[9];
            boolean success = SensorManager.getRotationMatrix(R, I, mGravity, mGeomagnetic);
            if (success) {
                float orientation[] = new float[3];
                SensorManager.getOrientation(R, orientation);
                mOrientation = orientation;
                if(startTime == -1){
                    startTime = System.currentTimeMillis();
                }
            }
        }
    }
    private Thread CommunicationThread = new Thread(new Runnable(){
        public PrintWriter writer;
        public BufferedReader reader;
        public Socket socket;

        @Override
        public void run() {
            try{
                socket = new Socket(InetAddress.getByName("10.0.0.14"), 8080);
                reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                writer = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()));
                String line;
                while(true){
                    line = mAccelRotation[0] + "," + FR + ", " + BL+ "\n";
                    writer.write(line);
                    writer.flush();
                    if(socket.getInputStream().available() > 0){
                        if((line = reader.readLine()) != null){
                            throttle = Integer.parseInt(line);
                        }
                    }
                    Thread.sleep(50);
                }
            }
            catch(Exception e){
                e.printStackTrace();
            }
        }
    });


    private Thread AcroThread = new Thread(new Runnable() {
        @Override
        public void run() {
            long lastCommand = -1;
            long startTime = -1;
            pitchpid.setKpid(.7, .3, 50);
            pitchpid.setOutputLimits(-100.0, 100.0);
            rollpid.setKpid(.7, .3, 50);
            rollpid.setOutputLimits(-100.0, 100.0);
            try {
                new File("/sdcard/sensorData.txt").createNewFile();
                new File("/sdcard/motorData.txt").createNewFile();
                sensorWriter = new PrintWriter("/sdcard/sensorData.txt", "UTF-8");
                motorWriter = new PrintWriter("/sdcard/motorData.txt", "UTF-8");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            while(true){
                if(mAccelRotation != null && System.currentTimeMillis() - lastCommand > 100){
                    lastCommand = System.currentTimeMillis();
                    runAcro(throttle,
                            new double[]{
                                    0,
                                    0,
                                    180 * mAccelRotation[2] / Math.PI,
                            },
                            new double[]{
                                    180 * mAccelRotation[1] / Math.PI,
                                    180 * mAccelRotation[0] / Math.PI,
                                    180 * mAccelRotation[2] / Math.PI,
                            });
                }
            }
        }
    });

    int FL, BL, FR, BR;
    Pid pitchpid = new DefaultPid();
    Pid rollpid = new DefaultPid();

    void runAcro(double throttle, double[] desiredAccel, double[] currentAccel){
        pitchpid.setSetPoint(0.0);
        Double pitchOutput = pitchpid.compute(currentAccel[1]);
        Double rollOutput = rollpid.compute(currentAccel[1]);
        double yawOutput = 0; //P_YAW * (currentAccel[0] - desiredAccel[0]);
        Log.d("DELTA-PITCH", "" + pitchOutput);
        Log.d("DELTA-ROLL", "" + rollOutput);

        if(throttle > 5){
            FL = (int)(throttle - rollOutput - pitchOutput - yawOutput);
            BL = (int)(throttle - rollOutput + pitchOutput + yawOutput);
            FR = (int)(throttle + rollOutput - pitchOutput + yawOutput);
            BR = (int)(throttle + rollOutput + pitchOutput - yawOutput);
        } else {
            FL = (int)throttle;
            BL = (int)throttle;
            FR = (int)throttle;
            BR = (int)throttle;
        }
        if(sensorWriter != null && motorWriter != null){
            sensorWriter.write(currentAccel[0] + "," + currentAccel[1] + "," + currentAccel[2]);
            motorWriter.write(FL + "," + BL + "," + FR + "," + BR);
        }
        if(connected){
            sendMotor(FRONT_LEFT, FL);
            sendMotor(REAR_LEFT, BL);
            sendMotor(FRONT_RIGHT, FR);
            sendMotor(REAR_RIGHT, BR);
        }
        Log.d("Acro test", "FL:" + FL + ", BL: " + BL + ", FR:" + FR + ", BR:" + BR);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}
