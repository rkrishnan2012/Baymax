package com.baymax;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialPort;
import com.hoho.android.usbserial.driver.UsbSerialProber;
import com.hoho.android.usbserial.util.SerialInputOutputManager;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements View.OnClickListener,
        SeekBar.OnSeekBarChangeListener {

    private UsbManager mUsbManager;
    private SerialInputOutputManager mSerialIoManager;
    UsbSerialPort mPort;

    private TextView mStatus;
    private TextView mDumpText;
    private ScrollView mScrollView;

    private static final int TIMEOUT = 500;
    private static final int CMD_FLAP = 10;
    private static final int CMD_CALIBRATE = 11;
    private static final int CMD_INDIVIDUAL = 12;

    private static final String TAG = "Baymax";
    private static final String ACTION_USB_PERMISSION = "com.hoho.android.usbserial.examples.USB_PERMISSION";

    private final ExecutorService mExecutor = Executors.newSingleThreadExecutor();

    private void updateReceivedData(final byte[] data) {
        mDumpText.post(new Runnable() {
            @Override
            public void run() {
                mDumpText.append(new String(data));
            }
        });
    }

    private void onConnected(){
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
        findViewById(R.id.calibrate).setOnClickListener(this);
        findViewById(R.id.individual).setOnClickListener(this);

        ((SeekBar)findViewById((R.id.motor1))).setOnSeekBarChangeListener(this);
        ((SeekBar)findViewById((R.id.motor2))).setOnSeekBarChangeListener(this);
        ((SeekBar)findViewById((R.id.motor3))).setOnSeekBarChangeListener(this);
        ((SeekBar)findViewById((R.id.motor4))).setOnSeekBarChangeListener(this);

        mUsbManager = (UsbManager) getSystemService(Context.USB_SERVICE);
    }

    @Override
    protected void onPause() {
        super.onPause();
        unregisterReceiver(mUsbReceiver);
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

    @Override
    public void onClick(View v) {
        try{
            int value = 0;
            switch(v.getId()){
                case R.id.flap:
                    value = CMD_FLAP;
                    break;
                case R.id.calibrate:
                    value = CMD_CALIBRATE;
                    break;
                case R.id.individual:
                    value = CMD_INDIVIDUAL;
                    break;
            }
            byte[] ary = new byte[4];
            ary[0] = (byte) ((value >> 24) & 0xFF);
            ary[1] = (byte) ((value >> 16) & 0xFF);
            ary[2] = (byte) ((value >> 8) & 0xFF);
            ary[3] = (byte) (value & 0xFF);
            mPort.write(ary, TIMEOUT);
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
            }
        } catch (IOException ignored){}
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {}

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {}
}
