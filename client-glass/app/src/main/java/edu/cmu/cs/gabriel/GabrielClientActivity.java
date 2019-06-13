package edu.cmu.cs.gabriel;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.glass.view.WindowUtils;

import edu.cmu.cs.gabriel.network.AccStreamingThread;
import edu.cmu.cs.gabriel.network.NetworkProtocol;
import edu.cmu.cs.gabriel.network.ResultReceivingThread;
import edu.cmu.cs.gabriel.network.VideoStreamingThread;
import edu.cmu.cs.gabriel.token.ReceivedPacketInfo;
import edu.cmu.cs.gabriel.token.TokenController;

public class GabrielClientActivity extends Activity implements TextToSpeech.OnInitListener, SensorEventListener{

    private static final String LOG_TAG = "Main";
    private static final int REQ_CODE_SPEECH_INPUT = 2345434;

    // major components for streaming sensor data and receiving information
    private VideoStreamingThread videoStreamingThread = null;
    private AccStreamingThread accStreamingThread = null;
    private ResultReceivingThread resultThread = null;
    private TokenController tokenController = null;

    private boolean isRunning = false;
    private boolean isFirstExperiment = true;
    private CameraPreview preview = null;

    private SensorManager sensorManager = null;
    private Sensor sensorAcc = null;
    private TextToSpeech tts = null;

    private ReceivedPacketInfo receivedPacketInfo = null;
    private ImageButton btnSpeak;

    private boolean speechVerifying =false;
    private String potentialGaugeColor=null;
    public static volatile String gaugeColor=null;
    private ArrayList<String> colorOptions=new ArrayList<String>();

    private boolean onGlass=false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.v(LOG_TAG, "++onCreate");
        super.onCreate(savedInstanceState);
        // if running on glass
        if ("Google".equalsIgnoreCase(Build.MANUFACTURER) && Build.MODEL.startsWith("Glass")){
            onGlass=true;
            getWindow().requestFeature(WindowUtils.FEATURE_VOICE_COMMANDS);
        }

        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED+
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON+
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        colorOptions.add("blue");
        colorOptions.add("green");
        colorOptions.add("fuchsia");
        colorOptions.add("gold");
        btnSpeak=(ImageButton)findViewById(R.id.btnSpeak);
        btnSpeak.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        promptSpeechInput();
                    }
                }
        );
    }

    @Override
    public boolean onCreatePanelMenu(int featureId, Menu menu) {
        if (featureId == WindowUtils.FEATURE_VOICE_COMMANDS) {
            getMenuInflater().inflate(R.menu.main, menu);
            return true;
        }
        // Pass through to super to setup touch menu.
        return super.onCreatePanelMenu(featureId, menu);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onMenuItemSelected(int featureId, MenuItem item) {
        if (featureId == WindowUtils.FEATURE_VOICE_COMMANDS) {
            String userInput="";
            switch (item.getItemId()) {
                case R.id.gold_menu_item:
                    userInput=getResources().getString(R.string.gold_color);
                    break;
                case R.id.fuchsia_menu_item:
                    userInput=getResources().getString(R.string.fuchsia_color);
                    break;
                case R.id.green_menu_item:
                    userInput=getResources().getString(R.string.green_color);
                    break;
                case R.id.blue_menu_item:
                    userInput=getResources().getString(R.string.blue_color);
                    break;
                default:
                    return true;
            }
            Log.d(LOG_TAG, "user said " + gaugeColor);
            gaugeColor=userInput;
            return true;
        }
        // Good practice to pass through to super if not handled
        return super.onMenuItemSelected(featureId, item);
    }

    // void input related
    private void promptSpeechInput() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US");
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 10);
        intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "Listening");
//        intent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true);
        try{
            startActivityForResult(intent, REQ_CODE_SPEECH_INPUT);
        } catch (ActivityNotFoundException a ){
            Toast.makeText(this, "speech not supported", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode){
            case REQ_CODE_SPEECH_INPUT: {
                if (resultCode == RESULT_OK && null != data){
                    String result="";
                    ArrayList<String> candidates=data.getStringArrayListExtra(
                            RecognizerIntent.EXTRA_RESULTS);
                    Log.d(LOG_TAG,"reconigzed candidates: " + candidates.toString());
                    for (String candidate:candidates){
                        Log.d(LOG_TAG, candidate);
                        if (colorOptions.contains(candidate) || candidate.equals("yes")){
                            result=candidate;
                            break;
                        }
                    }

                    Log.i(LOG_TAG, "recognized: " + result);
                    if (!speechVerifying){
                        if (colorOptions.contains(result)){
                            speechVerifying=true;
                            potentialGaugeColor=result;
                            //tts repeat
                            tts.speak("Did you say " + result, TextToSpeech.QUEUE_FLUSH, null);
                            SystemClock.sleep(500);
                            //promopt speech input
                            promptSpeechInput();
                        } else {
                            tts.speak("I didn't understand what you said", TextToSpeech.QUEUE_FLUSH, null);
                        }
                    } else {
                        //verifying speech
                        if (!result.equals("yes")){
                            tts.speak("Please tell me what the color it is again",
                                    TextToSpeech.QUEUE_FLUSH,
                                    null);
                        } else {
                            gaugeColor=potentialGaugeColor;
                        }
                        speechVerifying=false;
                    }
                }
                break;
            }
        }
    }


    @Override
    protected void onResume() {
        Log.v(LOG_TAG, "++onResume");
        super.onResume();
        if (videoStreamingThread==null){
            initOnce();
            if (Const.IS_EXPERIMENT) { // experiment mode
                runExperiments();
            } else { // demo mode
                initPerRun(Const.SERVER_IP, Const.TOKEN_SIZE, null);
            }
        }
        if (preview==null){
            initOnce();
        }
    }

    @Override
    protected void onPause() {
        Log.v(LOG_TAG, "++onPause");
        this.terminate(false);
        super.onPause();
    }

    @Override
    protected void onStop() {
        Log.v(LOG_TAG, "++onStop");
        this.terminate(true);
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        Log.v(LOG_TAG, "++onDestroy");
        super.onDestroy();
    }

    /**
     * Does initialization for the entire application. Called only once even for multiple experiments.
     */
    private void initOnce() {
        Log.v(LOG_TAG, "++initOnce");
        if (preview == null){
            Log.v(LOG_TAG, "++initCamera");
            preview = (CameraPreview) findViewById(R.id.camera_preview);
            preview.checkCamera();
            if (!preview.cameraInited){
                preview.initCamera(preview.getHolder());
            }
            preview.setPreviewCallback(previewCallback);
        }

        Const.ROOT_DIR.mkdirs();
        Const.EXP_DIR.mkdirs();
        
        // TextToSpeech.OnInitListener
        if (tts == null) {
            tts = new TextToSpeech(this, this);
        }
        
        // IMU sensors
        if (sensorManager == null) {
            sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
            sensorAcc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            sensorManager.registerListener(this, sensorAcc, SensorManager.SENSOR_DELAY_NORMAL);
        }
        
        isRunning = true;
    }
    
    /**
     * Does initialization before each run (connecting to a specific server).
     * Called once before each experiment.
     */
    private void initPerRun(String serverIP, int tokenSize, File latencyFile) {
        Log.v(LOG_TAG, "++initPerRun");
        if (tokenController != null){
            tokenController.close();
        }
        if ((videoStreamingThread != null) && (videoStreamingThread.isAlive())) {
            videoStreamingThread.stopStreaming();
            videoStreamingThread = null;
        }
        if ((accStreamingThread != null) && (accStreamingThread.isAlive())) {
            accStreamingThread.stopStreaming();
            accStreamingThread = null;
        }
        if ((resultThread != null) && (resultThread.isAlive())) {
            resultThread.close();
            resultThread = null;
        }

        if (Const.IS_EXPERIMENT) {
            if (isFirstExperiment) {
                isFirstExperiment = false;
            } else {
                try {
                    Thread.sleep(20*1000);
                } catch (InterruptedException e) {}
            }
        }

        tokenController = new TokenController(tokenSize, latencyFile);
        resultThread = new ResultReceivingThread(serverIP, Const.RESULT_RECEIVING_PORT, returnMsgHandler);
        resultThread.start();

        videoStreamingThread = new VideoStreamingThread(serverIP, Const.VIDEO_STREAM_PORT, returnMsgHandler, tokenController);
        videoStreamingThread.start();
    }

    /**
     * Runs a set of experiments with different server IPs and token numbers.
     * IP list and token sizes are defined in the Const file.
     */
    private void runExperiments(){
        final Timer startTimer = new Timer();
        TimerTask autoStart = new TimerTask(){
            int ipIndex = 0;
            int tokenIndex = 0;
            @Override
            public void run() {
                GabrielClientActivity.this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        // end condition
                        if ((ipIndex == Const.SERVER_IP_LIST.length) || (tokenIndex == Const.TOKEN_SIZE_LIST.length)) {
                            Log.d(LOG_TAG, "Finish all experiemets");
                            startTimer.cancel();
                            terminate(true);
                            return;
                        }

                        // make a new configuration
                        String serverIP = Const.SERVER_IP_LIST[ipIndex];
                        int tokenSize = Const.TOKEN_SIZE_LIST[tokenIndex];
                        File latencyFile = new File (Const.EXP_DIR.getAbsolutePath() + File.separator + 
                                "latency-" + serverIP + "-" + tokenSize + ".txt");
                        Log.i(LOG_TAG, "Start new experiment - IP: " + serverIP +"\tToken: " + tokenSize);

                        // run the experiment
                        initPerRun(serverIP, tokenSize, latencyFile);

                        // move to the next experiment
                        tokenIndex++;
                        if (tokenIndex == Const.TOKEN_SIZE_LIST.length){
                            tokenIndex = 0;
                            ipIndex++;
                        }
                    }
                });
            }
        };

        // run 5 minutes for each experiment
        startTimer.schedule(autoStart, 1000, 5*60*1000);
    }

    private PreviewCallback previewCallback = new PreviewCallback() {
        // called whenever a new frame is captured
        public void onPreviewFrame(byte[] frame, Camera mCamera) {
            if (isRunning) {
                Camera.Parameters parameters = mCamera.getParameters();
                if (videoStreamingThread != null){
                    videoStreamingThread.push(frame, parameters);
                }
            }
        }
    };

    /**
     * Notifies token controller that some response is back
     */
    private void notifyToken() {
        Message msg = Message.obtain();
        msg.what = NetworkProtocol.NETWORK_RET_TOKEN;
        receivedPacketInfo.setGuidanceDoneTime(System.currentTimeMillis());
        msg.obj = receivedPacketInfo;
        try {
            tokenController.tokenHandler.sendMessage(msg);
        } catch (NullPointerException e) {
            // might happen because token controller might have been terminated
        }
    }
    
    /**
     * Handles messages passed from streaming threads and result receiving threads.
     */
    private Handler returnMsgHandler = new Handler() {
        public void handleMessage(Message msg) {
//            if (msg.what == NetworkProtocol.NETWORK_RET_FAILED) {
//                terminate(false);
//            }
            if (msg.what == NetworkProtocol.NETWORK_RET_MESSAGE) {
                receivedPacketInfo = (ReceivedPacketInfo) msg.obj;
                receivedPacketInfo.setMsgRecvTime(System.currentTimeMillis());
            }
            if (msg.what == NetworkProtocol.NETWORK_RET_SPEECH) {
                String ttsMessage = (String) msg.obj;

//                if (tts != null && !tts.isSpeaking()){
                if (tts != null ){
                    Log.d(LOG_TAG, "tts to be played: " + ttsMessage);
                    tts.setSpeechRate(1.5f);
                    String[] splitMSGs = ttsMessage.split("\\.");
                    HashMap<String, String> map = new HashMap<String, String>();
                    map.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "unique");

                    if (splitMSGs.length == 1)
                        tts.speak(splitMSGs[0].toString().trim(), TextToSpeech.QUEUE_FLUSH, map); // the only sentence
                    else {
                        tts.speak(splitMSGs[0].toString().trim(), TextToSpeech.QUEUE_FLUSH, null); // the first sentence
                        for (int i = 1; i < splitMSGs.length - 1; i++) {
                            tts.playSilence(350, TextToSpeech.QUEUE_ADD, null); // add pause for every period
                            tts.speak(splitMSGs[i].toString().trim(),TextToSpeech.QUEUE_ADD, null);
                        }
                        tts.playSilence(350, TextToSpeech.QUEUE_ADD, null);
                        tts.speak(splitMSGs[splitMSGs.length - 1].toString().trim(),TextToSpeech.QUEUE_ADD, map); // the last sentence
                    }
                }
            }
            if (msg.what == NetworkProtocol.NETWORK_RET_IMAGE || msg.what == NetworkProtocol.NETWORK_RET_ANIMATION) {
                Bitmap feedbackImg = (Bitmap) msg.obj;
                ImageView img = (ImageView) findViewById(R.id.guidance_image);
                img.setImageBitmap(feedbackImg);
            }
            if (msg.what == NetworkProtocol.NETWORK_RET_DONE) {
                notifyToken();
            }
        }
    };

    /**
     * Terminates all services.
     */
    private void terminate(boolean stopThreads) {
        Log.v(LOG_TAG, "++terminate. ");
        
        isRunning = false;

        if (stopThreads){
            if ((resultThread != null) && (resultThread.isAlive())) {
                resultThread.close();
                resultThread = null;
            }
            if ((videoStreamingThread != null) && (videoStreamingThread.isAlive())) {
                videoStreamingThread.stopStreaming();
                videoStreamingThread = null;
            }
            if ((accStreamingThread != null) && (accStreamingThread.isAlive())) {
                accStreamingThread.stopStreaming();
                accStreamingThread = null;
            }
            if (tokenController != null){
                tokenController.close();
                tokenController = null;
            }

            if (tts != null) {
                tts.stop();
                tts.shutdown();
                tts = null;
            }
        }

        if (preview != null) {
            preview.setPreviewCallback(null);
            preview.close();
            preview = null;
        }

        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
            sensorManager = null;
            sensorAcc = null;
        }
    }

    /**************** SensorEventListener ***********************/
    // TODO: test accelerometer streaming
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() != Sensor.TYPE_ACCELEROMETER)
            return;
        if (accStreamingThread != null) {
//          accStreamingThread.push(event.values);
        }
        // Log.d(LOG_TAG, "acc_x : " + mSensorX + "\tacc_y : " + mSensorY);
    }
    /**************** End of SensorEventListener ****************/
    
    /**************** TextToSpeech.OnInitListener ***************/
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            if (tts == null) {
                tts = new TextToSpeech(this, this);
            }
            int result = tts.setLanguage(Locale.US);
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(LOG_TAG, "Language is not available.");
            }
            int listenerResult = tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                @Override
                public void onDone(String utteranceId) {
                    Log.v(LOG_TAG,"progress on Done " + utteranceId);
//                  notifyToken();
                }
                @Override
                public void onError(String utteranceId) {
                    Log.v(LOG_TAG,"progress on Error " + utteranceId);
                }
                @Override
                public void onStart(String utteranceId) {
                    Log.v(LOG_TAG,"progress on Start " + utteranceId);
                }
            });
            if (listenerResult != TextToSpeech.SUCCESS) {
                Log.e(LOG_TAG, "failed to add utterance progress listener");
            }
        } else {
            // Initialization failed.
            Log.e(LOG_TAG, "Could not initialize TextToSpeech.");
        }
    }
    /**************** End of TextToSpeech.OnInitListener ********/
}
