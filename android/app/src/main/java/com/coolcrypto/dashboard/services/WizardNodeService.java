package com.coolcrypto.dashboard.services;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import com.coolcrypto.dashboard.Notifications;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs the W1z4rDV1510n Rust node as a native process on :8090.
 *
 * <p>The node is a real executable, not a JNI library, so it is shipped inside
 * {@code jniLibs} as {@code libw1z4rd_node.so}. That naming is not cosmetic:
 * Android only extracts and grants execute permission to files matching
 * {@code lib*.so} in the native library directory. Anything dropped in assets
 * would land on a {@code noexec} mount and fail with "Permission denied".
 *
 * <p>Separate from {@link DjangoService} so the node can be restarted or
 * hot-swapped by the C4rr13rX updater without bouncing the web UI.
 */
public class WizardNodeService extends Service {

    private static final String TAG = "WizardNode";
    public static final int PORT = 8090;
    /** Set by {@code WizardUpdateWorker} after it stages a new binary. */
    public static final String ACTION_RESTART = "com.coolcrypto.dashboard.RESTART_NODE";

    private static volatile boolean sRunning = false;
    private Process mProcess;
    private Thread mLogPump;

    public static boolean isRunning() {
        return sRunning;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Notification n = Notifications.build(this, Notifications.CHANNEL_SERVICES,
                "Wizard node", "Starting…");
        Notifications.startForeground(this, Notifications.ID_WIZARD, n);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_RESTART.equals(intent.getAction())) {
            stopNode();
        }
        if (mProcess != null && mProcess.isAlive()) {
            return START_STICKY;
        }
        new Thread(this::startNode, "wizard-start").start();
        return START_STICKY;
    }

    private void startNode() {
        try {
            File binary = resolveBinary();
            if (binary == null) {
                Notifications.update(this, Notifications.ID_WIZARD,
                        "Wizard node unavailable",
                        "No node binary for this device");
                Log.w(TAG, "no wizard node binary present");
                return;
            }

            File home = new File(getFilesDir(), "wizard");
            if (!home.exists() && !home.mkdirs()) {
                Log.w(TAG, "could not create wizard home");
            }
            ensureConfig(home);

            List<String> cmd = new ArrayList<>();
            cmd.add(binary.getAbsolutePath());
            cmd.add("--config");
            cmd.add(new File(home, "node_config.json").getAbsolutePath());
            cmd.add("api");
            cmd.add("--addr");
            // Loopback only. The node has no authentication of its own, so
            // binding 0.0.0.0 would expose the brain API to the local network.
            cmd.add("127.0.0.1:" + PORT);

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.directory(home);
            pb.redirectErrorStream(true);
            mProcess = pb.start();
            sRunning = true;

            Notifications.update(this, Notifications.ID_WIZARD,
                    "Wizard node", "Listening on 127.0.0.1:" + PORT);
            Log.i(TAG, "started " + binary.getName());

            pumpLogs(mProcess.getInputStream());

            int code = mProcess.waitFor();
            sRunning = false;
            Log.w(TAG, "wizard node exited with " + code);
            Notifications.update(this, Notifications.ID_WIZARD,
                    "Wizard node stopped", "Exit code " + code);
        } catch (Throwable t) {
            sRunning = false;
            Log.e(TAG, "wizard node failed", t);
            Notifications.update(this, Notifications.ID_WIZARD,
                    "Wizard node failed", String.valueOf(t.getMessage()));
        }
    }

    /**
     * Prefer a binary staged by the updater, else the one shipped in the APK.
     *
     * <p>Checking the staged copy first is what makes over-the-air updates from
     * C4rr13rX take effect without reinstalling the app.
     */
    private File resolveBinary() {
        File staged = new File(getFilesDir(), "wizard/bin/w1z4rd_node");
        if (staged.isFile() && staged.canExecute()) {
            return staged;
        }
        File bundled = new File(getApplicationInfo().nativeLibraryDir,
                "libw1z4rd_node.so");
        return bundled.isFile() ? bundled : null;
    }

    /** Write a default config on first run; never clobber an existing one. */
    private void ensureConfig(File home) throws IOException {
        File config = new File(home, "node_config.json");
        if (config.exists()) {
            return;
        }
        String json = "{\n"
                + "  \"data_dir\": \"" + new File(home, "data").getAbsolutePath() + "\",\n"
                + "  \"listen\": \"127.0.0.1:" + PORT + "\",\n"
                + "  \"p2p_enabled\": false\n"
                + "}\n";
        try (FileOutputStream out = new FileOutputStream(config)) {
            out.write(json.getBytes("UTF-8"));
        }
    }

    /**
     * Drain the node's stdout into logcat.
     *
     * <p>Not optional: a child process whose pipe buffer fills blocks forever,
     * so this thread is what keeps the node running at all.
     */
    private void pumpLogs(InputStream stream) {
        mLogPump = new Thread(() -> {
            try (BufferedReader reader =
                         new BufferedReader(new InputStreamReader(stream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    Log.d(TAG, line);
                }
            } catch (IOException ignored) {
                // Expected on shutdown when the stream closes.
            }
        }, "wizard-logs");
        mLogPump.setDaemon(true);
        mLogPump.start();
    }

    private void stopNode() {
        if (mProcess != null) {
            mProcess.destroy();
            try {
                // Give it a moment to flush state, then insist.
                if (!mProcess.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)) {
                    mProcess.destroyForcibly();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            mProcess = null;
        }
        sRunning = false;
    }

    @Override
    public void onDestroy() {
        stopNode();
        if (mLogPump != null) {
            mLogPump.interrupt();
        }
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
