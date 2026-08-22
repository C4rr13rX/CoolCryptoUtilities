package com.coolcrypto.dashboard.services;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.coolcrypto.dashboard.Notifications;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Hosts the local API Gateway on 127.0.0.1.
 *
 * <p>This is not a web server in the usual sense. It accepts a socket, converts
 * each request into a Lambda event, and invokes the same handlers that run in
 * AWS ({@code serverless/handlers/}). Between requests nothing runs -- which is
 * the whole point of the design on a device where idle costs battery rather
 * than dollars.
 *
 * <p>Foreground rather than background: the WebView is useless the instant this
 * dies, and Android aggressively kills background processes holding a listening
 * socket. The notification is the honest price of that.
 *
 * <p>It deliberately does <em>not</em> run scheduled work. That belongs to
 * {@link ScheduledJobService}, so the OS can batch and defer it.
 */
public class DjangoService extends Service {

    private static final String TAG = "DjangoService";
    public static final int PORT = 8765;

    private static volatile boolean sRunning = false;
    private final ExecutorService mExecutor = Executors.newSingleThreadExecutor();

    public static boolean isRunning() {
        return sRunning;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Notification n = Notifications.build(
                this,
                Notifications.CHANNEL_SERVICES,
                "Dashboard",
                "Starting…");
        // startForeground before any slow work: Android gives a service only a
        // few seconds to post its notification before killing it outright.
        Notifications.startForeground(this, Notifications.ID_DJANGO, n);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (sRunning) {
            return START_STICKY;
        }
        // Off the main thread: Django's first import costs seconds (30 apps,
        // pandas, numpy) and doing it here would ANR before the UI appeared.
        mExecutor.execute(this::startPython);
        // START_STICKY: if the OS reclaims us under memory pressure, come back.
        return START_STICKY;
    }

    private void startPython() {
        try {
            Python py = Python.getInstance();
            PyObject bootstrap = py.getModule("android_bootstrap");
            PyObject result = bootstrap.callAttr(
                    "start_server",
                    getFilesDir().getAbsolutePath(),
                    PORT);

            String status = result.callAttr("get", "status").toString();
            if ("running".equals(status)) {
                sRunning = true;
                // First launch has no database; migrate before the WebView
                // issues its first request, or every panel 500s.
                bootstrap.callAttr("migrate");
                // Pre-import the request-path handlers so the user's first tap
                // costs ~20 ms instead of the ~3 s cold start.
                bootstrap.callAttr("warm_handlers");
                Notifications.update(this, Notifications.ID_DJANGO,
                        "Dashboard", "Ready on 127.0.0.1:" + PORT);
                Log.i(TAG, "django up on " + PORT);
            } else {
                String error = String.valueOf(result.callAttr("get", "error"));
                // Surface it: a silent failure leaves the user staring at a
                // WebView that never loads, with nothing to act on.
                Notifications.update(this, Notifications.ID_DJANGO,
                        "Dashboard failed", error);
                Log.e(TAG, "django failed: " + error);
            }
        } catch (Throwable t) {
            Log.e(TAG, "python bootstrap threw", t);
            Notifications.update(this, Notifications.ID_DJANGO,
                    "Dashboard failed", String.valueOf(t.getMessage()));
        }
    }

    @Override
    public void onDestroy() {
        try {
            Python.getInstance().getModule("android_bootstrap").callAttr("stop_server");
        } catch (Throwable t) {
            Log.w(TAG, "stop_server failed", t);
        }
        sRunning = false;
        mExecutor.shutdownNow();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;   // started service only; the UI talks to it over HTTP
    }
}
