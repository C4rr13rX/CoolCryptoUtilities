package com.coolcrypto.dashboard;

import android.content.Context;
import android.content.Intent;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.Constraints;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.NetworkType;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.coolcrypto.dashboard.services.WizardNodeService;

import java.util.concurrent.TimeUnit;

/**
 * Pulls wizard-node updates published by the C4rr13rX repo and installs them.
 *
 * <p>WorkManager rather than a thread or AlarmManager, because this is exactly
 * the case it exists for: the check must survive reboots and app death, must
 * not run when the device is offline, and must respect Doze instead of
 * fighting it. A bare thread would be killed and never rescheduled.
 *
 * <p>The actual work is Python ({@code wizard_updater.py}) so the download,
 * signature check and staging logic is shared with the desktop rather than
 * reimplemented in Java. This class is scheduling and process control only.
 */
public class WizardUpdateWorker extends Worker {

    private static final String TAG = "WizardUpdate";
    private static final String WORK_NAME = "wizard-node-update";

    public WizardUpdateWorker(@NonNull Context context,
                              @NonNull WorkerParameters params) {
        super(context, params);
    }

    /**
     * Register the periodic check.
     *
     * <p>Six hours: node builds land occasionally, and each check costs a
     * network round trip. KEEP rather than REPLACE so an in-flight update is
     * not cancelled every time the app starts.
     */
    public static void schedule(Context context) {
        Constraints constraints = new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                // A node binary is tens of megabytes; downloading it on a
                // metered connection without asking is not ours to decide.
                .setRequiresBatteryNotLow(true)
                .build();

        PeriodicWorkRequest request = new PeriodicWorkRequest.Builder(
                WizardUpdateWorker.class, 6, TimeUnit.HOURS)
                .setConstraints(constraints)
                .addTag(WORK_NAME)
                .build();

        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME, ExistingPeriodicWorkPolicy.KEEP, request);
    }

    @NonNull
    @Override
    public Result doWork() {
        try {
            Python py = Python.getInstance();
            PyObject updater = py.getModule("wizard_updater");
            PyObject result = updater.callAttr(
                    "check_and_stage",
                    getApplicationContext().getFilesDir().getAbsolutePath());

            String status = String.valueOf(result.callAttr("get", "status"));
            Log.i(TAG, "update check -> " + status);

            if ("installed".equals(status)) {
                String version = String.valueOf(result.callAttr("get", "version"));
                Notifications.notifyOnce(getApplicationContext(),
                        Notifications.ID_UPDATE,
                        "Wizard node updated", "Now running " + version);
                // Restart rather than wait for the next boot: a staged binary
                // that never loads is the stale-binary failure this project
                // has already been bitten by (routes present in source,
                // missing in the running process).
                Intent restart = new Intent(getApplicationContext(),
                        WizardNodeService.class);
                restart.setAction(WizardNodeService.ACTION_RESTART);
                getApplicationContext().startForegroundService(restart);
            }

            // "unchanged" and "installed" are both successes. Only a transport
            // or verification failure is worth a retry.
            if ("error".equals(status)) {
                return Result.retry();
            }
            return Result.success();
        } catch (Throwable t) {
            Log.e(TAG, "update check failed", t);
            return Result.retry();
        }
    }
}
