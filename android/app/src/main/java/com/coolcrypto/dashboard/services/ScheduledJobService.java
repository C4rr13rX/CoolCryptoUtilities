package com.coolcrypto.dashboard.services;

import android.app.job.JobInfo;
import android.app.job.JobParameters;
import android.app.job.JobScheduler;
import android.app.job.JobService;
import android.content.ComponentName;
import android.content.Context;
import android.os.PersistableBundle;
import android.util.Log;

import com.chaquo.python.Python;

import java.util.concurrent.TimeUnit;

/**
 * Runs scheduled Lambda invocations. The device-side EventBridge.
 *
 * <p>This is a {@link JobService}, not a foreground service with a loop, and
 * the difference is the entire point of the architecture. A loop would keep the
 * CPU and radio awake for work that happens every three hours. JobScheduler
 * instead:
 *
 * <ul>
 *   <li><b>batches</b> our wakeups with every other app's, so the device wakes
 *       once for many jobs instead of once for ours;</li>
 *   <li>survives reboot without a BootReceiver of its own
 *       ({@code setPersisted});</li>
 *   <li>defers under Doze rather than being killed by it, so a missed job runs
 *       late instead of never.</li>
 * </ul>
 *
 * <p>Nothing runs between invocations. That is what makes this cheap on a
 * phone, exactly as it makes Lambda cheap on a server.
 */
public class ScheduledJobService extends JobService {

    private static final String TAG = "ScheduledJob";
    public static final String KEY_SCHEDULE = "schedule";

    // Stable per-schedule ids so rescheduling replaces rather than duplicates.
    private static final int JOB_AUTO_PIPELINE = 1001;
    private static final int JOB_WEEKLY_BOOTSTRAP = 1002;
    private static final int JOB_GUARDIAN = 1003;

    /**
     * Register every schedule. Intervals mirror the EventBridge rules in
     * {@code serverless/local/deploy_local.sh}, so the phone and AWS run the
     * same jobs at the same cadence.
     */
    public static void scheduleAll(Context context) {
        JobScheduler scheduler = context.getSystemService(JobScheduler.class);
        if (scheduler == null) {
            Log.w(TAG, "no JobScheduler available");
            return;
        }

        // 3 hours: matches rate(180 minutes).
        scheduler.schedule(build(context, JOB_AUTO_PIPELINE, "auto_pipeline",
                TimeUnit.HOURS.toMillis(3), true));

        // 7 days. JobScheduler caps a periodic interval in practice, so this
        // runs "about weekly" -- acceptable for a bootstrap task, and the
        // handler is idempotent.
        scheduler.schedule(build(context, JOB_WEEKLY_BOOTSTRAP, "weekly_bootstrap",
                TimeUnit.DAYS.toMillis(7), true));

        // Guardian health check. 15 minutes is the floor JobScheduler honours
        // for periodic work; asking for less is silently clamped, so we ask
        // for what we will actually get.
        scheduler.schedule(build(context, JOB_GUARDIAN, "guardian",
                TimeUnit.MINUTES.toMillis(15), false));

        Log.i(TAG, "scheduled 3 jobs");
    }

    private static JobInfo build(Context context, int id, String schedule,
                                 long intervalMs, boolean requiresNetwork) {
        PersistableBundle extras = new PersistableBundle();
        extras.putString(KEY_SCHEDULE, schedule);

        JobInfo.Builder builder = new JobInfo.Builder(id,
                new ComponentName(context, ScheduledJobService.class))
                .setPeriodic(intervalMs)
                .setPersisted(true)          // survive reboot
                .setExtras(extras);

        if (requiresNetwork) {
            // A pipeline run that cannot reach the market is wasted battery.
            builder.setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY);
            builder.setRequiresBatteryNotLow(true);
        }
        return builder.build();
    }

    public static void cancelAll(Context context) {
        JobScheduler scheduler = context.getSystemService(JobScheduler.class);
        if (scheduler != null) {
            scheduler.cancel(JOB_AUTO_PIPELINE);
            scheduler.cancel(JOB_WEEKLY_BOOTSTRAP);
            scheduler.cancel(JOB_GUARDIAN);
        }
    }

    @Override
    public boolean onStartJob(JobParameters params) {
        String schedule = params.getExtras() != null
                ? params.getExtras().getString(KEY_SCHEDULE) : null;
        if (schedule == null) {
            return false;   // nothing to do; do not hold the wakelock
        }

        // JobScheduler calls this on the main thread and holds a wakelock
        // until jobFinished(). The handler can take minutes, so it must not
        // run here.
        new Thread(() -> {
            boolean reschedule = false;
            try {
                Python py = Python.getInstance();
                String result;
                if ("guardian".equals(schedule)) {
                    result = py.getModule("android_workers")
                            .callAttr("run_guardian_check").toString();
                } else {
                    result = py.getModule("android_workers")
                            .callAttr("run_scheduled", schedule).toString();
                }
                Log.i(TAG, schedule + " -> " + truncate(result));
                // A failed invocation is retried with backoff rather than
                // waiting a full period; a periodic job that silently fails
                // forever is the worst outcome here.
                reschedule = result.contains("\"status\": \"error\"")
                        || result.contains("\"status\":\"error\"");
            } catch (Throwable t) {
                Log.e(TAG, schedule + " failed", t);
                reschedule = true;
            } finally {
                jobFinished(params, reschedule);
            }
        }, "job-" + schedule).start();

        return true;   // work continues on the background thread
    }

    @Override
    public boolean onStopJob(JobParameters params) {
        // The system reclaimed us mid-run (Doze, memory pressure). Ask for a
        // retry: the handlers are idempotent, so re-running is safe.
        return true;
    }

    private static String truncate(String s) {
        return s.length() > 180 ? s.substring(0, 180) + "…" : s;
    }
}
