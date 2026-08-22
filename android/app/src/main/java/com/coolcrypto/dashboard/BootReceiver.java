package com.coolcrypto.dashboard;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import com.coolcrypto.dashboard.services.ScheduledJobService;

/**
 * Re-registers scheduled work after a reboot or an app update.
 *
 * <p>Deliberately does NOT start the gateway or the wizard node. Those exist to
 * serve the UI, and starting them when the user has not opened the app would
 * burn battery for nothing -- the opposite of the point of this architecture.
 * Scheduled jobs are {@code setPersisted}, so this is belt-and-braces for the
 * update case where the OS drops them.
 */
public class BootReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        String action = intent != null ? intent.getAction() : null;
        if (Intent.ACTION_BOOT_COMPLETED.equals(action)
                || Intent.ACTION_MY_PACKAGE_REPLACED.equals(action)) {
            ScheduledJobService.scheduleAll(context);
            WizardUpdateWorker.schedule(context);
        }
    }
}
