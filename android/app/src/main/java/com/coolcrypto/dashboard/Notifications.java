package com.coolcrypto.dashboard;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;

import androidx.core.app.NotificationCompat;
import androidx.core.app.ServiceCompat;

/** Notification channels and helpers for the foreground services. */
public final class Notifications {

    public static final String CHANNEL_SERVICES = "services";
    public static final String CHANNEL_TRADING = "trading";
    public static final String CHANNEL_UPDATES = "updates";

    public static final int ID_DJANGO = 1;
    public static final int ID_WIZARD = 2;
    public static final int ID_TRADING = 3;
    public static final int ID_UPDATE = 4;

    private Notifications() {
    }

    public static void createChannels(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return;
        }
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager == null) {
            return;
        }
        // LOW: these are persistent status notifications, not alerts. Anything
        // higher would buzz the phone every time a service restarts.
        manager.createNotificationChannel(new NotificationChannel(
                CHANNEL_SERVICES, "Local services", NotificationManager.IMPORTANCE_LOW));
        manager.createNotificationChannel(new NotificationChannel(
                CHANNEL_TRADING, "Trading", NotificationManager.IMPORTANCE_LOW));
        manager.createNotificationChannel(new NotificationChannel(
                CHANNEL_UPDATES, "Updates", NotificationManager.IMPORTANCE_DEFAULT));
    }

    public static Notification build(Context context, String channel,
                                     String title, String text) {
        Intent open = new Intent(context, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(context, 0, open,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);

        return new NotificationCompat.Builder(context, channel)
                .setContentTitle(title)
                .setContentText(text)
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentIntent(pending)
                .setOngoing(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build();
    }

    /**
     * Post a foreground notification with the right service type.
     *
     * <p>Android 14+ rejects startForeground() without a type that matches the
     * manifest declaration, killing the service outright.
     */
    public static void startForeground(android.app.Service service, int id,
                                       Notification notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ServiceCompat.startForeground(service, id, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            service.startForeground(id, notification);
        }
    }

    public static void update(Context context, int id, String title, String text) {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager != null) {
            manager.notify(id, build(context, channelFor(id), title, text));
        }
    }

    /** One-shot notification (update installed); dismissible, unlike the rest. */
    public static void notifyOnce(Context context, int id, String title, String text) {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager == null) {
            return;
        }
        Notification n = new NotificationCompat.Builder(context, CHANNEL_UPDATES)
                .setContentTitle(title)
                .setContentText(text)
                .setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setAutoCancel(true)
                .build();
        manager.notify(id, n);
    }

    private static String channelFor(int id) {
        return id == ID_TRADING ? CHANNEL_TRADING
                : id == ID_UPDATE ? CHANNEL_UPDATES : CHANNEL_SERVICES;
    }
}
