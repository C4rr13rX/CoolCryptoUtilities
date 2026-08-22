package com.coolcrypto.dashboard;

import android.app.Application;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

/**
 * Starts the embedded CPython runtime once per process.
 *
 * <p>Doing it here rather than in a service means every entry point -- the
 * activity, a scheduled job waking the app with no UI, the update worker --
 * finds Python already initialised.
 */
public class DashboardApplication extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        if (!Python.isStarted()) {
            // Cheap: this only sets up the interpreter. The expensive part is
            // importing Django, which lambda_runtime defers until an actual
            // invocation needs it.
            Python.start(new AndroidPlatform(this));
        }
        Notifications.createChannels(this);
    }
}
