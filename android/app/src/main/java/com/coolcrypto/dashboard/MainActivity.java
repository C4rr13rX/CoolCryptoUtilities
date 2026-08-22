package com.coolcrypto.dashboard;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.coolcrypto.dashboard.services.DjangoService;
import com.coolcrypto.dashboard.services.ScheduledJobService;
import com.coolcrypto.dashboard.services.WizardNodeService;

/**
 * Full-screen WebView over the local gateway.
 *
 * <p>The Vue GUI is used exactly as built -- no frontend changes. It talks to
 * {@code baseURL: '/api'}, which resolves against {@code 127.0.0.1:8765} here,
 * so it is same-origin and CSRF/session behaviour is identical to the desktop.
 */
public class MainActivity extends AppCompatActivity {

    private static final int REQ_NOTIFICATIONS = 100;
    private static final String URL = "http://127.0.0.1:" + DjangoService.PORT + "/";

    private WebView mWebView;
    private TextView mStatus;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private int mAttempts = 0;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mWebView = findViewById(R.id.webview);
        mStatus = findViewById(R.id.status);

        WebSettings settings = mWebView.getSettings();
        settings.setJavaScriptEnabled(true);       // the GUI is a Vue SPA
        settings.setDomStorageEnabled(true);       // AllezORM + session token
        settings.setDatabaseEnabled(true);
        // IndexedDB is where the AllezORM local tier lives; without storage the
        // hybrid model has no local half at all.
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        settings.setMediaPlaybackRequiresUserGesture(false);

        mWebView.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView view, WebResourceRequest req,
                                        WebResourceError err) {
                // The gateway may still be importing Django on first launch.
                // Retry rather than showing the WebView's own error page.
                if (req.isForMainFrame()) {
                    scheduleRetry();
                }
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                mStatus.setVisibility(View.GONE);
                mWebView.setVisibility(View.VISIBLE);
            }
        });

        requestNotificationPermission();
        startServices();
        loadWhenReady();
    }

    /**
     * Android 13+ will not show a foreground-service notification without this,
     * and a service the user cannot see is one they cannot stop.
     */
    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                && ContextCompat.checkSelfPermission(this,
                Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    REQ_NOTIFICATIONS);
        }
    }

    private void startServices() {
        ContextCompat.startForegroundService(this,
                new Intent(this, DjangoService.class));
        ContextCompat.startForegroundService(this,
                new Intent(this, WizardNodeService.class));
        // Idempotent: JobScheduler replaces jobs with the same id.
        ScheduledJobService.scheduleAll(this);
        WizardUpdateWorker.schedule(this);
    }

    /**
     * Poll until the gateway answers, then load it.
     *
     * <p>Cold start is a few seconds on first launch (Django, 30 apps, pandas).
     * Loading before it is ready shows an error page the user would have to
     * dismiss themselves.
     */
    private void loadWhenReady() {
        if (DjangoService.isRunning()) {
            mWebView.loadUrl(URL);
            return;
        }
        if (mAttempts++ > 120) {          // ~60 s, then say so plainly
            mStatus.setText(R.string.startup_failed);
            return;
        }
        mHandler.postDelayed(this::loadWhenReady, 500);
    }

    private void scheduleRetry() {
        mHandler.postDelayed(() -> mWebView.loadUrl(URL), 1000);
    }

    @Override
    public void onBackPressed() {
        // In-app navigation first; only leave the app at the SPA's root.
        if (mWebView != null && mWebView.canGoBack()) {
            mWebView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // Declining only costs visibility of the notifications; the services
        // still run, so there is nothing to abort here.
    }
}
