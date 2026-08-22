# Chaquopy reaches into these by reflection from native code; obfuscating or
# stripping them breaks the Python bridge in ways that only show at runtime.
-keep class com.chaquo.python.** { *; }
-keep class com.coolcrypto.dashboard.** { *; }
