# Source this file before running any android tooling.
#   source android_env.sh
# (You can append a `source /tmp2/$USER/DRL_final_project/android_env.sh` line to
# your ~/.bashrc so it loads automatically in new terminals.)

export ANDROID_HOME=/tmp2/$USER/DRL_final/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export ANDROID_USER_HOME=/tmp2/$USER/DRL_final/.android
export ANDROID_AVD_HOME=/tmp2/$USER/DRL_final/.android/avd
export ANDROID_EMULATOR_HOME=/tmp2/$USER/DRL_final/.android
# Use the SDK adb (newer) ahead of /sbin/adb (Arch system 35.0.2).
export PATH=$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/emulator:$PATH

# Per-user adb daemon: avoid colliding with other students on ws10 who share the
# default port 5037. Picked from the user's UID so different users on the same
# host don't stomp on each other.
export ANDROID_ADB_SERVER_PORT=$((5137 + ($(id -u) % 800)))

mkdir -p "$ANDROID_USER_HOME/cache" "$ANDROID_AVD_HOME"
