## DRL Final Project Env Setup on CSIE Workstation

### 1. Create directories
```bash
mkdir -p /tmp2/$USER/DRL_final/android-sdk/cmdline-tools
cd /tmp2/$USER/DRL_final/android-sdk/cmdline-tools
```

### 2. Download & unzip command line tools
Go to https://developer.android.com/studio/index.html#command-line-tools-only and find the latest Linux URL, then:
```bash
wget https://dl.google.com/android/repository/commandlinetools-linux-14742923_latest.zip
unzip commandlinetools-linux-14742923_latest.zip
mv cmdline-tools latest
```

### 3. Set environment variables
```bash
export ANDROID_HOME=/tmp2/$USER/DRL_final/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export ANDROID_USER_HOME=/tmp2/$USER/DRL_final/.android
export ANDROID_AVD_HOME=/tmp2/$USER/DRL_final/.android/avd
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator
export ANDROID_EMULATOR_HOME=/tmp2/$USER/DRL_final/.android
```

### 4. Create cache and AVD directories
```bash
mkdir -p $ANDROID_USER_HOME/cache
mkdir -p $ANDROID_AVD_HOME
```

### 5. Accept licenses
```bash
sdkmanager --licenses
```
Press `y` for all prompts.

### 6. Install SDK packages (1 mins)
```bash
sdkmanager \
  "platform-tools" \
  "platforms;android-31" \
  "system-images;android-31;google_apis;x86_64" \
  "emulator"
```

### 7. Create AVD (Pixel 5, API 31)
```bash
avdmanager create avd \
  -n pixel5_api31 \
  -k "system-images;android-31;google_apis;x86_64" \
  -d "pixel_5"
```

### 8. First launch — cold boot (one time only, takes 5 mins)
```bash
emulator -avd pixel5_api31 \
  -no-window \
  -no-audio \
  -no-boot-anim \
  -gpu swiftshader_indirect \
  -no-metrics &
```

跑這個後直接開另一個 terminal 跑下一個步驟，這步驟可能會出現 warning 或 error，但沒關係

### 9. Install your APK

open another terminal, run **step3 Set environment variables** first then

```bash
until adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1" && \
      adb shell service check package 2>/dev/null | grep -q "found"; do
  echo "Waiting for package manager..."; sleep 5
done
echo "Ready! Installing..."
adb install ~/Desktop/Bouncy\ Basketball_3.2.1_APKPure.apk
```


### 10. Save snapshot (one time only, after APK installed)
```bash
adb emu avd snapshot save clean_boot
```
Snapshot is saved to:
`$ANDROID_AVD_HOME/pixel5_api31.avd/snapshots/clean_boot/`

Kill the emulator after saving:
```bash
adb emu kill
```

### 11. Subsequent launches — fast boot from snapshot (~10 sec)
```bash
emulator -avd pixel5_api31 \
  -no-window \
  -no-audio \
  -no-boot-anim \
  -gpu swiftshader_indirect \
  -no-metrics \
  -snapshot clean_boot &
```

Wait for boot:
```bash
until adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do
  echo "Waiting for boot..."; sleep 3
done
echo "Device ready!"
```


## 即時查看遊戲畫面 for debug tsai
### 13. 將寫好的環境腳本複製到你的家目錄底下
`cp /tmp2/b12902140/stream_env.py ~/`

### 14. 啟動 Android 串流伺服器 
確保已經執行 11. 開啟 emulator
```
# 1. 重新載入環境變數
export ANDROID_HOME=/tmp2/$USER/DRL_final/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator

# 2. 啟動 DRL 監控伺服器
cd ~/
python stream_env.py
```

### 15. 在自己電腦建立 SSH Tunnel
請在你自己的個人電腦開啟一個全新視窗，並複製貼上以下指令
`ssh -L 5000:localhost:5000 <你的學號>@meow1.csie.ntu.edu.tw`

### 16. 打開自己電腦的瀏覽器
`http://localhost:5000`
紅色的十字準心閃爍，那是 Python 正在對模擬器下達點擊指令的視覺化

![image](https://hackmd.io/_uploads/SyoU-ec1Ge.png)


