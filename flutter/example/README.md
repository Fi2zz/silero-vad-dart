# Silero VAD Flutter Example

A comprehensive example app demonstrating the `silero_vad_flutter` package for Voice Activity Detection (VAD) in Flutter.

## Features

This example app showcases two modes of operation:

### 1. File VAD
- Pick a WAV audio file from your device
- Run Voice Activity Detection on the entire file
- View detected speech segments with timestamps
- See confidence scores for each segment

### 2. Real-time VAD
- Start/stop microphone recording
- Stream audio chunks through the VAD in real-time
- See live speech detection status and timestamps
- Uses the `VadIterator` for stateful, streaming processing

## Prerequisites

Before running the example app:

1. **Download the Silero VAD ONNX model:**
   ```bash
   bash tool/download_model.sh
   ```
   This downloads the model to `assets/models/silero_vad.onnx`.

2. **Install Flutter dependencies:**
   ```bash
   flutter pub get
   ```

3. **Platform-specific setup:**

### Android
Add the following permissions to `android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
```

### iOS
Add the following to `ios/Runner/Info.plist`:
```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs microphone access for real-time voice activity detection.</string>
```

### macOS
Add the following to `macos/Runner/DebugProfile.entitlements` and `macos/Runner/Release.entitlements`:
```xml
<key>com.apple.security.device.microphone</key>
<true/>
```

## Running the App

```bash
flutter run
```

Or for a specific platform:

```bash
flutter run -d android
flutter run -d ios
flutter run -d macos
flutter run -d windows
flutter run -d linux
flutter run -d chrome
```

## Project Structure

```
lib/
  main.dart          - Main app entry point with tab-based UI
```

## Notes

- The File VAD tab works on all platforms (desktop, mobile, web).
- The Real-time VAD tab requires a microphone and works on mobile and desktop platforms.
- On web, real-time microphone access may be limited by browser permissions.
