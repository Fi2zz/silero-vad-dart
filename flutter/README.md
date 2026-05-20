# Silero VAD Flutter

A Dart/Flutter port of the [Silero Voice Activity Detection (VAD)](https://github.com/snakers4/silero-vad) library. Uses ONNX Runtime for efficient inference on mobile, desktop, and web platforms.

[![CI](https://github.com/Fi2zz/silero-vad-dart/actions/workflows/ci.yml/badge.svg)](https://github.com/Fi2zz/silero-vad-dart/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Cross-Platform** - Works on iOS, Android, macOS, Windows, Linux, and Web
- **Real-time VAD** - Stream audio chunks and detect speech on-the-fly with `VADIterator`
- **Batch Processing** - Process entire audio files with `getSpeechTimestamps`
- **ONNX Runtime** - Uses `flutter_onnxruntime` for high-performance inference
- **API Compatible** - Similar API to the original Python `silero-vad` library
- **WAV I/O** - Built-in WAV file reading and writing utilities
- **Configurable** - Adjustable thresholds, timing parameters, and speech padding

## Installation

### 1. Add dependency

```yaml
dependencies:
  silero_vad_flutter: ^0.1.0
```

### 2. Download the ONNX model

```bash
# From your project root
curl -L -o assets/models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

Or use the provided script:

```bash
dart run silero_vad_flutter:download_model
```

### 3. Register the asset

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/silero_vad.onnx
```

## Quick Start

```dart
import 'package:silero_vad_flutter/silero_vad_flutter.dart';

// Load the model
final model = await SileroVadModel.fromAsset('assets/models/silero_vad.onnx');

// Read a WAV file
final audio = await AudioUtils.readAudio('path/to/audio.wav', samplingRate: 16000);

// Detect speech segments
final timestamps = await getSpeechTimestamps(audio, model);

for (final ts in timestamps) {
  print('Speech from ${ts.start} to ${ts.end} samples');
}

// Extract speech audio
final speechAudio = collectChunks(timestamps, audio);

// Save speech-only audio
await saveAudio('path/to/speech_only.wav', speechAudio);

// Clean up
model.dispose();
```

## Real-Time VAD

```dart
final iterator = VADIterator(model, samplingRate: 16000);

// For each audio chunk from microphone:
for (final chunk in microphoneStream) {
  final result = await iterator.process(chunk);

  if (result != null) {
    if (result.containsKey('start')) {
      print('Speech started at ${result["start"]} samples');
    } else if (result.containsKey('end')) {
      print('Speech ended at ${result["end"]} samples');
    }
  }
}
```

## Configuration Options

```dart
final timestamps = await getSpeechTimestamps(
  audio,
  model,
  threshold: 0.5,                    // Speech probability threshold
  negThreshold: 0.35,               // Negative threshold (speech end)
  minSpeechDurationMs: 250,         // Minimum speech duration
  maxSpeechDurationS: 10.0,         // Maximum speech duration
  minSilenceDurationMs: 100,        // Minimum silence to split
  speechPadMs: 30,                  // Padding on each side
  returnSeconds: true,              // Return timestamps in seconds
  progressTrackingCallback: (p) => print('${(p * 100).toStringAsFixed(0)}%'),
);
```

## Platform Setup

### Android

No additional setup required. The `flutter_onnxruntime` plugin handles native libraries automatically.

### iOS

Add to `ios/Podfile` if needed:

```ruby
post_install do |installer|
  installer.pods_project.targets.each do |target|
    flutter_additional_ios_build_settings(target)
    target.build_configurations.each do |config|
      config.build_settings['EXCLUDED_ARCHS[sdk=iphonesimulator*]'] = 'i386'
    end
  end
end
```

### macOS / Windows / Linux

No additional setup required.

### Web

For web support, ensure your `web/index.html` loads the ONNX Runtime WASM:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js"></script>
```

## Example App

Check the [example](example/) directory for a complete Flutter app demonstrating both file-based and real-time VAD.

```bash
cd example
flutter pub get
# Place model at example/assets/models/silero_vad.onnx
flutter run
```

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `SileroVadModel` | ONNX model wrapper - load, predict, manage state |
| `VADIterator` | Real-time streaming VAD processor |
| `SpeechTimestamp` | Speech segment with start/end positions |
| `AudioUtils` | Audio I/O and manipulation utilities |
| `WavReader` | WAV file parser |
| `WavWriter` | WAV file writer |

### Functions

| Function | Description |
|----------|-------------|
| `getSpeechTimestamps` | Detect speech segments in audio |
| `collectChunks` | Extract and concatenate audio segments |
| `dropChunks` | Remove specified audio segments |
| `readAudio` | Read WAV file to Float32List |
| `saveAudio` | Save Float32List to WAV file |

## Architecture

```
Silero VAD ONNX Model
┌─────────────────────────────────────────────┐
│  Inputs: input[batch, window], state[2,batch,128], sr[1]  │
│  Outputs: output[batch, 1], state[2, batch, 128]          │
└─────────────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
getSpeechTimestamps  VADIterator      predict
(Batch processing)   (Streaming)      (Single chunk)
```

## Performance

- **Model Size**: ~2MB (ONNX)
- **Inference Time**: <1ms per 30ms audio chunk on modern mobile CPU
- **Memory**: Minimal - only stores GRU state (2 x 1 x 128 floats)

## License

MIT License - see [LICENSE](LICENSE) file.

This project uses the Silero VAD model which is also licensed under MIT.
