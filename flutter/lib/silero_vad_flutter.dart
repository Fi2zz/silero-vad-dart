/// Silero VAD Flutter - Voice Activity Detection for Flutter
///
/// A Dart/Flutter port of the Silero Voice Activity Detection (VAD) library./// Uses ONNX Runtime for efficient inference on mobile and desktop platforms.
///
/// ## Key Features
/// - Real-time speech detection in audio streams
/// - Batch processing of audio files
/// - Configurable thresholds and timing parameters
/// - WAV file reading and writing utilities
///
/// ## Setup
///
/// 1. Add `flutter_onnxruntime` to your `pubspec.yaml`:
/// ```yaml
/// dependencies:
///   silero_vad_flutter: ^0.1.0
///   flutter_onnxruntime: ^1.6.0
/// ```
///
/// 2. Download the ONNX model:
/// ```bash
/// curl -L -o assets/models/silero_vad.onnx \
///   https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
/// ```
///
/// 3. Register the asset in `pubspec.yaml`:
/// ```yaml
/// flutter:
///   assets:
///     - assets/models/silero_vad.onnx
/// ```
///
/// ## Usage
///
/// ```dart
/// import 'package:silero_vad_flutter/silero_vad_flutter.dart';
///
/// // Load model
/// final model = await SileroVadModel.fromAsset('assets/models/silero_vad.onnx');
///
/// // Read audio
/// final audio = await AudioUtils.readAudio('path/to/audio.wav');
///
/// // Detect speech
/// final timestamps = await getSpeechTimestamps(audio, model);
///
/// // Extract speech
/// final speechAudio = collectChunks(timestamps, audio);
/// ```
///
/// ## Real-time Processing
///
/// ```dart
/// final iterator = VADIterator(model, samplingRate: 16000);
/// for (final chunk in audioChunks) {
///   final result = await iterator.process(chunk);
///   if (result != null) {
///     if (result.containsKey('start')) {
///       print('Speech started at ${result["start"]}');
///     } else if (result.containsKey('end')) {
///       print('Speech ended at ${result["end"]}');
///     }
///   }
/// }
/// ```
library;

// Core model
export 'src/silero_vad_model.dart' show SileroVadModel, SileroVadException;

// VAD processing functions and data model
export 'src/utils_vad.dart'
    show
        SpeechTimestamp,
        VADIterator,
        getSpeechTimestamps,
        collectChunks,
        dropChunks;

// Audio I/O utilities
export 'src/audio_utils.dart'
    show AudioUtils, readAudio, saveAudio, WavReader, WavWriter;
