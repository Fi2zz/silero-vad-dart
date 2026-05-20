import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:meta/meta.dart';

/// Exception thrown when the Silero VAD model encounters an error.
class SileroVadException implements Exception {
  /// Error message describing what went wrong.
  final String message;

  /// Creates a [SileroVadException] with the given [message].
  SileroVadException(this.message);

  @override
  String toString() => 'SileroVadException: $message';
}

/// Dart port of the Python `OnnxWrapper` class for Silero VAD.
///
/// Wraps an ONNX Runtime session for the Silero Voice Activity Detection
/// model and manages the streaming state (hidden state + context) that
/// the model requires between successive windows.
///
/// ## Model I/O
///
/// The ONNX model expects **3 inputs**:
/// | Name    | Type   | Shape              | Description          |
/// |---------|--------|--------------------|----------------------|
/// | `input` | float32| `[batch, window]`  | Audio window(s)      |
/// | `state` | float32| `[2, batch, 128]`  | GRU hidden state     |
/// | `sr`    | int64  | `[1]`              | Sampling rate (Hz)   |
///
/// And produces **2 outputs**:
/// | Name     | Type   | Shape           | Description              |
/// |----------|--------|-----------------|--------------------------|
/// | `output` | float32| `[batch, 1]`     | Speech probability       |
/// | `state`  | float32| `[2, batch, 128]`| Updated GRU hidden state |
///
/// ## Usage
///
/// ```dart
/// final model = await SileroVadModel.fromAsset(
///   'assets/models/silero_vad.onnx',
/// );
/// final prob = await model.predict(audioChunk, 16000);
/// model.dispose();
/// ```
class SileroVadModel {
  // ------------------------------------------------------------------
  // Internal ONNX handles
  // ------------------------------------------------------------------

  /// ONNX Runtime instance.
  final OnnxRuntime _ort;

  /// ONNX inference session.
  final OrtSession _session;

  /// Whether this instance owns the session and should destroy it on
  /// [dispose]. When a session is injected via the constructor this
  /// is `false`.
  final bool _ownsSession;

  /// Internal state tensor stored flat as `[2, batch, 128]`.
  Float32List _state = Float32List(2 * 1 * 128);

  /// Context buffer from the previous window (last *contextSize*
  /// samples of each batch item).
  Float32List _context = Float32List(0);

  /// The sample rate used for the last inference call.
  int _lastSr = 0;

  /// The batch size used for the last inference call.
  int _lastBatchSize = 0;

  /// Sampling rates supported by this model instance.
  ///
  /// Defaults to `[8000, 16000]`. When loading a 16 kHz-specific
  /// variant only `[16000]` is advertised.
  final List<int> sampleRates;

  // -- Cached derived values ------------------------------------------

  /// Context size in samples (64 for 16 kHz, 32 for 8 kHz).
  int get _contextSize => _lastSr >= 16000 ? 64 : 32;

  /// Window size in samples (512 for 16 kHz, 256 for 8 kHz).
  int get _windowSize => _lastSr >= 16000 ? 512 : 256;

  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /// **Private** constructor used by the factory methods.
  SileroVadModel._(this._ort, this._session, this.sampleRates,
      this._ownsSession);

  /// **Advanced** – inject a pre-created ONNX session.
  ///
  /// Use this constructor when you already have an `OrtSession` from
  /// flutter_onnxruntime and want to wrap it manually.
  ///
  /// ```dart
  /// final ort = OnnxRuntime();
  /// final session = await ort.createSession('model.onnx');
  /// final model = SileroVadModel.fromSession(session, ort);
  /// ```
  ///
  /// The caller is responsible for disposing [session] and [ort] – this
  /// instance will **not** call `.destroy()` on them.
  factory SileroVadModel.fromSession(OrtSession session, OnnxRuntime ort,
      {List<int> sampleRates = const [8000, 16000]}) {
    final model = SileroVadModel._(ort, session, sampleRates, false);
    model.resetStates();
    return model;
  }

  // ------------------------------------------------------------------
  // Named factory constructors
  // ------------------------------------------------------------------

  /// Load a Silero VAD model from a Flutter asset bundle.
  ///
  /// [assetPath] is the key used in `pubspec.yaml` assets
  /// (e.g. `'assets/models/silero_vad.onnx'`).
  ///
  /// Set [forceCpu] to `true` (default) to restrict inference to CPU.
  static Future<SileroVadModel> fromAsset(String assetPath,
      {bool forceCpu = true}) async {
    final rawAssetFile = await rootBundle.load(assetPath);
    final bytes = rawAssetFile.buffer.asUint8List();

    final ort = OnnxRuntime();
    final sessionOptions = OrtSessionOptions();
    if (forceCpu) {
      sessionOptions.appendSessionConfigEntry(
          'session.intra_op_thread_affinities', '0');
    }
    final session =
        await ort.createSessionFromBuffer(bytes, sessionOptions: sessionOptions);

    final sampleRates = assetPath.contains('16k') ? [16000] : [8000, 16000];

    final model = SileroVadModel._(ort, session, sampleRates, true);
    model.resetStates();
    return model;
  }

  /// Load a Silero VAD model from an absolute file path.
  ///
  /// [filePath] is the path to the `.onnx` model file on disk.
  ///
  /// Set [forceCpu] to `true` (default) to restrict inference to CPU.
  static Future<SileroVadModel> fromFile(String filePath,
      {bool forceCpu = true}) async {
    final ort = OnnxRuntime();
    final sessionOptions = OrtSessionOptions();
    if (forceCpu) {
      sessionOptions.appendSessionConfigEntry(
          'session.intra_op_thread_affinities', '0');
    }
    final session =
        await ort.createSession(filePath, sessionOptions: sessionOptions);

    final sampleRates = filePath.contains('16k') ? [16000] : [8000, 16000];

    final model = SileroVadModel._(ort, session, sampleRates, true);
    model.resetStates();
    return model;
  }

  // ------------------------------------------------------------------
  // State management
  // ------------------------------------------------------------------

  /// Reset the internal model state.
  ///
  /// Call this before processing a new, unrelated audio stream or
  /// file so that hidden-state and context buffers do not leak across
  /// utterances.
  ///
  /// [batchSize] controls the shape of the hidden state tensor
  /// (`[2, batchSize, 128]`).
  void resetStates({int batchSize = 1}) {
    _state = Float32List(2 * batchSize * 128);
    _context = Float32List(0);
    _lastSr = 0;
    _lastBatchSize = 0;
  }

  // ------------------------------------------------------------------
  // Input validation
  // ------------------------------------------------------------------

  /// Validates the input audio tensor and determines the effective
  /// sample rate.
  ///
  /// [x] must be a flat [Float32List] whose length is an integer
  /// multiple of the window size derived from [sr].
  ///
  /// Returns a record `(processedInput, effectiveSr)`.
  @visibleForTesting
  (Float32List, int) validateInput(Float32List x, int sr) {
    return _validateInput(x, sr);
  }

  (Float32List, int) _validateInput(Float32List x, int sr) {
    // -- Validate sample rate ----------------------------------------
    late final int effectiveSr;
    if (sampleRates.contains(sr)) {
      effectiveSr = sr;
    } else if (sr % 16000 == 0) {
      // Multiple of 16000 – downsample by using the highest supported
      // rate the model was trained for.
      effectiveSr = sampleRates.last;
    } else {
      throw SileroVadException(
          'Unsupported sampling rate: $sr. '
          'Supported rates: $sampleRates, or multiples of 16000.');
    }

    // -- Determine window size for *this* effective sample rate ------
    final windowSize = effectiveSr >= 16000 ? 512 : 256;

    // -- Validate input length ---------------------------------------
    if (x.isEmpty) {
      throw SileroVadException('Input audio is empty.');
    }

    final batchSize = x.length ~/ windowSize;
    if (batchSize < 1) {
      throw SileroVadException(
          'Input too short: ${x.length} samples (need at least $windowSize).');
    }

    // -- Re-allocate state when dimensions change --------------------
    if (_lastSr != effectiveSr || _lastBatchSize != batchSize) {
      _lastSr = effectiveSr;
      _lastBatchSize = batchSize;
      _state = Float32List(2 * batchSize * 128);
      _context = Float32List(0);
    }

    return (x, effectiveSr);
  }

  // ------------------------------------------------------------------
  // Inference
  // ------------------------------------------------------------------

  /// Run inference on a single audio window.
  ///
  /// [input] is a [Float32List] of length `windowSize` (512 for 16kHz
  /// or 256 for 8kHz).
  ///
  /// Returns the speech probability as a [double] in range `[0, 1]`.
  Future<double> predict(Float32List input, int sr) async {
    final (_, effectiveSr) = _validateInput(input, sr);
    final windowSize = _windowSize;
    final batchSize = input.length ~/ windowSize;
    final contextSize = _contextSize;

    // -- Manage context ----------------------------------------------
    if (_context.isEmpty) {
      _context = Float32List(batchSize * contextSize);
    }

    // Concatenate context + input: [batch, context + window]
    final concatLength = batchSize * (contextSize + windowSize);
    final concatInput = Float32List(concatLength);

    for (int b = 0; b < batchSize; b++) {
      // Context
      for (int c = 0; c < contextSize; c++) {
        concatInput[b * (contextSize + windowSize) + c] =
            _context[b * contextSize + c];
      }
      // Input window
      for (int w = 0; w < windowSize; w++) {
        concatInput[b * (contextSize + windowSize) + contextSize + w] =
            input[b * windowSize + w];
      }
    }

    // -- Prepare ONNX inputs -----------------------------------------
    final inputTensor = await OrtValue.fromList(
        concatInput, [batchSize, contextSize + windowSize]);
    final stateTensor =
        await OrtValue.fromList(_state, [2, batchSize, 128]);
    final srTensor = await OrtValue.fromList(Int64List.fromList([effectiveSr]), [1]);

    final inputs = {
      'input': inputTensor,
      'state': stateTensor,
      'sr': srTensor,
    };

    // -- Run inference -----------------------------------------------
    final outputs = await _session.run(inputs);

    // -- Extract results ---------------------------------------------
    final outputData = await outputs['output']!.asList() as List<double>;
    final newStateData = await outputs['state']!.asList() as List<double>;

    // Update state
    for (int i = 0; i < newStateData.length && i < _state.length; i++) {
      _state[i] = newStateData[i].toDouble();
    }

    // Update context (last contextSize samples of input)
    for (int b = 0; b < batchSize; b++) {
      for (int c = 0; c < contextSize; c++) {
        final srcIdx = b * windowSize + (windowSize - contextSize + c);
        if (srcIdx < input.length) {
          _context[b * contextSize + c] = input[srcIdx];
        }
      }
    }

    // -- Release OrtValues -------------------------------------------
    inputTensor.dispose();
    stateTensor.dispose();
    srTensor.dispose();
    for (final v in outputs.values) {
      v.dispose();
    }

    // Return speech probability for first batch item
    return outputData.isNotEmpty ? outputData[0] : 0.0;
  }

  /// Process an entire audio clip and return speech probabilities
  /// for each window.
  ///
  /// This is useful for batch processing where you want the raw
  /// probabilities before thresholding.
  ///
  /// [audio] is a [Float32List] of mono audio samples.
  /// [sr] is the sampling rate in Hz.
  ///
  /// Returns a [List<double>] of speech probabilities, one per window.
  Future<List<double>> audioForward(Float32List audio, int sr) async {
    final (_, effectiveSr) = _validateInput(audio, sr);
    final windowSize = _windowSize;

    // Pad audio to window boundary
    final numSamples = audio.length;
    final remainder = numSamples % windowSize;
    final padLength = remainder == 0 ? 0 : windowSize - remainder;

    Float32List paddedAudio;
    if (padLength > 0) {
      paddedAudio = Float32List(numSamples + padLength);
      paddedAudio.setAll(0, audio);
      // Already zero-initialized
    } else {
      paddedAudio = audio;
    }

    final numWindows = paddedAudio.length ~/ windowSize;
    final probs = <double>[];

    resetStates();

    for (int i = 0; i < numWindows; i++) {
      final window = Float32List(windowSize);
      final start = i * windowSize;
      for (int j = 0; j < windowSize; j++) {
        window[j] = paddedAudio[start + j];
      }
      final prob = await predict(window, effectiveSr);
      probs.add(prob);
    }

    return probs;
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources.
  ///
  /// After calling [dispose] the model must not be used again.
  void dispose() {
    if (_ownsSession) {
      _session.destroy();
      _ort.destroy();
    }
  }
}
