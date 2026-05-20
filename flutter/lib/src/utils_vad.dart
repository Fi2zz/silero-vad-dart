import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'audio_utils.dart';
import 'silero_vad_model.dart';

/// Represents a detected speech segment with start and end positions.
///
/// By default, [start] and [end] are in sample indices (as [int]).
/// When [returnSeconds] is used with [getSpeechTimestamps], they store
/// seconds as [double].
class SpeechTimestamp {
  /// Start position of the speech segment.
  ///
  /// In samples when from [getSpeechTimestamps] with `returnSeconds: false`,
  /// or in seconds as a [double] when `returnSeconds: true`.
  num start;

  /// End position of the speech segment.
  ///
  /// In samples when from [getSpeechTimestamps] with `returnSeconds: false`,
  /// or in seconds as a [double] when `returnSeconds: true`.
  num end;

  SpeechTimestamp({required this.start, required this.end});

  /// Convert to JSON map.
  Map<String, num> toJson() => {'start': start, 'end': end};

  @override
  String toString() => 'SpeechTimestamp(start: $start, end: $end)';
}

/// Extract and concatenate audio chunks specified by [timestamps].
///
/// If [seconds] is true, [timestamps] are interpreted as being in seconds
/// and [samplingRate] must be provided to convert to samples.
Float32List collectChunks(
  List<SpeechTimestamp> timestamps,
  Float32List wav, {
  bool seconds = false,
  int? samplingRate,
}) {
  if (timestamps.isEmpty) {
    return Float32List(0);
  }

  List<SpeechTimestamp> ts = timestamps;
  if (seconds && samplingRate != null) {
    ts = AudioUtils.secondsToSamples(timestamps, samplingRate);
  }

  // Calculate total length
  int totalLength = 0;
  for (final t in ts) {
    totalLength += t.end - t.start;
  }

  final result = Float32List(totalLength);
  int offset = 0;
  for (final t in ts) {
    final chunkLen = t.end - t.start;
    result.setRange(offset, offset + chunkLen, wav, t.start);
    offset += chunkLen;
  }

  return result;
}

/// Remove audio chunks specified by [timestamps] and return the remainder.
///
/// If [seconds] is true, [timestamps] are interpreted as being in seconds
/// and [samplingRate] must be provided to convert to samples.
Float32List dropChunks(
  List<SpeechTimestamp> timestamps,
  Float32List wav, {
  bool seconds = false,
  int? samplingRate,
}) {
  if (timestamps.isEmpty) {
    return Float32List.fromList(wav);
  }

  List<SpeechTimestamp> ts = timestamps;
  if (seconds && samplingRate != null) {
    ts = AudioUtils.secondsToSamples(timestamps, samplingRate);
  }

  // Mark samples to keep
  final keep = List<bool>.filled(wav.length, true);
  for (final t in ts) {
    for (int i = t.start; i < math.min(t.end, wav.length); i++) {
      keep[i] = false;
    }
  }

  // Collect kept samples
  final result = <double>[];
  for (int i = 0; i < wav.length; i++) {
    if (keep[i]) {
      result.add(wav[i]);
    }
  }

  return Float32List.fromList(result);
}

/// Get speech timestamps from an audio clip using a Silero VAD model.
///
/// This function is a Dart port of the Python `get_speech_timestamps` function.
/// It processes the audio in overlapping windows, runs VAD inference on each
/// window, and applies hysteresis thresholding to detect speech segments.
///
/// Parameters:
/// - [audio]: Float32List of audio samples (mono).
/// - [model]: Loaded SileroVadModel instance.
/// - [threshold]: Speech probability threshold (default 0.5).
/// - [samplingRate]: Audio sampling rate in Hz (default 16000).
/// - [minSpeechDurationMs]: Minimum speech duration in ms (default 250).
/// - [maxSpeechDurationS]: Maximum speech duration in seconds (default infinity).
/// - [minSilenceDurationMs]: Minimum silence duration to split speech (default 100).
/// - [speechPadMs]: Padding applied to speech segments in ms (default 30).
/// - [returnSeconds]: If true, return timestamps in seconds instead of samples.
/// - [timeResolution]: Time resolution factor (default 1).
/// - [negThreshold]: Negative threshold (default = max(threshold - 0.15, 0.01)).
/// - [minSilenceAtMaxSpeech]: Minimum silence samples at max speech duration (default 98).
/// - [useMaxPossSilAtMaxSpeech]: Use possible silence at max speech duration (default true).
/// - [progressTrackingCallback]: Optional callback for progress updates (0.0 to 1.0).
///
/// Returns a list of [SpeechTimestamp] objects representing detected speech segments.
Future<List<SpeechTimestamp>> getSpeechTimestamps(
  Float32List audio,
  SileroVadModel model, {
  double threshold = 0.5,
  int samplingRate = 16000,
  int minSpeechDurationMs = 250,
  double maxSpeechDurationS = double.infinity,
  int minSilenceDurationMs = 100,
  int speechPadMs = 30,
  bool returnSeconds = false,
  int timeResolution = 1,
  double? negThreshold,
  int minSilenceAtMaxSpeech = 98,
  bool useMaxPossSilAtMaxSpeech = true,
  void Function(double progress)? progressTrackingCallback,
}) async {
  // ------------------------------------------------------------------
  // Parameter validation and setup
  // ------------------------------------------------------------------
  if (![8000, 16000].contains(samplingRate)) {
    throw ArgumentError('getSpeechTimestamps only supports 8000 and 16000 sampling rates');
  }

  final windowSize = samplingRate == 16000 ? 512 : 256;
  final contextSize = samplingRate == 16000 ? 64 : 32;
  final negThresh = negThreshold ?? math.max(threshold - 0.15, 0.01);

  final minSpeechSamples = samplingRate * minSpeechDurationMs ~/ 1000;
  final maxSpeechSamples =
      maxSpeechDurationS == double.infinity
          ? double.infinity
          : samplingRate * maxSpeechDurationS;
  final minSilenceSamples = samplingRate * minSilenceDurationMs ~/ 1000;
  final speechPadSamples = samplingRate * speechPadMs ~/ 1000;

  // ------------------------------------------------------------------
  // Pad audio and compute per-window probabilities
  // ------------------------------------------------------------------

  // Pad audio to window boundary with zeros
  final numWindows = (audio.length / windowSize).ceil();
  final paddedLength = numWindows * windowSize;
  final paddedAudio = Float32List(paddedLength);
  paddedAudio.setAll(0, audio);

  final List<double> probs = [];
  model.resetStates();

  final totalWindows = numWindows;
  for (int i = 0; i < numWindows; i++) {
    final window = Float32List(windowSize);
    final start = i * windowSize;
    for (int j = 0; j < windowSize; j++) {
      window[j] = paddedAudio[start + j];
    }
    final prob = await model.predict(window, samplingRate);
    probs.add(prob);

    if (progressTrackingCallback != null && totalWindows > 0) {
      progressTrackingCallback((i + 1) / totalWindows);
    }
  }

  // ------------------------------------------------------------------
  // Hysteresis thresholding to find speech segments
  // ------------------------------------------------------------------

  final List<SpeechTimestamp> speeches = [];
  bool triggered = false;
  int tempEnd = 0;
  int prevEnd = 0;
  final List<int> possibleEnds = [];

  for (int i = 0; i < probs.length; i++) {
    final currentProb = probs[i];
    final windowStartSample = i * windowSize;
    final windowEndSample = windowStartSample + windowSize;

    if (!triggered) {
      // Waiting for speech to start
      if (currentProb >= threshold) {
        // Speech starts somewhere in this window
        // Try to find the exact sample where it crosses threshold
        int speechStart = windowStartSample;

        // Look backwards to refine start within context
        if (i > 0 && probs[i - 1] < threshold) {
          // The exact crossing is somewhere in this window;
          // start from the beginning of the window.
          speechStart = windowStartSample;
        }

        speeches.add(SpeechTimestamp(start: speechStart, end: 0));
        triggered = true;
        tempEnd = 0;
        prevEnd = 0;
        possibleEnds.clear();
      }
    } else {
      // Currently in speech
      if (currentProb < negThresh) {
        // Speech might be ending
        if (tempEnd == 0) {
          tempEnd = windowStartSample;
        }

        // Track possible end points
        if (useMaxPossSilAtMaxSpeech) {
          possibleEnds.add(windowStartSample);
          // Keep only recent possible ends
          while (possibleEnds.length > minSilenceAtMaxSpeech) {
            possibleEnds.removeAt(0);
          }
        }

        // Check if silence duration exceeds minimum
        final silenceDuration = windowEndSample - tempEnd;
        if (silenceDuration >= minSilenceSamples) {
          // End of speech segment
          int speechEnd = tempEnd;

          // Check max speech duration
          final speechDuration = speechEnd - speeches.last.start;
          if (maxSpeechSamples != double.infinity &&
              speechDuration > maxSpeechSamples) {
            // Speech too long - try to split at a possible end
            if (possibleEnds.isNotEmpty) {
              // Find the best split point
              int bestSplit = speeches.last.start + maxSpeechSamples.toInt();
              int closestDist = double.maxFinite.toInt();
              for (final pe in possibleEnds) {
                if (pe > speeches.last.start) {
                  final dist = (pe - bestSplit).abs();
                  if (dist < closestDist) {
                    closestDist = dist;
                    bestSplit = pe;
                  }
                }
              }

              speeches.last.end = bestSplit;
              // Start new segment
              speeches.add(SpeechTimestamp(start: bestSplit, end: 0));
              triggered = true;
              tempEnd = 0;
              possibleEnds.clear();
              continue;
            }
          }

          speeches.last.end = speechEnd;
          prevEnd = speechEnd;
          triggered = false;
          tempEnd = 0;
          possibleEnds.clear();
        }
      } else {
        // Still speech - reset temp end
        tempEnd = 0;
        possibleEnds.clear();
      }
    }
  }

  // ------------------------------------------------------------------
  // Handle trailing speech
  // ------------------------------------------------------------------

  if (triggered && speeches.isNotEmpty) {
    // Speech continues to end of audio
    int speechEnd = audio.length;

    // Check max speech duration
    final speechDuration = speechEnd - speeches.last.start;
    if (maxSpeechSamples != double.infinity &&
        speechDuration > maxSpeechSamples) {
      if (possibleEnds.isNotEmpty) {
        int bestSplit = speeches.last.start + maxSpeechSamples.toInt();
        int closestDist = double.maxFinite.toInt();
        for (final pe in possibleEnds) {
          if (pe > speeches.last.start) {
            final dist = (pe - bestSplit).abs();
            if (dist < closestDist) {
              closestDist = dist;
              bestSplit = pe;
            }
          }
        }
        speeches.last.end = bestSplit;
        speeches.add(SpeechTimestamp(start: bestSplit, end: speechEnd));
      } else {
        speeches.last.end = speeches.last.start + maxSpeechSamples.toInt();
        speeches.add(SpeechTimestamp(
            start: speeches.last.end, end: speechEnd));
      }
    } else {
      speeches.last.end = speechEnd;
    }
  }

  // ------------------------------------------------------------------
  // Remove too-short segments and apply padding
  // ------------------------------------------------------------------

  final List<SpeechTimestamp> filtered = [];
  for (int i = 0; i < speeches.length; i++) {
    final s = speeches[i];
    final duration = s.end - s.start;

    if (duration < minSpeechSamples) {
      continue;
    }

    // Apply padding
    int paddedStart = math.max(0, s.start - speechPadSamples);
    int paddedEnd = math.min(audio.length, s.end + speechPadSamples);

    // Merge with previous if overlapping after padding
    if (filtered.isNotEmpty && paddedStart <= filtered.last.end) {
      filtered.last.end = paddedEnd;
    } else {
      filtered.add(SpeechTimestamp(start: paddedStart, end: paddedEnd));
    }
  }

  // ------------------------------------------------------------------
  // Convert to seconds if requested
  // ------------------------------------------------------------------

  if (returnSeconds) {
    final result = <SpeechTimestamp>[];
    for (final s in filtered) {
      result.add(SpeechTimestamp(
        start: s.start / samplingRate * timeResolution,
        end: s.end / samplingRate * timeResolution,
      ));
    }
    return result;
  }

  return filtered;
}

// ============================================================================
// VADIterator - Real-time streaming VAD
// ============================================================================

/// Streaming VAD iterator for real-time audio processing.
///
/// Processes audio chunks one at a time and returns events when speech
/// starts or ends.
///
/// ## Usage
/// ```dart
/// final vad = VADIterator(model, samplingRate: 16000);
/// for (final chunk in microphoneChunks) {
///   final result = await vad.process(chunk);
///   if (result != null) {
///     if (result.containsKey('start')) {
///       print('Speech started: ${result["start"]}');
///     } else if (result.containsKey('end')) {
///       print('Speech ended: ${result["end"]}');
///     }
///   }
/// }
/// ```
class VADIterator {
  /// The underlying Silero VAD model.
  final SileroVadModel model;

  /// Speech probability threshold.
  final double threshold;

  /// Negative threshold (speech end).
  final double negThreshold;

  /// Audio sampling rate in Hz.
  final int samplingRate;

  /// Minimum silence duration in samples.
  final int minSilenceSamples;

  /// Speech padding in samples.
  final int speechPadSamples;

  /// Whether speech is currently detected.
  bool triggered = false;

  /// Temporary end position of current speech segment.
  int tempEnd = 0;

  /// Current sample position in the stream.
  int currentSample = 0;

  /// Previous end position.
  int _prevEnd = 0;

  /// Start position of current speech segment.
  int _speechStart = 0;

  /// Window size based on sample rate.
  late final int _windowSize;

  /// Context size based on sample rate.
  late final int _contextSize;

  /// Create a VAD iterator.
  ///
  /// [model] is a loaded SileroVadModel.
  /// [threshold] is the speech probability threshold (default 0.5).
  /// [samplingRate] must be 8000 or 16000.
  /// [minSilenceDurationMs] is the minimum silence to end speech (default 100).
  /// [speechPadMs] is padding around speech segments (default 30).
  VADIterator(
    this.model, {
    this.threshold = 0.5,
    this.samplingRate = 16000,
    int minSilenceDurationMs = 100,
    int speechPadMs = 30,
  })  : negThreshold = math.max(threshold - 0.15, 0.01),
        minSilenceSamples = samplingRate * minSilenceDurationMs ~/ 1000,
        speechPadSamples = samplingRate * speechPadMs ~/ 1000,
        _windowSize = samplingRate == 16000 ? 512 : 256,
        _contextSize = samplingRate == 16000 ? 64 : 32 {
    if (![8000, 16000].contains(samplingRate)) {
      throw ArgumentError(
          'VADIterator does not support sampling rates other than [8000, 16000]');
    }
  }

  /// Reset the iterator state for a new audio stream.
  void resetStates() {
    model.resetStates();
    triggered = false;
    tempEnd = 0;
    currentSample = 0;
    _prevEnd = 0;
    _speechStart = 0;
  }

  /// Process a single audio chunk.
  ///
  /// [x] is a Float32List of audio samples. The chunk size should match
  /// the model's expected window size (512 for 16kHz, 256 for 8kHz).
  ///
  /// If [returnSeconds] is true, returned positions are in seconds.
  /// [timeResolution] scales the time values.
  ///
  /// Returns:
  /// - `{'start': position}` when speech begins.
  /// - `{'end': position}` when speech ends.
  /// - `null` if no state change occurred.
  Future<Map<String, dynamic>?> process(
    Float32List x, {
    bool returnSeconds = false,
    int timeResolution = 1,
  }) async {
    // Get speech probability for this chunk
    final double prob = await model.predict(x, samplingRate);

    final chunkStart = currentSample;
    final chunkEnd = currentSample + x.length;
    currentSample = chunkEnd;

    Map<String, dynamic>? result;

    if (!triggered) {
      // Not in speech - check if speech starts
      if (prob >= threshold) {
        triggered = true;
        _speechStart = math.max(0, chunkStart - speechPadSamples);
        tempEnd = 0;
        _prevEnd = 0;

        final startPos = returnSeconds
            ? (_speechStart / samplingRate * timeResolution).round()
            : _speechStart;
        result = {'start': startPos};
      }
    } else {
      // Currently in speech
      if (prob < negThreshold) {
        // Speech might be ending
        if (tempEnd == 0) {
          tempEnd = chunkStart;
        }

        // Check if silence has lasted long enough
        final silenceDuration = chunkEnd - tempEnd;
        if (silenceDuration >= minSilenceSamples) {
          // End speech segment
          final endPos = tempEnd + speechPadSamples;
          _prevEnd = endPos;

          final finalEnd = returnSeconds
              ? (endPos / samplingRate * timeResolution).round()
              : endPos;
          result = {'end': finalEnd};

          triggered = false;
          tempEnd = 0;
        }
      } else {
        // Still in speech - reset temporary end
        tempEnd = 0;
      }
    }

    return result;
  }

  /// Process a chunk and return both the event and current probability.
  ///
  /// This is useful for applications that want to display a live VAD
  /// probability alongside the start/end events.
  ///
  /// Unlike calling [process] then [model.predict] separately, this method
  /// performs only a single inference pass.
  Future<(Map<String, dynamic>?, double)> processWithProb(
    Float32List x, {
    bool returnSeconds = false,
    int timeResolution = 1,
  }) async {
    // Single inference
    final double prob = await model.predict(x, samplingRate);

    final chunkStart = currentSample;
    final chunkEnd = currentSample + x.length;
    currentSample = chunkEnd;

    Map<String, dynamic>? event;

    if (!triggered) {
      if (prob >= threshold) {
        triggered = true;
        _speechStart = math.max(0, chunkStart - speechPadSamples);
        tempEnd = 0;
        _prevEnd = 0;

        final startPos = returnSeconds
            ? (_speechStart / samplingRate * timeResolution).round()
            : _speechStart;
        event = {'start': startPos};
      }
    } else {
      if (prob < negThreshold) {
        if (tempEnd == 0) {
          tempEnd = chunkStart;
        }

        final silenceDuration = chunkEnd - tempEnd;
        if (silenceDuration >= minSilenceSamples) {
          final endPos = tempEnd + speechPadSamples;
          _prevEnd = endPos;

          final finalEnd = returnSeconds
              ? (endPos / samplingRate * timeResolution).round()
              : endPos;
          event = {'end': finalEnd};

          triggered = false;
          tempEnd = 0;
        }
      } else {
        tempEnd = 0;
      }
    }

    return (event, prob);
  }

  /// Force-end any ongoing speech segment.
  ///
  /// Call this at the end of a stream to finalize the last segment.
  /// Returns `{'end': position}` if speech was active, null otherwise.
  Map<String, dynamic>? finalize({bool returnSeconds = false, int timeResolution = 1}) {
    if (triggered) {
      final endPos = currentSample + speechPadSamples;
      triggered = false;
      tempEnd = 0;

      final finalEnd = returnSeconds
          ? (endPos / samplingRate * timeResolution).round()
          : endPos;
      return {'end': finalEnd};
    }
    return null;
  }
}
