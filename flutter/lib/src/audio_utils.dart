import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'utils_vad.dart';

/// Audio utility functions for reading, writing, and manipulating audio data.
///
/// Provides WAV file I/O and audio chunk extraction/dropping operations
/// compatible with the Silero VAD pipeline.
class AudioUtils {
  AudioUtils._(); // prevent instantiation

  /// Read a WAV audio file and return samples as Float32List.
  ///
  /// [path] is the path to the WAV file.
  /// [samplingRate] is the target sampling rate. If the file's rate differs,
  /// a simple sample-rate conversion is performed.
  ///
  /// Returns a mono Float32List with samples normalized to [-1.0, 1.0].
  static Future<Float32List> readAudio(String path,
      {int samplingRate = 16000}) async {
    final file = File(path);
    if (!await file.exists()) {
      throw FileSystemException('Audio file not found', path);
    }
    final bytes = await file.readAsBytes();
    final reader = WavReader.fromBytes(bytes);

    Float32List samples = reader.samples;

    // Resample if necessary (simple nearest-neighbor / linear)
    if (reader.sampleRate != samplingRate) {
      samples = _resample(samples, reader.sampleRate, samplingRate);
    }

    return samples;
  }

  /// Save audio samples as a PCM16 WAV file.
  ///
  /// [path] is the output file path.
  /// [audio] is a Float32List with samples in [-1.0, 1.0].
  /// [samplingRate] is the output sampling rate in Hz.
  static Future<void> saveAudio(
    String path,
    Float32List audio, {
    int samplingRate = 16000,
  }) async {
    final wavBytes = WavWriter.write(audio, samplingRate);
    final file = File(path);
    await file.writeAsBytes(wavBytes);
  }

  /// Extract and concatenate audio chunks specified by timestamps.
  ///
  /// [timestamps] list of speech segment timestamps in samples.
  /// [wav] source audio samples.
  /// [seconds] if true, timestamps are in seconds and [samplingRate] must be provided.
  /// [samplingRate] required when [seconds] is true.
  static Float32List collectChunks(
    List<SpeechTimestamp> timestamps,
    Float32List wav, {
    bool seconds = false,
    int? samplingRate,
  }) {
    return collectChunksFrom(timestamps, wav,
        seconds: seconds, samplingRate: samplingRate);
  }

  /// Remove audio chunks and return the remaining audio.
  ///
  /// [timestamps] list of speech segment timestamps to remove.
  /// [wav] source audio samples.
  /// [seconds] if true, timestamps are in seconds and [samplingRate] must be provided.
  /// [samplingRate] required when [seconds] is true.
  static Float32List dropChunks(
    List<SpeechTimestamp> timestamps,
    Float32List wav, {
    bool seconds = false,
    int? samplingRate,
  }) {
    return dropChunksFrom(timestamps, wav,
        seconds: seconds, samplingRate: samplingRate);
  }

  /// Convert timestamps from seconds to samples.
  ///
  /// [timestamps] list with [start] and [end] in seconds.
  /// [samplingRate] the sampling rate in Hz.
  static List<SpeechTimestamp> secondsToSamples(
    List<SpeechTimestamp> timestamps,
    int samplingRate,
  ) {
    return timestamps
        .map((t) => SpeechTimestamp(
              start: (t.start * samplingRate).round(),
              end: (t.end * samplingRate).round(),
            ))
        .toList();
  }

  /// Simple linear resampling from [fromRate] to [toRate].
  static Float32List _resample(
      Float32List samples, int fromRate, int toRate) {
    if (fromRate == toRate) return samples;

    final ratio = toRate / fromRate;
    final newLength = (samples.length * ratio).round();
    final result = Float32List(newLength);

    for (int i = 0; i < newLength; i++) {
      final srcPos = i / ratio;
      final srcIdx = srcPos.floor();
      final frac = srcPos - srcIdx;

      if (srcIdx >= samples.length - 1) {
        result[i] = samples.last;
      } else {
        result[i] =
            samples[srcIdx] * (1 - frac) + samples[srcIdx + 1] * frac;
      }
    }

    return result;
  }
}

/// Top-level convenience function to read audio.
///
/// Same as [AudioUtils.readAudio].
Future<Float32List> readAudio(String path, {int samplingRate = 16000}) =>
    AudioUtils.readAudio(path, samplingRate: samplingRate);

/// Top-level convenience function to save audio.
///
/// Same as [AudioUtils.saveAudio].
Future<void> saveAudio(String path, Float32List audio,
        {int samplingRate = 16000}) =>
    AudioUtils.saveAudio(path, audio, samplingRate: samplingRate);

// ============================================================================
// WAV File Reader
// ============================================================================

/// Reads WAV (RIFF) format audio files.
///
/// Supports PCM16 and PCM24 formats. Automatically converts stereo to mono
/// by averaging channels. Normalizes samples to [-1.0, 1.0].
class WavReader {
  /// Sample rate in Hz.
  final int sampleRate;

  /// Number of channels (1 = mono, 2 = stereo).
  final int channels;

  /// Bits per sample (typically 16 or 24).
  final int bitsPerSample;

  /// Audio samples normalized to [-1.0, 1.0] in mono.
  final Float32List samples;

  WavReader._(this.sampleRate, this.channels, this.bitsPerSample, this.samples);

  /// Parse a WAV file from raw bytes.
  ///
  /// [bytes] is the raw byte content of a .wav file.
  factory WavReader.fromBytes(Uint8List bytes) {
    final byteData = ByteData.sublistView(bytes);
    int offset = 0;

    // --- RIFF header ---
    if (_readString(bytes, 0, 4) != 'RIFF') {
      throw FormatException('Invalid WAV file: missing RIFF header');
    }
    // File size (excluding RIFF header)
    // final fileSize = _readUint32LE(bytes, 4);
    if (_readString(bytes, 8, 4) != 'WAVE') {
      throw FormatException('Invalid WAV file: missing WAVE marker');
    }
    offset = 12;

    int fmtOffset = -1;
    int dataOffset = -1;
    int dataSize = 0;
    int fmtSize = 0;

    // --- Chunk scanning ---
    while (offset < bytes.length - 8) {
      final chunkId = _readString(bytes, offset, 4);
      final chunkSize = _readUint32LE(bytes, offset + 4);

      if (chunkId == 'fmt ') {
        fmtOffset = offset + 8;
        fmtSize = chunkSize;
      } else if (chunkId == 'data') {
        dataOffset = offset + 8;
        dataSize = chunkSize;
        break; // Found data chunk
      }

      offset += 8 + chunkSize;
      // Align to word boundary
      if (chunkSize % 2 != 0) offset++;
    }

    if (fmtOffset < 0) {
      throw FormatException('Invalid WAV file: missing fmt chunk');
    }
    if (dataOffset < 0 || dataSize == 0) {
      throw FormatException('Invalid WAV file: missing data chunk');
    }

    // --- Parse fmt chunk ---
    final audioFormat = _readUint16LE(bytes, fmtOffset);
    final numChannels = _readUint16LE(bytes, fmtOffset + 2);
    final sampleRate = _readUint32LE(bytes, fmtOffset + 4);
    // final byteRate = _readUint32LE(bytes, fmtOffset + 8);
    // final blockAlign = _readUint16LE(bytes, fmtOffset + 12);
    final bitsPerSample = _readUint16LE(bytes, fmtOffset + 14);

    if (audioFormat != 1) {
      // PCM format code is 1
      // Try to handle common extended formats gracefully
      if (fmtSize >= 18) {
        // Check for extensible
        final extensionSize = _readUint16LE(bytes, fmtOffset + 16);
        if (extensionSize >= 22) {
          // For extensible, check sub-format
          // GUID for PCM: 00000001-0000-0010-8000-00aa00389b71
          // First 2 bytes of GUID should match audio format
          final guidFormat = _readUint16LE(bytes, fmtOffset + 18);
          if (guidFormat != 1) {
            throw FormatException(
                'Unsupported WAV format: $audioFormat (only PCM supported)');
          }
        }
      } else {
        throw FormatException(
            'Unsupported WAV format: $audioFormat (only PCM supported)');
      }
    }

    if (![8, 16, 24, 32].contains(bitsPerSample)) {
      throw FormatException(
          'Unsupported bits per sample: $bitsPerSample');
    }

    // --- Read audio data ---
    final sampleData = bytes.sublist(dataOffset, dataOffset + dataSize);
    final numSamples = dataSize ~/ (numChannels * (bitsPerSample ~/ 8));

    Float32List monoSamples;

    if (bitsPerSample == 16) {
      monoSamples = _decodePCM16(sampleData, numChannels);
    } else if (bitsPerSample == 24) {
      monoSamples = _decodePCM24(sampleData, numChannels);
    } else if (bitsPerSample == 32) {
      monoSamples = _decodePCM32(sampleData, numChannels);
    } else {
      // 8-bit (unsigned)
      monoSamples = _decodePCM8(sampleData, numChannels);
    }

    return WavReader._(sampleRate, numChannels, bitsPerSample, monoSamples);
  }

  // --- Helper methods ---

  static String _readString(Uint8List bytes, int offset, int length) {
    return String.fromCharCodes(bytes.sublist(offset, offset + length));
  }

  static int _readUint32LE(Uint8List bytes, int offset) {
    return bytes[offset] |
        (bytes[offset + 1] << 8) |
        (bytes[offset + 2] << 16) |
        (bytes[offset + 3] << 24);
  }

  static int _readUint16LE(Uint8List bytes, int offset) {
    return bytes[offset] | (bytes[offset + 1] << 8);
  }

  static Float32List _decodePCM16(Uint8List data, int channels) {
    final byteData = ByteData.sublistView(data);
    final numFrames = data.length ~/ (2 * channels);
    final result = Float32List(numFrames);

    for (int i = 0; i < numFrames; i++) {
      double sum = 0.0;
      for (int ch = 0; ch < channels; ch++) {
        final idx = (i * channels + ch) * 2;
        final sample = byteData.getInt16(idx, Endian.little);
        sum += sample / 32768.0; // Normalize to [-1, 1]
      }
      result[i] = sum / channels;
    }

    return result;
  }

  static Float32List _decodePCM24(Uint8List data, int channels) {
    final numFrames = data.length ~/ (3 * channels);
    final result = Float32List(numFrames);

    for (int i = 0; i < numFrames; i++) {
      double sum = 0.0;
      for (int ch = 0; ch < channels; ch++) {
        final idx = (i * channels + ch) * 3;
        int sample = data[idx] | (data[idx + 1] << 8) | (data[idx + 2] << 16);
        // Sign extend 24-bit to 32-bit
        if (sample & 0x800000 != 0) {
          sample |= 0xFF000000;
        }
        sum += sample / 8388608.0; // Normalize to [-1, 1]
      }
      result[i] = sum / channels;
    }

    return result;
  }

  static Float32List _decodePCM32(Uint8List data, int channels) {
    final byteData = ByteData.sublistView(data);
    final numFrames = data.length ~/ (4 * channels);
    final result = Float32List(numFrames);

    for (int i = 0; i < numFrames; i++) {
      double sum = 0.0;
      for (int ch = 0; ch < channels; ch++) {
        final idx = (i * channels + ch) * 4;
        final sample = byteData.getInt32(idx, Endian.little);
        sum += sample / 2147483648.0; // Normalize to [-1, 1]
      }
      result[i] = sum / channels;
    }

    return result;
  }

  static Float32List _decodePCM8(Uint8List data, int channels) {
    final numFrames = data.length ~/ channels;
    final result = Float32List(numFrames);

    for (int i = 0; i < numFrames; i++) {
      double sum = 0.0;
      for (int ch = 0; ch < channels; ch++) {
        final idx = i * channels + ch;
        // 8-bit PCM is unsigned [0, 255], convert to [-1, 1]
        sum += (data[idx] - 128) / 128.0;
      }
      result[i] = sum / channels;
    }

    return result;
  }
}

// ============================================================================
// WAV File Writer
// ============================================================================

/// Writes audio samples to a WAV (RIFF) format file.
///
/// Supports 16-bit PCM output. Input samples should be in [-1.0, 1.0].
class WavWriter {
  WavWriter._(); // prevent instantiation

  /// Create WAV file bytes from Float32List samples.
  ///
  /// [samples] audio samples normalized to [-1.0, 1.0].
  /// [sampleRate] output sampling rate in Hz.
  /// [bitsPerSample] bit depth (default 16).
  static Uint8List write(
    Float32List samples,
    int sampleRate, {
    int bitsPerSample = 16,
  }) {
    if (bitsPerSample != 16) {
      throw ArgumentError('Only 16-bit PCM is currently supported');
    }

    final numChannels = 1; // Mono
    final bytesPerSample = bitsPerSample ~/ 8;
    final byteRate = sampleRate * numChannels * bytesPerSample;
    final blockAlign = numChannels * bytesPerSample;
    final dataSize = samples.length * bytesPerSample;
    final fileSize = 36 + dataSize; // 44 - 8 (excluding RIFF header)

    final buffer = Uint8List(44 + dataSize);
    final bd = ByteData.sublistView(buffer);
    int offset = 0;

    // --- RIFF header ---
    _writeString(buffer, offset, 'RIFF');
    offset += 4;
    bd.setUint32(offset, fileSize, Endian.little);
    offset += 4;
    _writeString(buffer, offset, 'WAVE');
    offset += 4;

    // --- fmt chunk ---
    _writeString(buffer, offset, 'fmt ');
    offset += 4;
    bd.setUint32(offset, 16, Endian.little); // fmt chunk size
    offset += 4;
    bd.setUint16(offset, 1, Endian.little); // Audio format (PCM)
    offset += 2;
    bd.setUint16(offset, numChannels, Endian.little);
    offset += 2;
    bd.setUint32(offset, sampleRate, Endian.little);
    offset += 4;
    bd.setUint32(offset, byteRate, Endian.little);
    offset += 4;
    bd.setUint16(offset, blockAlign, Endian.little);
    offset += 2;
    bd.setUint16(offset, bitsPerSample, Endian.little);
    offset += 2;

    // --- data chunk ---
    _writeString(buffer, offset, 'data');
    offset += 4;
    bd.setUint32(offset, dataSize, Endian.little);
    offset += 4;

    // --- PCM samples ---
    for (int i = 0; i < samples.length; i++) {
      // Clamp and convert to int16
      double clamped = samples[i];
      clamped = clamped < -1.0 ? -1.0 : (clamped > 1.0 ? 1.0 : clamped);
      final intSample = (clamped * 32767.0).round();
      bd.setInt16(offset, intSample, Endian.little);
      offset += 2;
    }

    return buffer;
  }

  static void _writeString(Uint8List buffer, int offset, String str) {
    final bytes = str.codeUnits;
    for (int i = 0; i < bytes.length; i++) {
      buffer[offset + i] = bytes[i];
    }
  }
}

// ============================================================================
// Chunk helpers (also exported from utils_vad.dart)
// ============================================================================

/// Internal implementation of [collectChunks].
Float32List collectChunksFrom(
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
    totalLength += (t.end - t.start).toInt();
  }

  final result = Float32List(totalLength);
  int offset = 0;
  for (final t in ts) {
    final chunkLen = (t.end - t.start).toInt();
    result.setRange(offset, offset + chunkLen, wav, t.start.toInt());
    offset += chunkLen;
  }

  return result;
}

/// Internal implementation of [dropChunks].
Float32List dropChunksFrom(
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
    final start = math.max(0, t.start.toInt());
    final end = math.min(t.end.toInt(), wav.length);
    for (int i = start; i < end; i++) {
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
