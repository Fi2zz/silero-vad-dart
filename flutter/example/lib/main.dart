/// Silero VAD Flutter Example App
///
/// A comprehensive demonstration of the silero_vad_flutter package featuring:
/// - File-based Voice Activity Detection on WAV files
/// - Real-time VAD via microphone streaming
///
/// The app uses Material 3 design with proper error handling,
/// loading states, and clean separation of concerns.
library;

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:intl/intl.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:silero_vad_flutter/silero_vad_flutter.dart';
import 'package:wav/wav.dart';

// ---------------------------------------------------------------------------
// Entry Point
// ---------------------------------------------------------------------------

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SileroVadExampleApp());
}

// ---------------------------------------------------------------------------
// App Root
// ---------------------------------------------------------------------------

/// Root widget of the Silero VAD example application.
class SileroVadExampleApp extends StatelessWidget {
  /// Creates the root app widget.
  const SileroVadExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Silero VAD Flutter',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6750A4),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6750A4),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const MainScreen(),
    );
  }
}

// ---------------------------------------------------------------------------
// Main Screen with Tabs
// ---------------------------------------------------------------------------

/// The main screen containing File VAD and Real-time VAD tabs.
class MainScreen extends StatefulWidget {
  /// Creates the main screen.
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Silero VAD Flutter'),
        centerTitle: true,
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.audio_file), text: 'File VAD'),
            Tab(icon: Icon(Icons.mic), text: 'Real-time VAD'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: const [
          FileVadTab(),
          RealtimeVadTab(),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// File VAD Tab
// ---------------------------------------------------------------------------

/// Tab for performing VAD on a pre-recorded audio file.
///
/// Allows the user to pick a WAV file, then runs Silero VAD on it to detect
/// speech segments. Results are displayed in a scrollable list with timestamps.
class FileVadTab extends StatefulWidget {
  /// Creates the File VAD tab.
  const FileVadTab({super.key});

  @override
  State<FileVadTab> createState() => _FileVadTabState();
}

class _FileVadTabState extends State<FileVadTab> {
  /// The loaded Silero VAD model.
  SileroVadModel? _model;

  /// Whether the model is being loaded.
  bool _isLoadingModel = false;

  /// Whether VAD processing is running.
  bool _isProcessing = false;

  /// Error message to display, if any.
  String? _errorMessage;

  /// Detected speech segments from the last VAD run.
  List<SpeechTimestamp> _segments = [];

  /// The path of the currently selected file.
  String? _selectedFilePath;

  /// Model load progress (0.0 to 1.0).
  double _modelLoadProgress = 0.0;

  /// VAD configuration
  final double _threshold = 0.5;
  final int _sampleRate = 16000;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _model?.dispose();
    super.dispose();
  }

  /// Loads the Silero VAD ONNX model from application assets.
  Future<void> _loadModel() async {
    if (_isLoadingModel || _model != null) return;

    setState(() {
      _isLoadingModel = true;
      _errorMessage = null;
      _modelLoadProgress = 0.0;
    });

    try {
      // Copy the model from assets to a temporary file so ONNX Runtime can
      // load it.
      final ByteData modelData = await rootBundle.load(
        'assets/models/silero_vad.onnx',
      );

      setState(() => _modelLoadProgress = 0.3);

      final Directory tempDir = await getTemporaryDirectory();
      final String modelPath = '${tempDir.path}/silero_vad.onnx';
      final File modelFile = File(modelPath);

      if (!modelFile.existsSync()) {
        await modelFile.writeAsBytes(modelData.buffer.asUint8List());
      }

      setState(() => _modelLoadProgress = 0.6);

      final model = await SileroVadModel.fromFile(modelPath);

      if (!mounted) {
        model.dispose();
        return;
      }

      setState(() {
        _model = model;
        _isLoadingModel = false;
        _modelLoadProgress = 1.0;
      });
    } on Exception catch (e, stackTrace) {
      if (!mounted) return;
      setState(() {
        _isLoadingModel = false;
        _errorMessage = 'Failed to load VAD model: $e';
      });
      debugPrint('Model load error: $e\n$stackTrace');
    }
  }

  /// Opens a file picker dialog so the user can select a WAV file.
  Future<void> _pickWavFile() async {
    try {
      final FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav'],
        allowMultiple: false,
      );

      if (result == null || result.files.single.path == null) return;

      setState(() {
        _selectedFilePath = result.files.single.path;
        _segments = [];
        _errorMessage = null;
      });

      await _processFile(_selectedFilePath!);
    } on Exception catch (e, stackTrace) {
      setState(() => _errorMessage = 'Failed to pick file: $e');
      debugPrint('File picker error: $e\n$stackTrace');
    }
  }

  /// Reads a WAV file and runs VAD to detect speech segments.
  Future<void> _processFile(String filePath) async {
    if (_model == null) return;

    setState(() {
      _isProcessing = true;
      _segments = [];
      _errorMessage = null;
    });

    try {
      // Parse the WAV file.
      final Wav wavFile = await Wav.readFile(filePath);
      final Float64List monoSamples = _convertToMono(wavFile);
      final int wavSampleRate = wavFile.samplesPerSecond;

      // Validate sample rate.
      if (wavSampleRate != 16000 && wavSampleRate != 8000) {
        setState(() {
          _isProcessing = false;
          _errorMessage =
              'Unsupported sample rate: $wavSampleRate Hz. '
              'Only 16 kHz and 8 kHz are supported.';
        });
        return;
      }

      // Convert Float64List to Float32List for the VAD.
      final Float32List audioData = Float32List(monoSamples.length);
      for (int i = 0; i < monoSamples.length; i++) {
        audioData[i] = monoSamples[i].toDouble();
      }

      // Run VAD using the real API.
      final List<SpeechTimestamp> segments = await getSpeechTimestamps(
        audioData,
        _model!,
        threshold: _threshold,
        samplingRate: wavSampleRate,
        returnSeconds: true,
      );

      if (!mounted) return;

      setState(() {
        _segments = segments;
        _isProcessing = false;
      });
    } on Exception catch (e, stackTrace) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _errorMessage = 'VAD processing failed: $e';
      });
      debugPrint('VAD processing error: $e\n$stackTrace');
    }
  }

  /// Converts multi-channel WAV data to mono by averaging channels.
  Float64List _convertToMono(Wav wavFile) {
    final int numChannels = wavFile.channels.length;
    if (numChannels == 1) {
      return wavFile.channels[0];
    }

    final int numSamples = wavFile.channels[0].length;
    final Float64List mono = Float64List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      double sum = 0.0;
      for (int ch = 0; ch < numChannels; ch++) {
        sum += wavFile.channels[ch][i];
      }
      mono[i] = sum / numChannels;
    }
    return mono;
  }

  /// Formats seconds into a human-readable duration string.
  String _formatDuration(double seconds) {
    final int mins = seconds ~/ 60;
    final double secs = seconds % 60;
    return '${mins.toString().padLeft(2, '0')}:${secs.toStringAsFixed(3).padLeft(6, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final ColorScheme colorScheme = theme.colorScheme;

    // Show model loading UI.
    if (_isLoadingModel) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 120,
              height: 120,
              child: CircularProgressIndicator(
                value: _modelLoadProgress > 0 && _modelLoadProgress < 1
                    ? _modelLoadProgress
                    : null,
                strokeWidth: 8,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Loading Silero VAD Model...',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(
              '${(_modelLoadProgress * 100).toInt()}%',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );
    }

    // Show error if model failed to load.
    if (_errorMessage != null && _model == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.error_outline,
                size: 64,
                color: colorScheme.error,
              ),
              const SizedBox(height: 16),
              Text(
                'Error',
                style: theme.textTheme.headlineSmall?.copyWith(
                  color: colorScheme.error,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium,
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _loadModel,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // File selection card.
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Select Audio File',
                    style: theme.textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Choose a WAV file to analyze for speech segments.',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      FilledButton.icon(
                        onPressed: _isProcessing ? null : _pickWavFile,
                        icon: _isProcessing
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : const Icon(Icons.folder_open),
                        label: Text(
                          _isProcessing ? 'Processing...' : 'Pick WAV File',
                        ),
                      ),
                      const SizedBox(width: 12),
                      if (_selectedFilePath != null)
                        Expanded(
                          child: Text(
                            _selectedFilePath!.split(Platform.pathSeparator).last,
                            style: theme.textTheme.bodySmall,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Processing indicator.
          if (_isProcessing)
            const LinearProgressIndicator()
          else
            const Divider(),

          // Error message.
          if (_errorMessage != null && _model != null)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8.0),
              child: MaterialBanner(
                content: Text(_errorMessage!),
                leading: Icon(Icons.error, color: colorScheme.error),
                backgroundColor: colorScheme.errorContainer,
                actions: [
                  TextButton(
                    onPressed: () => setState(() => _errorMessage = null),
                    child: const Text('DISMISS'),
                  ),
                ],
              ),
            ),

          // Results header.
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8.0),
            child: Row(
              children: [
                Icon(Icons.format_list_bulleted, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  'Speech Segments',
                  style: theme.textTheme.titleSmall?.copyWith(
                    color: colorScheme.primary,
                  ),
                ),
                const Spacer(),
                if (_segments.isNotEmpty)
                  Chip(
                    avatar: const Icon(Icons.graphic_eq, size: 16),
                    label: Text('${_segments.length} segments'),
                  ),
              ],
            ),
          ),

          // Results list.
          Expanded(
            child: _segments.isEmpty
                ? _buildEmptyState(colorScheme, theme)
                : _buildSegmentsList(colorScheme, theme),
          ),
        ],
      ),
    );
  }

  /// Builds the empty state widget shown when no segments are available.
  Widget _buildEmptyState(ColorScheme colorScheme, ThemeData theme) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.audio_file_outlined,
            size: 72,
            color: colorScheme.onSurfaceVariant.withOpacity(0.5),
          ),
          const SizedBox(height: 16),
          Text(
            'No Speech Segments Yet',
            style: theme.textTheme.titleMedium?.copyWith(
              color: colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Pick a WAV file to detect speech segments.',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: colorScheme.onSurfaceVariant.withOpacity(0.7),
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  /// Builds the scrollable list of detected speech segments.
  Widget _buildSegmentsList(ColorScheme colorScheme, ThemeData theme) {
    return ListView.builder(
      itemCount: _segments.length,
      itemBuilder: (BuildContext context, int index) {
        final SpeechTimestamp segment = _segments[index];
        final double startSec = segment.start / _sampleRate;
        final double endSec = segment.end / _sampleRate;
        final double duration = endSec - startSec;

        return Card(
          margin: const EdgeInsets.symmetric(vertical: 4.0),
          child: Padding(
            padding: const EdgeInsets.all(12.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    CircleAvatar(
                      radius: 14,
                      backgroundColor: colorScheme.primaryContainer,
                      child: Text(
                        '${index + 1}',
                        style: theme.textTheme.labelSmall?.copyWith(
                          color: colorScheme.onPrimaryContainer,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        '${_formatDuration(startSec)} - ${_formatDuration(endSec)}',
                        style: theme.textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    Chip(
                      visualDensity: VisualDensity.compact,
                      label: Text(
                        '${duration.toStringAsFixed(2)}s',
                        style: theme.textTheme.labelSmall,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4.0),
                  child: LinearProgressIndicator(
                    value: 1.0, // Full since it's detected speech
                    minHeight: 8,
                    backgroundColor: colorScheme.surfaceContainerHighest,
                    valueColor: AlwaysStoppedAnimation<Color>(
                      colorScheme.primary,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Real-time VAD Tab
// ---------------------------------------------------------------------------

/// Tab for performing real-time VAD on microphone input.
///
/// Uses the [AudioRecorder] package to capture microphone audio and the
/// [VADIterator] class to process audio chunks in a streaming fashion.
class RealtimeVadTab extends StatefulWidget {
  /// Creates the Real-time VAD tab.
  const RealtimeVadTab({super.key});

  @override
  State<RealtimeVadTab> createState() => _RealtimeVadTabState();
}

class _RealtimeVadTabState extends State<RealtimeVadTab> {
  /// The loaded Silero VAD model.
  SileroVadModel? _model;

  /// The VAD iterator for streaming processing.
  VADIterator? _vadIterator;

  /// Audio recorder for capturing microphone input.
  final AudioRecorder _audioRecorder = AudioRecorder();

  /// Whether the model is being loaded.
  bool _isLoadingModel = false;

  /// Whether recording is currently active.
  bool _isRecording = false;

  /// Whether microphone permission was granted.
  bool _hasPermission = false;

  /// Error message to display, if any.
  String? _errorMessage;

  /// Stream of audio amplitude data for the visualizer.
  final StreamController<List<double>> _amplitudeController =
      StreamController<List<double>>.broadcast();

  /// List of real-time detection events.
  final List<VadEvent> _events = [];

  /// Controller for scrolling the events list.
  final ScrollController _scrollController = ScrollController();

  /// Current VAD state (speaking / not speaking).
  bool _isSpeaking = false;

  /// Rolling buffer of recent amplitude values for the visualizer.
  final List<double> _amplitudeBuffer = <double>[];

  /// Maximum size of the amplitude buffer.
  static const int _maxAmplitudeBufferSize = 100;

  /// Subscription to the audio stream.
  StreamSubscription<Uint8List>? _audioSubscription;

  /// Model load progress (0.0 to 1.0).
  double _modelLoadProgress = 0.0;

  /// Sample rate for real-time VAD.
  final int _sampleRate = 16000;

  @override
  void initState() {
    super.initState();
    _loadModel();
    _checkPermission();
  }

  @override
  void dispose() {
    _stopRecording();
    _audioRecorder.dispose();
    _model?.dispose();
    _amplitudeController.close();
    _scrollController.dispose();
    _audioSubscription?.cancel();
    super.dispose();
  }

  /// Checks and requests microphone permission.
  Future<void> _checkPermission() async {
    final PermissionStatus status = await Permission.microphone.request();
    setState(() => _hasPermission = status.isGranted);
  }

  /// Loads the Silero VAD ONNX model from application assets.
  Future<void> _loadModel() async {
    if (_isLoadingModel || _model != null) return;

    setState(() {
      _isLoadingModel = true;
      _errorMessage = null;
      _modelLoadProgress = 0.0;
    });

    try {
      final ByteData modelData = await rootBundle.load(
        'assets/models/silero_vad.onnx',
      );

      setState(() => _modelLoadProgress = 0.3);

      final Directory tempDir = await getTemporaryDirectory();
      final String modelPath = '${tempDir.path}/silero_vad.onnx';
      final File modelFile = File(modelPath);

      if (!modelFile.existsSync()) {
        await modelFile.writeAsBytes(modelData.buffer.asUint8List());
      }

      setState(() => _modelLoadProgress = 0.6);

      final model = await SileroVadModel.fromFile(modelPath);

      // Create VAD iterator for streaming
      final iterator = VADIterator(model, samplingRate: _sampleRate);

      if (!mounted) {
        model.dispose();
        return;
      }

      setState(() {
        _model = model;
        _vadIterator = iterator;
        _isLoadingModel = false;
        _modelLoadProgress = 1.0;
      });
    } on Exception catch (e, stackTrace) {
      if (!mounted) return;
      setState(() {
        _isLoadingModel = false;
        _errorMessage = 'Failed to load VAD model: $e';
      });
      debugPrint('Model load error: $e\n$stackTrace');
    }
  }

  /// Starts microphone recording and VAD processing.
  Future<void> _startRecording() async {
    if (_vadIterator == null || _isRecording) return;

    // Check permission.
    if (!_hasPermission) {
      await _checkPermission();
      if (!_hasPermission) {
        setState(() => _errorMessage = 'Microphone permission denied.');
        return;
      }
    }

    setState(() {
      _isRecording = true;
      _errorMessage = null;
      _events.clear();
      _amplitudeBuffer.clear();
      _isSpeaking = false;
    });

    try {
      // Configure the audio recorder.
      const RecordConfig config = RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
      );

      // Start recording and listen to the stream.
      final Stream<Uint8List> stream =
          await _audioRecorder.startStream(config);

      _audioSubscription = stream.listen(
        _processAudioChunk,
        onError: (Object error) {
          setState(() => _errorMessage = 'Audio stream error: $error');
          _stopRecording();
        },
        onDone: () {
          if (_isRecording) {
            _stopRecording();
          }
        },
      );
    } on Exception catch (e, stackTrace) {
      setState(() {
        _isRecording = false;
        _errorMessage = 'Failed to start recording: $e';
      });
      debugPrint('Recording start error: $e\n$stackTrace');
    }
  }

  /// Stops microphone recording.
  Future<void> _stopRecording() async {
    if (!_isRecording) return;

    await _audioSubscription?.cancel();
    _audioSubscription = null;

    try {
      await _audioRecorder.stop();
    } on Exception catch (e) {
      debugPrint('Error stopping recorder: $e');
    }

    if (!mounted) return;

    setState(() {
      _isRecording = false;
      _isSpeaking = false;
    });
  }

  /// Processes a single chunk of PCM audio data from the microphone.
  void _processAudioChunk(Uint8List pcmData) {
    if (_vadIterator == null) return;

    try {
      // Convert PCM16 to Float32 in the range [-1, 1].
      final Float32List floatData = _pcm16ToFloat32(pcmData);

      // Update amplitude buffer for visualizer.
      _updateAmplitudeBuffer(floatData);

      // Process through VAD iterator using the real API.
      // Process in 512-sample windows
      const int windowSize = 512;
      for (int i = 0; i + windowSize <= floatData.length; i += windowSize) {
        final window = Float32List(windowSize);
        window.setRange(0, windowSize, floatData, i);

        // Call the real VADIterator
        _vadIterator!.process(window).then((result) {
          if (!mounted) return;

          if (result != null) {
            final double timestamp = i / _sampleRate;

            if (result.containsKey('start')) {
              final double confidence = (result['start'] is num)
                  ? (result['start'] as num).toDouble()
                  : 0.8;
              setState(() {
                _isSpeaking = true;
                _events.add(
                  VadEvent.speechStart(
                    timestamp: timestamp + confidence / _sampleRate,
                    confidence: confidence,
                  ),
                );
              });
            } else if (result.containsKey('end')) {
              setState(() {
                _isSpeaking = false;
                _events.add(
                  VadEvent.speechEnd(
                    timestamp: timestamp,
                    duration: timestamp - (_events.isNotEmpty ? _events.last.timestamp : 0),
                  ),
                );
              });
            }

            // Auto-scroll to the latest event.
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (_scrollController.hasClients) {
                _scrollController.animateTo(
                  _scrollController.position.maxScrollExtent,
                  duration: const Duration(milliseconds: 200),
                  curve: Curves.easeOut,
                );
              }
            });
          }
        });
      }
    } on Exception catch (e, stackTrace) {
      debugPrint('Audio chunk processing error: $e\n$stackTrace');
    }
  }

  /// Converts PCM16 bytes to Float32 samples normalized to [-1, 1].
  Float32List _pcm16ToFloat32(Uint8List pcmData) {
    final int sampleCount = pcmData.length ~/ 2;
    final Float32List floatData = Float32List(sampleCount);
    final ByteData byteData = ByteData.sublistView(pcmData);

    for (int i = 0; i < sampleCount; i++) {
      final int sample = byteData.getInt16(i * 2, Endian.little);
      floatData[i] = sample / 32768.0;
    }

    return floatData;
  }

  /// Updates the rolling amplitude buffer with new samples.
  void _updateAmplitudeBuffer(Float32List samples) {
    if (samples.isEmpty) return;

    // Compute RMS amplitude of this chunk.
    double sum = 0.0;
    for (int i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    final double rms = (sum / samples.length).clamp(0.0, 1.0);

    _amplitudeBuffer.add(rms);
    while (_amplitudeBuffer.length > _maxAmplitudeBufferSize) {
      _amplitudeBuffer.removeAt(0);
    }

    if (!_amplitudeController.isClosed) {
      _amplitudeController.add(List<double>.from(_amplitudeBuffer));
    }
  }

  /// Formats seconds into MM:SS.mmm format.
  String _formatTimestamp(double seconds) {
    final DateTime dateTime = DateTime.fromMillisecondsSinceEpoch(
      (seconds * 1000).toInt(),
      isUtc: true,
    );
    return DateFormat('mm:ss.SSS').format(dateTime);
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final ColorScheme colorScheme = theme.colorScheme;

    // Show model loading UI.
    if (_isLoadingModel) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 120,
              height: 120,
              child: CircularProgressIndicator(
                value: _modelLoadProgress > 0 && _modelLoadProgress < 1
                    ? _modelLoadProgress
                    : null,
                strokeWidth: 8,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Loading Silero VAD Model...',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(
              '${(_modelLoadProgress * 100).toInt()}%',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );
    }

    // Show error if model failed to load.
    if (_errorMessage != null && _model == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.error_outline,
                size: 64,
                color: colorScheme.error,
              ),
              const SizedBox(height: 16),
              Text(
                'Error',
                style: theme.textTheme.headlineSmall?.copyWith(
                  color: colorScheme.error,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium,
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _loadModel,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Recording control card.
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Status indicator.
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: 12,
                        height: 12,
                        decoration: BoxDecoration(
                          color: _isRecording
                              ? (_isSpeaking ? Colors.green : colorScheme.error)
                              : colorScheme.outline,
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        _isRecording
                            ? (_isSpeaking ? 'Speech Detected' : 'Listening...')
                            : 'Not Recording',
                        style: theme.textTheme.titleSmall?.copyWith(
                          color: _isRecording
                              ? (_isSpeaking
                                  ? Colors.green
                                  : colorScheme.error)
                              : colorScheme.onSurfaceVariant,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),

                  // Amplitude visualizer.
                  StreamBuilder<List<double>>(
                    stream: _amplitudeController.stream,
                    builder: (
                      BuildContext context,
                      AsyncSnapshot<List<double>> snapshot,
                    ) {
                      final List<double> data =
                          snapshot.data ?? <double>[];
                      return AmplitudeVisualizer(
                        amplitudes: data,
                        isActive: _isRecording,
                      );
                    },
                  ),
                  const SizedBox(height: 16),

                  // Record / Stop button.
                  FilledButton.icon(
                    onPressed: _isRecording ? _stopRecording : _startRecording,
                    style: FilledButton.styleFrom(
                      backgroundColor: _isRecording
                          ? colorScheme.error
                          : colorScheme.primary,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 14,
                      ),
                    ),
                    icon: Icon(
                      _isRecording ? Icons.stop : Icons.mic,
                      size: 24,
                    ),
                    label: Text(
                      _isRecording ? 'Stop Recording' : 'Start Recording',
                      style: theme.textTheme.titleSmall?.copyWith(
                        color: _isRecording
                            ? colorScheme.onError
                            : colorScheme.onPrimary,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Error banner.
          if (_errorMessage != null && _model != null)
            MaterialBanner(
              content: Text(_errorMessage!),
              leading: Icon(Icons.error, color: colorScheme.error),
              backgroundColor: colorScheme.errorContainer,
              actions: [
                TextButton(
                  onPressed: () => setState(() => _errorMessage = null),
                  child: const Text('DISMISS'),
                ),
              ],
            ),

          // Events header.
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8.0),
            child: Row(
              children: [
                Icon(Icons.history, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  'Detection Events',
                  style: theme.textTheme.titleSmall?.copyWith(
                    color: colorScheme.primary,
                  ),
                ),
                const Spacer(),
                if (_events.isNotEmpty)
                  TextButton.icon(
                    onPressed: () => setState(() => _events.clear()),
                    icon: const Icon(Icons.clear_all, size: 18),
                    label: const Text('Clear'),
                  ),
              ],
            ),
          ),

          // Events list.
          Expanded(
            child: _events.isEmpty
                ? _buildEmptyEventsState(colorScheme, theme)
                : _buildEventsList(colorScheme, theme),
          ),
        ],
      ),
    );
  }

  /// Builds the empty state widget for the events list.
  Widget _buildEmptyEventsState(ColorScheme colorScheme, ThemeData theme) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.mic_none_outlined,
            size: 72,
            color: colorScheme.onSurfaceVariant.withOpacity(0.5),
          ),
          const SizedBox(height: 16),
          Text(
            'No Events Yet',
            style: theme.textTheme.titleMedium?.copyWith(
              color: colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Start recording to see real-time VAD events.',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: colorScheme.onSurfaceVariant.withOpacity(0.7),
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  /// Builds the scrollable list of VAD events.
  Widget _buildEventsList(ColorScheme colorScheme, ThemeData theme) {
    return ListView.builder(
      controller: _scrollController,
      itemCount: _events.length,
      itemBuilder: (BuildContext context, int index) {
        final VadEvent event = _events[index];
        return VadEventTile(
          event: event,
          formatTimestamp: _formatTimestamp,
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Amplitude Visualizer
// ---------------------------------------------------------------------------

/// A custom widget that visualizes audio amplitude in real-time.
///
/// Draws a series of vertical bars whose heights correspond to the
/// amplitude values. The bars animate smoothly as new data arrives.
class AmplitudeVisualizer extends StatelessWidget {
  /// Creates an amplitude visualizer.
  const AmplitudeVisualizer({
    required this.amplitudes,
    required this.isActive,
    this.barCount = 40,
    this.barWidth = 4.0,
    this.barSpacing = 2.0,
    super.key,
  });

  /// The list of amplitude values to display (0.0 to 1.0).
  final List<double> amplitudes;

  /// Whether the visualizer should show an active state.
  final bool isActive;

  /// Number of bars to display.
  final int barCount;

  /// Width of each bar in logical pixels.
  final double barWidth;

  /// Spacing between bars in logical pixels.
  final double barSpacing;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colorScheme = Theme.of(context).colorScheme;

    return Container(
      height: 80,
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest.withOpacity(0.5),
        borderRadius: BorderRadius.circular(12),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: CustomPaint(
          size: const Size(double.infinity, 80),
          painter: _AmplitudePainter(
            amplitudes: amplitudes,
            isActive: isActive,
            barCount: barCount,
            barWidth: barWidth,
            barSpacing: barSpacing,
            color: colorScheme.primary,
          ),
        ),
      ),
    );
  }
}

/// Custom painter for the amplitude visualizer bars.
class _AmplitudePainter extends CustomPainter {
  /// Creates the amplitude painter.
  _AmplitudePainter({
    required this.amplitudes,
    required this.isActive,
    required this.barCount,
    required this.barWidth,
    required this.barSpacing,
    required this.color,
  });

  final List<double> amplitudes;
  final bool isActive;
  final int barCount;
  final double barWidth;
  final double barSpacing;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = isActive ? color : color.withOpacity(0.3)
      ..style = PaintingStyle.fill;

    final double totalBarWidth = barCount * (barWidth + barSpacing);
    final double startX = (size.width - totalBarWidth) / 2;
    final double centerY = size.height / 2;

    for (int i = 0; i < barCount; i++) {
      double amplitude = 0.0;

      if (amplitudes.isNotEmpty) {
        // Map bar index to amplitude index.
        final int amplitudeIndex =
            ((i / barCount) * amplitudes.length).toInt().clamp(
                  0,
                  amplitudes.length - 1,
                );
        amplitude = amplitudes[amplitudeIndex];
      }

      // Add a small idle animation when no audio.
      if (!isActive || amplitude < 0.02) {
        amplitude = 0.02 + (i % 3) * 0.01;
      }

      final double barHeight =
          (amplitude * size.height).clamp(2.0, size.height);
      final double x = startX + i * (barWidth + barSpacing);

      final RRect bar = RRect.fromRectAndRadius(
        Rect.fromLTWH(
          x,
          centerY - barHeight / 2,
          barWidth,
          barHeight,
        ),
        const Radius.circular(2),
      );

      canvas.drawRRect(bar, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _AmplitudePainter oldDelegate) {
    return oldDelegate.amplitudes != amplitudes ||
        oldDelegate.isActive != isActive;
  }
}

// ---------------------------------------------------------------------------
// VAD Event Tile
// ---------------------------------------------------------------------------

/// A list tile widget displaying a single VAD event.
class VadEventTile extends StatelessWidget {
  /// Creates a VAD event tile.
  const VadEventTile({
    required this.event,
    required this.formatTimestamp,
    super.key,
  });

  /// The VAD event to display.
  final VadEvent event;

  /// Function to format timestamp values.
  final String Function(double) formatTimestamp;

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final ColorScheme colorScheme = theme.colorScheme;

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 2.0),
      color: event.isSpeechStart
          ? colorScheme.primaryContainer.withOpacity(0.5)
          : colorScheme.surfaceContainerHighest.withOpacity(0.3),
      child: ListTile(
        dense: true,
        leading: CircleAvatar(
          radius: 14,
          backgroundColor: event.isSpeechStart
              ? Colors.green.withOpacity(0.2)
              : Colors.orange.withOpacity(0.2),
          child: Icon(
            event.isSpeechStart ? Icons.record_voice_over : Icons.voice_over_off,
            size: 16,
            color: event.isSpeechStart ? Colors.green : Colors.orange,
          ),
        ),
        title: Text(
          event.isSpeechStart ? 'Speech Started' : 'Speech Ended',
          style: theme.textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        subtitle: Text(
          'At ${formatTimestamp(event.timestamp)}'
          '${event.duration != null ? ' | Duration: ${event.duration!.toStringAsFixed(2)}s' : ''}'
          '${event.confidence != null ? ' | Confidence: ${(event.confidence! * 100).toStringAsFixed(0)}%' : ''}',
          style: theme.textTheme.bodySmall?.copyWith(
            color: colorScheme.onSurfaceVariant,
          ),
        ),
        trailing: event.isSpeechStart
            ? const Icon(Icons.arrow_upward, size: 18, color: Colors.green)
            : const Icon(Icons.arrow_downward, size: 18, color: Colors.orange),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Data Models
// ---------------------------------------------------------------------------

/// A single VAD event (speech start or end).
class VadEvent {
  /// Creates a speech-start event.
  const VadEvent.speechStart({
    required this.timestamp,
    this.confidence,
  })  : isSpeechStart = true,
        duration = null;

  /// Creates a speech-end event.
  const VadEvent.speechEnd({
    required this.timestamp,
    required this.duration,
  })  : isSpeechStart = false,
        confidence = null;

  /// Whether this event represents the start of speech.
  final bool isSpeechStart;

  /// Timestamp of the event in seconds.
  final double timestamp;

  /// Duration of the speech segment (only for speech-end events).
  final double? duration;

  /// Confidence score (only for speech-start events).
  final double? confidence;
}
