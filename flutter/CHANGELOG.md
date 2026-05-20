# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial implementation of Silero VAD for Flutter using ONNX Runtime.
- `SileroVad` class for loading and running the Silero VAD ONNX model.
- `VadIterator` class for real-time voice activity detection with stateful processing.
- `SpeechSegment` model for representing detected speech segments with timestamps.
- `VadState` model for managing the hidden state of the VAD model.
- `VadConfig` for configuring VAD parameters (threshold, sample rate, min speech duration, etc.).
- `VadSession` for ONNX session management with automatic resource cleanup.
- File-based VAD processing with configurable chunk sizes.
- Real-time VAD processing via Dart Streams.
- Flutter example app with two tabs: "File VAD" and "Real-time VAD".
- Support for 16 kHz and 8 kHz sample rates.
- GitHub Actions CI workflow for automated testing and analysis.
- Model download script for fetching the official Silero VAD ONNX model.

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- N/A (initial release)

## [0.1.0] - 2024-01-15

### Added

- First public release of `silero_vad_flutter`.
