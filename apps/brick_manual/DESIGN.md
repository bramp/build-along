# BrickManual Design Document

This document outlines the design and architecture of the BrickManual Flutter application.

## 1. Overview

BrickManual is a Flutter application designed to enhance the experience of building with LEGO by providing a feature-rich viewer for LEGO instruction manuals.

## 2. Core Features

- Display a list of in-progress LEGO sets.
- Search for new LEGO sets by set number or name.
- Fetch official PDF instructions and JSON metadata from the internet, with a progress indicator.
- Cache the downloaded index and other assets to minimize network usage.
- Verify that the PDF and metadata are from the same source using a hash check.
- Display and view PDF instruction manuals.
- Advanced navigation within a PDF, including zooming to specific, pre-calculated areas on a page.
- Timer to track build sessions.
- Statistics and reporting on build times.
- Cross-platform support (iOS, Android, Web, Desktop).

## 3. Architecture

- **State Management**: We'll use `ChangeNotifier` and `Provider` for simple and effective state management.
- **UI**: We will use Material Design 3 for a modern and consistent look and feel.
- **PDF Rendering**: We will use the `pdfrx` package.
- **Data Fetching**: We will use the `dio` package, as it provides support for download progress tracking, which is a core requirement.
- **Testability**: Data fetching logic will be encapsulated in a `LegoSetRepository` class. This allows the UI to be tested independently of the network layer by using mock implementations of the repository.

## 4. Key Design Decisions

### PDF Viewer Library: `pdfrx`

After evaluating several PDF viewer libraries, `pdfrx` was chosen for the following reasons:

- **"Zoom to Rectangle" Feature**: It provides a direct `zoomToRect()` API, which is a core requirement for the app's navigation. This is a significant advantage over other libraries that would require manual implementation of this feature.
- **Cross-Platform Support**: It supports all target platforms: Android, iOS, Windows, macOS, Linux, and Web.
- **Licensing**: It is licensed under the MIT License, which aligns with the project's open-source goals.

## 5. Data Model and Fetching

### 5.1 Python/Dart Model Sharing

To maintain velocity, the `InstructionMetadata` and `PdfEntry` data models are currently duplicated manually in Dart (`lib/instruction_metadata.dart`) from their Python counterparts (`src/build_a_long/downloader/metadata.py`).

**TODO**: Investigate and implement a code generation solution (e.g., Protocol Buffers) to automate the synchronization of these models between Python and Dart, ensuring consistency and reducing manual maintenance.

### 5.2 Indexing and Search

To avoid maintaining a separate backend service and to ensure a responsive user experience, we will use a segmented static JSON index.

- A `manifest.json` file will be hosted at the root of the static server (e.g., `lego.bramp.net/manifest.json`). This file will contain a list of all the yearly index files (e.g., `["index-2023.json", "index-2022.json", ...]`).
- On startup, the Flutter app will first download the `manifest.json` file.
- It will then sequentially download each yearly index file listed in the manifest.
- After each yearly index is downloaded and parsed, the new sets will be added to the app's main list, and the search results will update in real-time.
- This provides a fast initial load while progressively enhancing the available data and search results.

### 5.3 Caching and Download Progress

- The manifest and yearly index files will be cached on the device to avoid re-fetching them on every app start. A simple time-based expiration policy (e.g., 24 hours) will be used.
- When downloading the index files, a progress bar will be displayed at the bottom of the main screen, indicating the overall progress of fetching all the yearly files.
- When downloading large files like instruction PDFs, a progress indicator will also be shown. This requires an HTTP client that supports progress callbacks, hence the choice of `dio`.

## 6. TODO List

- **Phase 1: Basic PDF Viewer**
    - [x] Create a new Flutter project.
    - [x] Add the `pdfrx` PDF viewing package.
    - [x] Create a home screen that lists dummy PDF files.
    - [x] Implement navigation to a PDF viewer screen.
    - [x] Display a sample PDF file.
    - [x] Implement basic page navigation buttons.

- **Phase 2: Data Model & Home Screen**
    - [x] Define the data models for a LEGO set.
    - [x] Update the home screen to display a list of in-progress sets (using dummy data for now).
    - [x] Add a search box to the home screen.

- **Phase 3: Refactor to Repository & Segmented Data Fetching**
    - [x] Replace the `http` package with `dio`.
    - [x] Create a `LegoSetRepository` class to encapsulate all data fetching logic.
    - [x] Update `HomeScreen` to use the repository.
    - [x] Update the widget test to use a mock version of the repository.
    - [ ] Modify the repository to support fetching a manifest file and then fetching yearly index files based on the manifest.
    - [ ] Update the `HomeScreen` to orchestrate the segmented download, update the UI with progress, and populate the search list as data arrives.

- **Phase 4: Caching & PDF Downloads**
    - [ ] Implement caching for the downloaded index files.
    - [ ] When a set is selected, implement fetching its specific PDF and JSON metadata with progress indication.
    - [ ] Implement the hash check between the downloaded PDF and its metadata.
    - [ ] Save the downloaded files to the device.

- **Phase 5: Advanced PDF Navigation**
    - [ ] Load the JSON metadata for a PDF.
    - [ ] Use the `zoomToRect()` feature of `pdfrx` to navigate to specific areas defined in the metadata.

- **Phase 6: Timer and Progress Tracking**
    - [ ] Add a timer to the viewer screen.
    - [ ] Save the elapsed time for each session.
