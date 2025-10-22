# TODO: Lego Instructions Project

This file outlines the tasks required to build the Lego instructions project.

## Phase 1: Project Setup

- [x] Initialize the project with the Pants build system.
- [x] Create `pants.toml` with initial configuration.
- [x] Set up the basic directory structure for the project (`src/lego_instructions`).
- [x] Create initial `BUILD` files for the main components.

## Phase 2: PDF Downloader

- [x] Research and determine the source for Lego instruction PDFs (e.g., official website, APIs).
- [x] Implement a script to download PDFs based on a Lego set number.
- [x] Add command-line interface to the downloader script.
- [x] Define a storage location for downloaded PDFs (e.g., a `data/` directory).
- [x] Don't download a pdf if it already exists locally.
- [x] Add a progress indicator for downloads.

## Phase 3: Bounding Box Extractor

- [ ] Choose a library for PDF processing (e.g., PyMuPDF/fitz) to split PDFs into pages/images.
- [ ] Research and decide on a strategy for bounding box extraction:
  - Option A: Inspect PDF object model directly to find bounding boxes of images, text, and shapes. Then use targeted OCR or image analysis.
  - Option B: Use a pre-trained object detection model.
  - Option C: Use traditional computer vision techniques with a library like OpenCV.
- [ ] Implement the logic to identify and extract bounding boxes for:
  - Instruction numbers
  - Parts lists
  - Build steps
- [ ] Define a data format for storing the bounding box information (e.g., JSON).
- [ ] Implement the script to process a PDF and save the extracted data.

## Phase 4: Instruction Viewer (Flutter Mobile App)

- [ ] Set up a Flutter project for the mobile application.
- [ ] Design the UI for the viewer, including:
  - A view to display the PDF page.
  - "Next" and "Back" buttons for navigation.
- [ ] Implement a backend service (e.g., in Python) to serve PDF pages and bounding box data to the Flutter app, or explore porting the logic to Dart.
- [ ] Implement the logic in the Flutter app to load and display a PDF page.
- [ ] Implement the pan-and-zoom functionality to focus on the current instruction's bounding box.

## Phase 5: Integration and Refinements

- [ ] Integrate the three tools into a cohesive application.
- [ ] Add error handling and logging.
- [ ] Write unit and integration tests.
- [ ] Document the code and usage.
