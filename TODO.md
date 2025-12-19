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
- [x] In the metadata for the pdfs, can you add a hash (if the file is
  downloaded and accessable on the disk), the file size in bytes.
- [ ] In the apollo lego.com metadata, there is a isAdditionalInfoBooklet, and
  sequence field. We should fetch them appropriately.
- [x] Fix the integration tests (or figure out a way to run on demand)

## Phase 3: Bounding Box Extractor

- [X] Choose a library for PDF processing (e.g., PyMuPDF/fitz) to split PDFs into pages/images.
- [X] Research and decide on a strategy for bounding box extraction:
  - **Option A: Inspect PDF object model directly to find bounding boxes of images, text, and shapes. Then use targeted OCR or image analysis.**
  - Option B: Use a pre-trained object detection model.
  - Option C: Use traditional computer vision techniques with a library like OpenCV.
- [ ] Implement the logic to identify and extract bounding boxes for:
  - Instruction numbers
  - Parts lists
  - Build steps
- [X] Define a data format for storing the bounding box information (e.g., JSON).
- [ ] Implement the script to process a PDF and save the extracted data.

## Phase 4: Instruction Viewer (Flutter Mobile App)

- [ ] Set up a Flutter project for the mobile application.
- [ ] Design the UI for the viewer, including:
  - A view to display the PDF page.
  - "Next" and "Back" buttons for navigation.
- [ ] The app starts with a simple input box to select the set number. Upon submission, it
  fetches the PDF and bounding box data from the backend.
  - [ ] Later, the app should display a index, have a search, etc.
  - [ ] Support scanning Lego QR codes (todo figure out format)
- [ ] Implement a backend service (e.g., in Python) to serve PDF pages and bounding box data to the Flutter app, or explore porting the logic to Dart.
- [ ] Implement the logic in the Flutter app to load and display a PDF page.
- [ ] Implement the pan-and-zoom functionality to focus on the current instruction's bounding box.
-     [ ] The pan-and-zoom should animate smoothly to the new bounding box when navigating steps.
-     [ ] A "next" button uses the bounding box data to zoom in on the next instruction.
- [ ] The PDF is displayed with overlays, that help highlight information on the page.
- [ ] The app should have a "index" that allows users to select a part and it'll
  tell them everywhere it appears in the book.
- [ ] The app should also provide extra information about the parts, such as colors,
  alternate part numbers, and links to buy them online.

## Phase 5: Integration and Refinements

- [ ] Integrate the three tools into a cohesive application.
- [ ] Add error handling and logging.
- [ ] Write unit and integration tests.
- [ ] Document the code and usage.

## Data Storage Optimization

- [ ] Compress all PDF files in the `data` directory and its subdirectories using `zstd -19`. This will create `.pdf.zst` files, and the original `.pdf` files should be deleted after successful compression. Decompression will be required to view the PDFs.

**Compression Results Summary (for reference):**

*   **`qpdf` (Lossless PDF Optimization):** Reorganizes PDF structure for smaller, usable PDFs. Achieved ~1.2% size reduction for a sample file.
*   **General-Purpose Lossless Compressors:** Create archive files (.bz2, .zst, .xz) requiring decompression.
    *   **Original Sample File:** 1,655,192 bytes
    *   **`bzip2 -9`:** 1,634,964 bytes (~1.2% reduction)
    *   **`xz -9`:** 1,625,800 bytes (~1.8% reduction)
    *   **`zstd -19` (Winner):** 1,623,925 bytes (~1.9% reduction) - Provided the best lossless compression for the sample.

## Future: Image-Based Recognition

Some LEGO instruction elements are rendered as raster images rather than vector drawings.
These currently cannot be detected by the classifier and would require image analysis
(OCR, pattern recognition, or ML-based detection) to identify.

Known examples:
- [ ] **Image-based arrows** - Some arrows are rendered as images (shaft + arrowhead).
      Example: Page 31 of document 6433200 (fixture: `6433200_page_031_raw.json`)
      has an arrow coming from the left, bending up to point at a diagram.
      The arrow consists of Image blocks (id=11 as shaft, id=12 as arrowhead)
      rather than Drawing blocks.

This would be a broader effort covering:
- Detecting arrow shapes in images
- OCR for text embedded in images
- Pattern recognition for other visual elements (rotation symbols, etc.)

## Misc

- [x] pytest is in requirements.txt but should it instead be in requirements-dev.txt?