// Dart equivalent of src/build_a_long/downloader/models.py

/// Represents a single instruction PDF file.
/// Corresponds to Python's `PdfEntry` in `src/build_a_long/downloader/models.py`.
class PdfEntry {
  const PdfEntry({
    required this.url,
    required this.filename,
    this.previewUrl,
    this.filesize,
    this.filehash,
  });

  final String url;
  final String filename;
  final String? previewUrl;
  final int? filesize;
  final String? filehash; // SHA256 hash of the file content

  factory PdfEntry.fromJson(Map<String, dynamic> json) {
    return PdfEntry(
      url: json['url'],
      filename: json['filename'],
      previewUrl: json['preview_url'],
      filesize: json['filesize'],
      filehash: json['filehash'],
    );
  }
}

/// Complete metadata for a LEGO set.
/// Corresponds to Python's `LegoSetMetadata` in `src/build_a_long/downloader/models.py`.
class LegoSetMetadata {
  const LegoSetMetadata({
    required this.set,
    required this.locale,
    this.name,
    this.theme,
    this.age,
    this.pieces,
    this.year,
    this.setImageUrl,
    this.lastUpdated,
    this.pdfs = const [],
  });

  final String set;
  final String locale;
  final String? name;
  final String? theme;
  final String? age;
  final int? pieces;
  final int? year;
  final String? setImageUrl;
  final DateTime? lastUpdated;
  final List<PdfEntry> pdfs;

  // TODO: Automate generation of this class and PdfEntry from Python models
  // to ensure consistency and reduce manual effort.

  factory LegoSetMetadata.fromJson(Map<String, dynamic> json) {
    var pdfsFromJson = json['pdfs'] as List<dynamic>?;
    List<PdfEntry> pdfsList = pdfsFromJson != null
        ? pdfsFromJson.map((i) => PdfEntry.fromJson(i)).toList()
        : [];

    return LegoSetMetadata(
      set: json['set'],
      locale: json['locale'],
      name: json['name'],
      theme: json['theme'],
      age: json['age'],
      pieces: json['pieces'],
      year: json['year'],
      setImageUrl: json['set_image_url'],
      lastUpdated: json['_last_updated'] != null
          ? DateTime.tryParse(json['_last_updated'] as String)
          : null,
      pdfs: pdfsList,
    );
  }
}
