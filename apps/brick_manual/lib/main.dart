import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:pdfrx/pdfrx.dart';
import 'package:brick_manual/instruction_metadata.dart';
import 'package:brick_manual/lego_set_repository.dart';
import 'package:dio/dio.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BrickManual',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  final LegoSetRepository? repository;

  const HomeScreen({super.key, this.repository});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late final LegoSetRepository _repository;
  late final Future<List<InstructionMetadata>> _allSetsFuture;
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? LegoSetRepositoryImpl(dio: Dio());
    _allSetsFuture = _loadLegoSets();
  }

  Future<List<InstructionMetadata>> _loadLegoSets() async {
    final manifest = await _repository.fetchManifest();
    final List<InstructionMetadata> allSets = [];
    for (final indexUrl in manifest) {
      final sets = await _repository.fetchIndexFile(indexUrl: indexUrl);
      allSets.addAll(sets);
    }
    return allSets;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('BrickManual')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextField(
              decoration: InputDecoration(
                hintText: 'Search for LEGO set by number or name...',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              onChanged: (value) {
                setState(() {
                  _searchQuery = value;
                });
              },
            ),
          ),
          Expanded(
            child: FutureBuilder<List<InstructionMetadata>>(
              future: _allSetsFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Center(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Text('Error: ${snapshot.error}', style: const TextStyle(color: Colors.red)),
                    ),
                  );
                } else if (snapshot.hasData) {
                  final allSets = snapshot.data!;
                  final filteredSets = allSets.where((set) {
                    final nameLower = set.name?.toLowerCase() ?? '';
                    final setNumberLower = set.set.toLowerCase();
                    final queryLower = _searchQuery.toLowerCase();
                    return nameLower.contains(queryLower) ||
                        setNumberLower.contains(queryLower);
                  }).toList();

                  if (filteredSets.isEmpty) {
                    return const Center(child: Text('No sets found.'));
                  }

                  return ListView.builder(
                    itemCount: filteredSets.length,
                    itemBuilder: (context, index) {
                      final set = filteredSets[index];
                      return ListTile(
                        leading: set.setImageUrl != null
                            ? Image.network(
                                set.setImageUrl!,
                                width: 50,
                                height: 50,
                                fit: BoxFit.cover,
                                errorBuilder: (context, error, stackTrace) =>
                                    const Icon(Icons.error),
                              )
                            : const Icon(Icons.category),
                        title: Text('${set.name} (${set.set})'),
                        subtitle: Text('Pieces: ${set.pieces}'),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => const PdfViewerScreen(
                                pdfPath: 'assets/sample.pdf',
                              ),
                            ),
                          );
                        },
                      );
                    },
                  );
                } else {
                  return const Center(child: Text('No sets available.'));
                }
              },
            ),
          ),
        ],
      ),
    );
  }
}

class PdfViewerScreen extends StatefulWidget {
  final String pdfPath;

  const PdfViewerScreen({super.key, required this.pdfPath});

  @override
  State<PdfViewerScreen> createState() => _PdfViewerScreenState();
}

class _PdfViewerScreenState extends State<PdfViewerScreen> {
  final _controller = PdfViewerController();
  int? _currentPage;
  int? _pageCount;
  late final Widget _pdfViewer;

  @override
  void initState() {
    super.initState();
    _pdfViewer = PdfViewer.asset(
      widget.pdfPath,
      controller: _controller,
      params: PdfViewerParams(
        onPageChanged: (page) {
          setState(() {
            if (page != null) {
              _currentPage = page;
            }
          });
        },
        onViewerReady: (document, controller) {
          setState(() {
            _currentPage = controller.pageNumber;
            _pageCount = controller.pageCount;
          });
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('PDF Viewer'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 20.0),
            child: Center(
              child: Text('Page: ${_currentPage ?? 0} / ${_pageCount ?? 0}'),
            ),
          ),
        ],
      ),
      body: _pdfViewer,
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            onPressed: () {
              if (_currentPage != null && _currentPage! > 1) {
                _controller.goToPage(pageNumber: _currentPage! - 1);
              }
            },
            child: const Icon(Icons.arrow_back),
          ),
          const SizedBox(width: 8),
          FloatingActionButton(
            onPressed: () {
              if (_currentPage != null &&
                  _pageCount != null &&
                  _currentPage! < _pageCount!) {
                _controller.goToPage(pageNumber: _currentPage! + 1);
              }
            },
            child: const Icon(Icons.arrow_forward),
          ),
        ],
      ),
    );
  }
}
