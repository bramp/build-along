import 'package:brick_manual/instruction_metadata.dart';
import 'package:brick_manual/lego_set_repository.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';

import 'package:brick_manual/main.dart';
import 'widget_test.mocks.dart';

@GenerateMocks([LegoSetRepository])
void main() {
  late MockLegoSetRepository mockRepository;

  setUp(() {
    mockRepository = MockLegoSetRepository();
  });

  testWidgets('HomeScreen loads and displays sets on success', (WidgetTester tester) async {
    // Define our dummy data
    final manifest = ['index-1.json'];
    final sets1 = [const InstructionMetadata(set: '1', locale: 'en', name: 'Set 1')];

    // Stub the repository methods
    when(mockRepository.fetchManifest()).thenAnswer((_) async => manifest);
    when(mockRepository.fetchIndexFile(indexUrl: 'index-1.json')).thenAnswer((_) async => sets1);

    // Build our app
    await tester.pumpWidget(MaterialApp(
      home: HomeScreen(repository: mockRepository),
    ));

    // At the start, we should see a loading indicator.
    expect(find.byType(CircularProgressIndicator), findsOneWidget);

    // Let all futures complete and the UI settle.
    await tester.pumpAndSettle();

    // We should see the set and the loading UI should be gone.
    expect(find.text('Set 1'), findsOneWidget);
    expect(find.byType(CircularProgressIndicator), findsNothing);
  });

  testWidgets('HomeScreen displays error message on failure', (WidgetTester tester) async {
    // Stub the fetchManifest method to throw an error.
    when(mockRepository.fetchManifest()).thenAnswer((_) async => throw Exception('Failed to load'));

    // Build our app
    await tester.pumpWidget(MaterialApp(
      home: HomeScreen(repository: mockRepository),
    ));

    // At the start, we should see a loading indicator.
    expect(find.byType(CircularProgressIndicator), findsOneWidget);

    // Pump a frame to allow the Future to complete.
    await tester.pumpAndSettle();

    // After loading, we should see an error message.
    expect(find.byType(CircularProgressIndicator), findsNothing);
    expect(find.textContaining('Error:'), findsOneWidget);
  });

  testWidgets('HomeScreen filters list on search', (WidgetTester tester) async {
    // Define our dummy data
    final manifest = ['index-1.json'];
    final sets = [
      const InstructionMetadata(set: '1', locale: 'en', name: 'TIE Fighter'),
      const InstructionMetadata(set: '2', locale: 'en', name: 'X-Wing'),
    ];

    // Stub the repository methods
    when(mockRepository.fetchManifest()).thenAnswer((_) async => manifest);
    when(mockRepository.fetchIndexFile(indexUrl: 'index-1.json')).thenAnswer((_) async => sets);

    // Build our app and wait for it to load
    await tester.pumpWidget(MaterialApp(
      home: HomeScreen(repository: mockRepository),
    ));
    await tester.pumpAndSettle();

    // Initially, both sets should be visible
    expect(find.textContaining('TIE Fighter'), findsOneWidget);
    expect(find.textContaining('X-Wing'), findsOneWidget);

    // Find the search field and enter text
    final searchField = find.byType(TextField);
    await tester.enterText(searchField, 'TIE');
    await tester.pump(); // Rebuild the widget with the search query

    // Now, only the TIE Fighter should be visible
    expect(find.textContaining('TIE Fighter'), findsOneWidget);
    expect(find.textContaining('X-Wing'), findsNothing);
  });
}
