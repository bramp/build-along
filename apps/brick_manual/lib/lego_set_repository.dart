import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:brick_manual/lego_set_metadata.dart';

const String BASE_URL = 'https://lego.bramp.net';
const String MANIFEST_URL = '$BASE_URL/manifest.json';

/// Abstract class for a repository that fetches LEGO set data.
/// This allows for easy mocking in tests.
abstract class LegoSetRepository {
  Future<List<String>> fetchManifest();
  Future<List<LegoSetMetadata>> fetchIndexFile({
    required String indexUrl,
    void Function(int count, int total)? onReceiveProgress,
  });
}

/// The real implementation of [LegoSetRepository] that uses dio to fetch data.
class LegoSetRepositoryImpl implements LegoSetRepository {
  LegoSetRepositoryImpl({required this.dio});

  final Dio dio;

  @override
  Future<List<String>> fetchManifest() async {
    try {
      final response = await dio.get(MANIFEST_URL);
      if (response.statusCode == 200 && response.data is List) {
        return List<String>.from(response.data);
      } else {
        throw Exception('Failed to load manifest: Status code ${response.statusCode}');
      }
    } on DioException catch (e) {
      throw Exception('Failed to load manifest: $e');
    } catch (e) {
      throw Exception('Failed to load manifest: $e');
    }
  }

  @override
  Future<List<LegoSetMetadata>> fetchIndexFile({
    required String indexUrl,
    void Function(int count, int total)? onReceiveProgress,
  }) async {
    try {
      final response = await dio.get(
        '$BASE_URL/$indexUrl',
        options: Options(responseType: ResponseType.bytes),
        onReceiveProgress: onReceiveProgress,
      );

      if (response.statusCode == 200 && response.data != null) {
        final String responseBody = utf8.decode(response.data as List<int>);
        final List<dynamic> data = json.decode(responseBody);
        return data.map((i) => LegoSetMetadata.fromJson(i)).toList();
      } else {
        throw Exception('Failed to load index file $indexUrl: Status code ${response.statusCode}');
      }
    } on DioException catch (e) {
      throw Exception('Failed to load index file $indexUrl: $e');
    } catch (e) {
      throw Exception('Failed to load index file $indexUrl: $e');
    }
  }
}
