import unittest
from unittest.mock import patch, mock_open, MagicMock

from build_a_long.bounding_box_extractor.main import extract_bounding_boxes
from build_a_long.bounding_box_extractor.bbox import BBox


class TestBoundingBoxExtractor(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("build_a_long.bounding_box_extractor.main.fitz.open")
    def test_extract_bounding_boxes_basic(
        self, mock_fitz_open, mock_json_dump, mock_file_open
    ):
        # Create a dummy PDF path for testing
        dummy_pdf_path = "/path/to/dummy.pdf"

        # Build a fake document with 1 page and simple rawdict content
        fake_page = MagicMock()
        fake_page.get_text.return_value = {
            "blocks": [
                {  # text block representing a numeric instruction number
                    "type": 0,
                    "bbox": [10, 20, 30, 40],
                    "lines": [
                        {
                            "spans": [
                                {"text": "1"},
                            ]
                        }
                    ],
                },
                {  # image block
                    "type": 1,
                    "bbox": [50, 60, 150, 200],
                },
            ]
        }

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 1
        # __getitem__ for index 0 returns our fake page

        def _getitem(idx):
            assert idx == 0
            return fake_page

        fake_doc.__getitem__.side_effect = _getitem
        mock_fitz_open.return_value = fake_doc

        # Call the function
        extract_bounding_boxes(dummy_pdf_path)

        # Assert that json.dump was called with the expected structure
        mock_json_dump.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        extracted_data = args[0]

        self.assertIn("pages", extracted_data)
        self.assertEqual(len(extracted_data["pages"]), 1)
        elements = extracted_data["pages"][0]["elements"]
        # Expect two elements: a text classified as instruction_number and an image
        self.assertEqual(len(elements), 2)
        self.assertEqual(elements[0]["type"], "instruction_number")
        self.assertEqual(elements[0]["bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(elements[1]["type"], "image")

        # Assert that the output file was attempted to be opened with 'w'
        expected_output_filename = dummy_pdf_path.replace(".pdf", ".json")
        mock_file_open.assert_called_once_with(expected_output_filename, "w")


class TestBBox(unittest.TestCase):
    def test_overlaps(self):
        bbox1 = BBox(0, 0, 10, 10)
        bbox2 = BBox(5, 5, 15, 15)
        bbox3 = BBox(10, 10, 20, 20)  # Touches at corner
        bbox4 = BBox(11, 11, 20, 20)  # No overlap
        bbox5 = BBox(0, 0, 10, 5)  # Partial overlap

        self.assertTrue(bbox1.overlaps(bbox2))
        self.assertTrue(bbox2.overlaps(bbox1))
        self.assertFalse(bbox1.overlaps(bbox3))  # Touching at corner is not overlapping
        self.assertFalse(bbox3.overlaps(bbox1))
        self.assertFalse(bbox1.overlaps(bbox4))
        self.assertTrue(bbox1.overlaps(bbox5))

    def test_fully_inside(self):
        bbox1 = BBox(0, 0, 10, 10)
        bbox2 = BBox(2, 2, 8, 8)
        bbox3 = BBox(0, 0, 10, 10)  # Same bbox
        bbox4 = BBox(0, 0, 10, 11)  # Not fully inside

        self.assertTrue(bbox2.fully_inside(bbox1))
        self.assertTrue(bbox3.fully_inside(bbox1))
        self.assertFalse(bbox1.fully_inside(bbox2))
        self.assertFalse(bbox4.fully_inside(bbox1))

    def test_adjacent(self):
        bbox1 = BBox(0, 0, 10, 10)
        bbox2 = BBox(10, 0, 20, 10)  # Right adjacent
        bbox3 = BBox(0, 10, 10, 20)  # Top adjacent
        bbox4 = BBox(10, 10, 20, 20)  # Corner adjacent
        bbox5 = BBox(11, 0, 20, 10)  # Not adjacent
        bbox6 = BBox(5, 5, 15, 15)  # Overlapping, not adjacent

        self.assertTrue(bbox1.adjacent(bbox2))
        self.assertTrue(bbox2.adjacent(bbox1))
        self.assertTrue(bbox1.adjacent(bbox3))
        self.assertTrue(bbox3.adjacent(bbox1))
        self.assertFalse(
            bbox1.adjacent(bbox4)
        )  # Corner adjacency is not considered adjacent
        self.assertFalse(bbox1.adjacent(bbox5))
        self.assertFalse(bbox1.adjacent(bbox6))


if __name__ == "__main__":
    unittest.main()
