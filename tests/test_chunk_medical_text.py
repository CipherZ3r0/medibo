"""
Tests for medical chunking (Phase 4).
"""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import chunk_medical_text as cm


class TestMedicalChunking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.proc = Path(self.tmp.name) / 'processed'
        self.chunks = Path(self.tmp.name) / 'chunks'
        self.proc.mkdir()
        # create sample processed doc
        pd = {
            'doc_id': 'test-doc',
            'title': 'Test Doc',
            'indexed_text': '[page 1]\nIntro text here. More info follows.\n\n[page 2]\nSecond page content for testing.\n',
            'source': {'source': 'icmr', 'region': 'india', 'authority': 'ICMR', 'license': 'Public'},
            'manifest_document': {
                'url': 'https://www.icmr.gov.in/test.pdf',
                'document_type': 'guideline',
                'topic': 'Test topic',
                'published_at': '2020',
                'updated_at': None,
            },
            'pages': [1,2],
            'sections': [],
            'raw_text_path': 'data/medical/raw/icmr/test-doc/raw_text.txt',
        }
        (self.proc / 'test-doc.json').write_text(json.dumps(pd), encoding='utf-8')
        cm.PROCESSED_DIR = self.proc
        cm.CHUNKS_DIR = self.chunks

    def test_chunk_generation_metadata_and_ids(self):
        res = cm.chunk_all()
        ch = json.load(open(self.chunks / 'medical_chunks.json'))
        self.assertGreater(len(ch), 0)
        ids = [x['chunk_id'] for x in ch]
        self.assertEqual(len(ids), len(set(ids)))
        c = ch[0]
        for k in ['chunk_id','text','doc_id','title','source','source_url','region','authority','document_type','disease_topic','published_at','updated_at','page_number','section','dataset_type']:
            self.assertIn(k, c)
        self.assertEqual(c['dataset_type'], 'default')

    def test_page_preserved(self):
        res = cm.chunk_all()
        ch = json.load(open(self.chunks / 'medical_chunks.json'))
        pages = set(c['page_number'] for c in ch)
        self.assertTrue(pages.intersection({1,2}))


if __name__ == '__main__':
    unittest.main()
