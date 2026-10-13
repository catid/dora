"""Concurrent confirmation startup must not race the historical split writer."""
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import init_confirm_extraction as runner


def worker(directory,barrier,queue):
    args=SimpleNamespace(output=Path(directory))
    def historical_writer(args,tokenizer):
        path=args.output/'split_manifest.json';temporary=path.with_suffix('.json.tmp')
        temporary.write_text('stable manifest\n')
        time.sleep(.05)
        temporary.replace(path)
        return {'train':[1]}, {'digest':'stable'}
    try:
        barrier.wait()
        with patch.object(runner.task,'data',historical_writer):
            result=runner.load_records(args,None)
        queue.put(('pass',result))
    except BaseException as error:
        queue.put(('fail',str(error)))


class ConcurrentStartupTests(unittest.TestCase):
    def test_parallel_seed_shards_serialize_split_manifest_writer(self):
        context=multiprocessing.get_context('fork')
        with TemporaryDirectory() as directory:
            barrier=context.Barrier(5);queue=context.Queue()
            children=[context.Process(target=worker,args=(directory,barrier,queue)) for _ in range(5)]
            for child in children:child.start()
            results=[queue.get(timeout=15) for _ in children]
            for child in children:
                child.join(15)
                self.assertEqual(child.exitcode,0)
            self.assertEqual(results,[('pass',({'train':[1]},{'digest':'stable'}))]*5)
            self.assertEqual((Path(directory)/'split_manifest.json').read_text(),'stable manifest\n')
            self.assertFalse((Path(directory)/'split_manifest.json.tmp').exists())


if __name__=='__main__':unittest.main()
