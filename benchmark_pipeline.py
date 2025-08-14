"""Benchmark pipeline runner."""

import asyncio
import logging

from src.pipeline import BenchmarkOrchestrator

logging.basicConfig(level=logging.INFO)
# logging.getLogger('src.synthetic_pdf').setLevel(logging.DEBUG)


async def main():
    await BenchmarkOrchestrator().run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())