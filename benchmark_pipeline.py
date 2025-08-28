"""Benchmark pipeline runner."""

import asyncio
import logging
from src.pipeline import BenchmarkOrchestrator

# Configure logging once at startup
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


async def main():
    await BenchmarkOrchestrator().run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())