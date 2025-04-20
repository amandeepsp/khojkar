import asyncio
import logging
import uuid

import litellm
from aiolimiter import AsyncLimiter
from litellm.exceptions import APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

llm_rate_limiter = AsyncLimiter(max_rate=2)

session_id = "khojkar-session-" + str(uuid.uuid4())


async def acompletion(**kwargs):
    """Wraps litellm.completion with rate limiting."""
    await llm_rate_limiter.acquire()
    logger.debug(
        f"Calling litellm.completion with args: {kwargs.get('model', 'default')}"
    )
    try:
        response = await litellm.acompletion(**kwargs)
        logger.debug("litellm.completion call successful.")
        return response
    except Exception as e:
        logger.error(f"litellm.completion call failed: {e}")

        if isinstance(e, RateLimitError):
            logger.error("Rate limit error, waiting for 60 seconds before retrying...")
            await asyncio.sleep(60)
            return await acompletion(**kwargs)

        if isinstance(e, APIConnectionError):
            error_status = e.status_code
            if error_status == 499:
                logger.error(
                    "API connection error, waiting for 60 seconds before retrying..."
                )
                await asyncio.sleep(60)
                return await acompletion(**kwargs)

        raise e
