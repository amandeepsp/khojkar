import logging
import uuid

import litellm
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

llm_rate_limiter = AsyncLimiter(max_rate=1)

session_id = "khojkar-session-" + str(uuid.uuid4())


async def acompletion(**kwargs):
    """Wraps litellm.completion with rate limiting."""
    await llm_rate_limiter.acquire()
    logger.debug(
        f"Calling litellm.completion with args: {kwargs.get('model', 'default')}"
    )
    response = await litellm.acompletion(**kwargs)
    logger.debug("litellm.completion call successful.")
    return response
