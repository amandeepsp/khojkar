import logging
import uuid

import litellm
from langfuse.decorators import observe

from core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Initialize the global rate limiter
llm_rate_limiter = RateLimiter(10)

session_id = "khojkar-session-" + str(uuid.uuid4())


@observe()
async def acompletion(**kwargs):
    """Wraps litellm.completion with rate limiting."""
    await llm_rate_limiter.acquire()
    try:
        logger.debug(
            f"Calling litellm.completion with args: {kwargs.get('model', 'default')}"
        )
        kwargs["metadata"] = {"session_id": session_id}
        response = await litellm.acompletion(**kwargs)
        logger.debug("litellm.completion call successful.")
        return response
    except Exception as e:
        logger.error(f"litellm.completion call failed: {e}", exc_info=True)
        raise  # Re-raise the exception after logging
