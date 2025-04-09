import litellm
import logging
from agents.planner import DEFAULT_REPORT_TEMPLATE

logger = logging.getLogger(__name__)

class Reporter:
    """Report the results of the research"""

    def __init__(
        self,
        model: str,
        topic: str,
        scraped_content: list[str],
        report_template: str = DEFAULT_REPORT_TEMPLATE,
    ):
        self.model = model
        self.topic = topic
        self.report_template = report_template
        self.scraped_content = scraped_content
        logger.info(f"Initialized Reporter for topic: {topic} using model: {model}")
        logger.debug(f"Received {len(scraped_content)} scraped content items")

    def _generate_report_prompt(self) -> str:
        """Generate the prompt for the report"""
        logger.debug("Generating report prompt")
        return f"""
        You are a reasearcher
        You are given a topic and a list of scraped content.
        You need to report on the topic based on the scraped content.
        The report should be in a structured format.

        <Topic>
        {self.topic}
        </Topic>

        <Report Template>
        {self.report_template}
        </Report Template>

        <Scraped Content>
        {'\n\n'.join([content for content in self.scraped_content if content])}
        </Scraped Content>
        """.strip()
    
    def generate_report(self) -> str:
        """Generate the report"""
        logger.info(f"Generating report for topic: {self.topic}")
        prompt = self._generate_report_prompt()
        
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        
        report_content = response["choices"][0]["message"]["content"]
        report_length = len(report_content)
        logger.info(f"Generated report with length: {report_length} characters")
        return report_content
