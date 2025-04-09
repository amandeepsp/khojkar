import json
import litellm
import utils
import logging

logger = logging.getLogger(__name__)

DEFAULT_REPORT_TEMPLATE = """
    Use this structure to create a report on the user-provided topic:

    1. Introduction (no research needed)
    - Brief overview of the topic area

    2..n
    - Each section should focus on a sub-topic of the user-provided topic
    
    (n+1). Conclusion
        - Aim for 1 structural element (either a list of table) that distills the main body sections 
        - Provide a concise summary of the report
"""

class QueryPlanner:
    def __init__(
            self, 
            model: str, 
            topic: str, 
            report_template: str = DEFAULT_REPORT_TEMPLATE, 
            number_of_queries: int = 5
        ) -> None:

        self.model = model
        self.report_template = report_template
        self.topic = topic
        self.number_of_queries = number_of_queries
        logger.info(f"Initialized QueryPlanner for topic: {topic} using model: {model}")

    def _generate_queries_prompt(self) -> str:
        return f"""
        You are performing research for a report. 

        <Report topic>
        {self.topic}
        </Report topic>

        <Report organization>
        {self.report_template}
        </Report organization>

        <Task>
        Your goal is to generate {self.number_of_queries} web search queries that will help gather information for planning the report sections. 

        The queries should:

        1. Be related to the Report topic
        2. Help satisfy the requirements specified in the report organization

        Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
        </Task>

        <Format>
        Output should be in the following JSON format. DO NOT INCLUDE BACKTICKS IN THE RESPONSE
        [
            "search query 1",
            "search query 2",
            "search query 3"
        ]
        </Format>
        """
    
    def generate_queries(self) -> list[str]:
        logger.info(f"Generating {self.number_of_queries} search queries for topic: {self.topic}")
        prompt = self._generate_queries_prompt()
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        queries = json.loads(response.choices[0].message.content)
        logger.info(f"Generated {len(queries)} search queries")
        logger.debug(f"Generated queries: {queries}")
        return queries
        

class ScrapePlanner:
    def __init__(
            self,
            model: str,
            topic: str,
            search_results: dict[str, str],
        ) -> None:

        self.model = model
        self.search_results = search_results
        self.topic = topic
        logger.info(f"Initialized ScrapePlanner for topic: {topic} using model: {model}")
        logger.debug(f"Received {len(search_results)} search result sets")
    
    def _generate_analysis_prompt(self):
        """Generate prompt for LLM to analyze search results.
        
        Args:
            search_results: Dict mapping queries to their search results
            
        Returns:
            Prompt string for LLM
        """
        prompt = f"""
        You are performing research for a report.

        <Report topic>
        {self.topic}
        </Report topic>

        I've performed searches for the following queries and received these results:        
        """

        for query, result in self.search_results.items():
            prompt += f"Query: {query}\n"
            prompt += f"{result}\n\n"
        
        prompt += """
        <Task>
        Analyze these search results and identify the most promising sources for researching the topic.
        Select sources that:
        1. Are most relevant to the report topic
        2. Appear to be credible and informative
        3. Collectively cover different aspects of the topic
        4. Provide comprehensive information
        
        For each selected source, explain briefly why it's valuable for the research.
        </Task>
        
        <Format>
        Output a JSON list of the most promising sources in this format:
        [
            {
                "url": "source URL",
                "title": "source title",
                "reason": "brief explanation of why this source is valuable"
            },
            ...
        ]
        </Format>
        """
        return prompt
    
    def analyze_and_select_sources(self):
        """Search for all queries and then analyze results to pick the best sources.
        
        Returns:
            List of selected sources with explanations
        """
        logger.info(f"Analyzing search results for topic: {self.topic}")
        
        prompt = self._generate_analysis_prompt()
        
        # Ask LLM to analyze and select the most promising sources
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        cleaned_response = utils.extract_json_block(response.choices[0].message.content)
        # Parse the response
        selected_sources = json.loads(cleaned_response)
        logger.info(f"Selected {len(selected_sources)} sources for research")
        logger.debug(f"Selected sources: {selected_sources}")
        return selected_sources
