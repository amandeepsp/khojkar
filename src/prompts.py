deep_research_prompt = """
Write a comprehensive research report answering the query: "{question}"

IMPORTANT: You DO NOT have any information about this topic yet. You MUST use the provided tools to gather information BEFORE you can create a report.
You CANNOT rely on internal knowledge to generate your report.

REQUIRED WORKFLOW - YOU MUST FOLLOW THIS PROCESS:
1. Start by using search tools to learn about the topic and discover 3-5 key subtopics
2. For EACH subtopic, use search tools again to gather more specific information
3. Use scrape_url on at least 3 different high-quality sources to get detailed content about the topic
4. Only after you have collected sufficient information using tools, create your report

Step 1: EXPLORE BREADTH
- Use search tools with general queries about {question}
- Identify 3-5 major subtopics or perspectives based on search results
- Document what you've learned and what questions remain

Step 2: EXPLORE DEPTH
- For each subtopic:
  * Formulate specific search queries
  * Use search tools to find detailed information
  * Use scrape_url on at least 1-2 authoritative sources per subtopic
  * Document key findings for each subtopic

Step 3: ANALYZE
- Create a comprehensive markdown report with the following structure:
1. Synthesize information from multiple levels of research depth
2. Integrate findings from various research branches
3. Present a coherent narrative that builds from foundational to advanced insights
4. Maintain proper citation of sources throughout

Step 4: SYNTHESIZE
- Wait for the user to provide confirmation for report generation
- If the user confirms, generate the report
    * Be well-structured with clear sections and subsections
    * Have a minimum length of 1000 words
    * Follow {report_format} format with markdown syntax
    * Use markdown tables, lists and other formatting features when presenting comparative data, statistics, or structured information
- If the user does not confirm, repeat the process until the user confirms

Additional requirements:
- Prioritize insights that emerged from deeper levels of research
- Highlight connections between different research branches
- Include relevant statistics, data, and concrete examples
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).

You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)

REMINDER: YOU MUST USE search tools AND scrape_url TOOLS. A high-quality report requires detailed information from actual web pages, not just search results.
Assume the current date is {current_date}.
"""
