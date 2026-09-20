"""
agent.py
--------
This file defines:
  1. A free DuckDuckGo search tool (no API key needed).
  2. A single CrewAI Agent (a "Research Analyst").
  3. A single Task that tells the agent how to research and write the report.
  4. A `run_research()` function that app.py (the Streamlit UI) calls.

Keeping everything in one file makes it easy to read as a beginner.
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

# ---------------------------------------------------------------------------
# WORKAROUND for a bug in the current stable crewai release.
#
# crewai tags certain messages with a "cache_breakpoint" flag, meant for
# providers that support prompt caching (like Anthropic). That flag is
# supposed to get stripped back out before the request is sent to any
# provider that doesn't understand it — but in the currently pinned stable
# release, that cleanup step only exists for native provider adapters
# (Anthropic, Bedrock). When a model is routed through the generic LiteLLM
# path instead (which is what happens for Groq today, see requirements.txt),
# the flag leaks straight into the request body and Groq's API rejects it
# with: "property 'cache_breakpoint' is unsupported".
#
# This fix has already landed in crewai's development branch and will
# eventually ship in a stable release, at which point this block can be
# deleted. Until then, we neutralize it ourselves: `mark_cache_breakpoint`
# is imported freshly (locally) each time crewai needs it, so replacing it
# here — before any Agent/Task/Crew is built — makes it a no-op everywhere.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda message: dict(message)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. THE SEARCH TOOL
# ---------------------------------------------------------------------------
# CrewAI agents can only use the internet if we give them a "tool" for it.
# We build our own tiny tool around the free `ddgs` (DuckDuckGo Search)
# library. No API key is required for this.
#
# The `@tool(...)` decorator turns a normal Python function into something
# a CrewAI agent can call. The docstring below the `def` line is IMPORTANT:
# the agent reads it to decide when to use this tool.
# ---------------------------------------------------------------------------

@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """
    Searches the web using DuckDuckGo and returns the top results
    (title, link, and a short snippet for each). Use this whenever you
    need current information, facts, statistics, or context about a topic.
    Call it several times with different, specific queries to gather
    enough material for a good report.
    """
    try:
        with DDGS() as ddgs:
            # max_results MUST be passed as a keyword argument (not positional)
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:  # network hiccups, rate limits, etc.
        return f"Search failed for query '{query}': {exc}"

    if not results:
        return f"No results found for '{query}'."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        link = r.get("href", "")
        snippet = r.get("body", "")
        formatted.append(f"{i}. {title}\n   Link: {link}\n   {snippet}")

    return "\n\n".join(formatted)


# ---------------------------------------------------------------------------
# 2 & 3. BUILDING THE AGENT, TASK AND CREW
# ---------------------------------------------------------------------------
# We build these fresh every time run_research() is called, because the
# `topic` and the Groq API key change per request (this is a web app, not a
# one-off script).
# ---------------------------------------------------------------------------

def build_crew(topic: str, groq_api_key: str) -> Crew:
    """Creates and returns a single-agent Crew ready to research `topic`."""

    # The LLM that powers our agent's "brain".
    # CrewAI has native support for Groq — the model string format is:
    #   groq/<model-id-on-groq>
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.5,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research the topic '{topic}' thoroughly using web search, "
            "and produce an accurate, well-organized, up-to-date report."
        ),
        backstory=(
            "You are a meticulous research analyst with years of experience "
            "turning messy web search results into clear, trustworthy reports "
            "for a general audience. You always double-check facts and cite "
            "your sources."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        verbose=True,          # prints the agent's thinking to the terminal/logs
        allow_delegation=False,  # single agent, nobody else to delegate to
    )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Follow these steps:\n"
            "1. Use the DuckDuckGo Search tool multiple times with different, "
            "specific queries to gather up-to-date facts, figures, and "
            "different perspectives on the topic.\n"
            "2. Cross-check important claims against more than one source when "
            "possible.\n"
            "3. Write a clear, well-structured report in Markdown.\n\n"
            "The final report MUST include:\n"
            "- A short introduction to the topic\n"
            "- Key findings, organized under clear headings/subheadings\n"
            "- Relevant facts, statistics, or recent developments you found\n"
            "- A brief conclusion / summary\n"
            "- A 'Sources' section at the end listing the links you actually used\n"
        ),
        expected_output=(
            "A well-formatted Markdown report, roughly 500-800 words, with "
            "headings, a conclusion, and a Sources section containing real links "
            "returned by the search tool."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,  # only one task, so this just runs it once
        verbose=True,
    )
    return crew


def run_research(topic: str, groq_api_key: str) -> str:
    """
    Public function used by the Streamlit app.
    Builds a crew for the given topic and API key, runs it, and returns
    the final report text.
    """
    crew = build_crew(topic=topic, groq_api_key=groq_api_key)
    result = crew.kickoff()
    return str(result)
