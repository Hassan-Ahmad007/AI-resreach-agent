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

import time

import litellm
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
            # max_results MUST be passed as a keyword argument (not positional).
            # Kept small (3) on purpose: every result gets echoed back into the
            # conversation on every following step, so a bigger number here
            # burns through Groq's free-tier tokens-per-minute limit fast.
            results = list(ddgs.text(query, max_results=3))
    except Exception as exc:  # network hiccups, rate limits, etc.
        return f"Search failed for query '{query}': {exc}"

    if not results:
        return f"No results found for '{query}'."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        link = r.get("href", "")
        # Truncate long snippets for the same token-budget reason as above.
        snippet = r.get("body", "")[:220]
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
        # Caps how many tokens the model can generate per call. Keeps each
        # step's usage predictable so we stay under Groq's free-tier
        # tokens-per-minute limit instead of one big reply eating it all.
        max_completion_tokens=1200,
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
        # Caps the number of think/act loop iterations. Without a limit, a
        # stubborn agent can keep searching and re-reading its own growing
        # context, which is the fastest way to blow through a tokens-per-
        # minute limit on a free API tier.
        max_iter=8,
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

    Groq's free tier has a tokens-per-minute limit. If the agent happens to
    hit it mid-run, we don't want the whole app to just crash — we retry a
    couple of times with a short wait, since the limit resets every minute.
    """
    max_attempts = 3
    wait_seconds = 20  # Groq's free-tier TPM budget resets on a per-minute window

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        crew = build_crew(topic=topic, groq_api_key=groq_api_key)
        try:
            result = crew.kickoff()
            return str(result)
        except litellm.RateLimitError as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(wait_seconds)
                wait_seconds *= 2  # back off a little more each retry

    raise RuntimeError(
        "Groq's free-tier rate limit (tokens per minute) was hit several "
        "times in a row. Wait about a minute and try again, or try a more "
        "specific/narrower topic so the agent needs fewer search steps."
    ) from last_error
