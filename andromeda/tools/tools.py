import os
from typing import List, Literal, Optional, Dict, Callable
import asyncio
import threading
from langchain_core.tools import tool
from andromeda.utils.langtils import get_chat_model
from andromeda.config.config import ModelConfig
from andromeda import HumanMessage
from tavily import TavilyClient
import functools
from andromeda.utils.prompts import clean_text_prompt
from andromeda.utils.schemas import CleanText
from andromeda.tools.toolkit import register_tools

def clean_with_AI(text: str, query: str, max_length: int = 10000) -> str:
    """Clean the text using LLM. Useful for cleaning up web scraped text, and narrowing down to the relevant information."""
    # Removed for now. TBD: Do we really need this?
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    model = get_chat_model(
        model_config=ModelConfig(
            name="qwen3:4b",
            provider="ollama",
            temperature=0.6,
            other_args={"num_ctx": 32768},
        )
    )
    if len(text) > 40000:
        text = text[:40000] + "..."  # Truncate to 40k characters
    prompt = clean_text_prompt(text, query)

    response = model.with_structured_output(CleanText, method="json_schema").invoke(
        [
            HumanMessage(prompt),
        ],
    )
    if response.get("relevant_to_target_company") == False:
        return None
    return response["clean_text"]


def _run_coro_blocking(coro):
    """Run async crawl code safely from sync tool calls."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = []
    error = []
    done = threading.Event()

    def runner():
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    done.wait()
    if error:
        raise error[0]
    return result[0] if result else None


# Singleton to track search results across tool invocations
class SearchContextManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchContextManager, cls).__new__(cls)
            cls._instance.search_results = []
            cls._instance.current_search_index = 1
            cls._instance.company_domain = ""
            cls._instance.last_search_results = []
        return cls._instance

    def add_search_result(self, query: str, result_data: Dict):
        """Add a search result to the context manager"""
        context_data = self.check_context(result_data)
        if context_data:
            return context_data
        search_id = self.current_search_index
        self.current_search_index += 1

        search_entry = {"id": search_id, "query": query, "data": result_data}
        self.search_results.append(search_entry)
        return search_entry

    def get_all_context(self) -> List[Dict]:
        """Get all search results as context"""
        return self.search_results

    def check_context(self, result_data: Dict) -> Optional[Dict]:
        """Check if a search result is already in the context"""
        for result in self.search_results:
            if "key" in result_data and "key" in result["data"]:
                if result_data["key"] == result["data"].get("key"):
                    return result
            elif result["data"]["url"] == result_data["url"]:
                return result
        return None

    def clear_context(self):
        """Clear all search results"""
        self.search_results = []
        self.current_search_index = 1

    def set_company_domain(self, domain: str):
        """Set the company domain for future searches"""
        self.company_domain = domain

    def get_company_domain(self) -> Optional[str]:
        """Get the company domain for future searches"""
        return self.company_domain

    def set_last_search_results(self, results: List[int]):
        """Set the last search results"""
        self.last_search_results += results

    def get_last_search_results(self) -> List[int]:
        """Get the last search results"""
        return self.last_search_results

    def clear_last_search_results(self):
        """Clear the last search results"""
        self.last_search_results = []


# Create singleton instance
search_context = SearchContextManager()
try:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except:
    tavily_client = None
    


def search_context_processor(
    func=None,
    *,
    clean_text: bool = True,
    url_field: str = "url",
    min_len: int = 50,
    fetch_cache=True,
) -> Callable:
    """
    Decorator that processes search results, adds them to context manager,
    and formats the output consistently for search tools.

    Can be used with or without parameters:
      @search_context_processor
    or
      @search_context_processor(clean_text=False, url_field='link')
    """

    def decorator(inner_func: Callable) -> Callable:
        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            # Get query from args or kwargs
            query = (
                args[0] if args else kwargs.get("query", "")
            ) + f"<<<{inner_func.__name__}>>>"

            search_results = {
                "results": [
                    data["data"]
                    for data in search_context.get_all_context()
                    if data["query"] == query
                ]
            }

            if not search_results["results"]:
                # Call the original function to get search results
                search_results = inner_func(*args, **kwargs)
            # If the function returns a string directly, return it unchanged
            if isinstance(search_results, str):
                return search_results

            result_str = ""
            search_ids = []

            # Process results and add to context
            for result in search_results.get("results", []):
                if result["content"].lower().startswith("error"):
                    return result["content"]
                context_data = search_context.check_context(result)
                if context_data:
                    search_id = context_data["id"]
                    if fetch_cache:
                        result = context_data["data"]
                    else:
                        result["content"] = result.get("raw_content", "") or result.get(
                            "content", ""
                        )
                else:
                    if clean_text:
                        result["content"] = clean_with_AI(
                            result.get("raw_content", ""), query.split("<<<")[0]
                        ) or result.get("content", "")
                    else:
                        result["content"] = result.get("raw_content", "") or result.get(
                            "content", ""
                        )
                    if len(result.get("content", "")) < min_len:
                        continue
                    added_result = search_context.add_search_result(query, result)
                    search_id = added_result["id"]

                search_ids.append(search_id)
                result_str += f"<start [Search #{search_id}] {result['title']} ({result[url_field]})>\n"
                result_str += f"TITLE: {result['title']}\n"
                if "published_date" in result:
                    result_str += f"Published on {result['published_date']}\n"
                result_str += f"{result['content']}\n<end (Search #{search_id})>\n"
            result_str += "\n\nAlways cite the sources in-text based on Search # as only [Search #n] \
where n is the Search #, for each line of information you write in the report. \
Include a sources/references section as well, which matches the Search #. \
For sources section, use format \"[Search #n] 'Title as given' (url/page_num as given)\" \
without double quotes. This is important for academic integrity and credibility. \
Use only the information from the above results to answer the user query. Do not use any other information. \
If any of the above information is incorrect or not useful, do not assume or make connections, use a different query or tool to get more information\n\n"
            search_context.set_last_search_results(search_ids)
            return result_str

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


@tool
@search_context_processor
def web_search(
    query: str, include_domains: Optional[List[str]] = None,
) -> Dict:
    """
    Use this tool to search the internet for general information. You must NOT use this for news articles.
    query: The query to search for information. Must be specific and detailed.
    include_domains: List of domains to restrict the search to.
    
    This tool always returns results from the last year.
    Verify the sources and use only reliable results.
    """
    if not tavily_client:
        raise ValueError("Tavily client not initialized. You cannot make the tool calls now.")
    search_results = tavily_client.search(
        query=query,
        topic="general",
        search_depth="advanced",
        max_results=3,
        time_range="year",
        include_raw_content=True,
        include_answer=False,
        include_domains=include_domains,
    )

    return search_results

@tool
@search_context_processor
def search_historical(
    query: str, 
    start_date: str, 
    end_date: str, 
    topic: Literal["general", "news"] = "general",
    include_domains: Optional[List[str]] = None,
) -> Dict:
    """
    Use this tool to search the internet or news articles for information.
    query: The query to search for information. Must be specific and detailed.
    start_date (string)
        Returns all results published after the specified start date.
        Must be written in the format: YYYY-MM-DD
        Example: "2025-02-09"
    end_date (string)
        Returns all results published before the specified end date.
        Must be written in the format: YYYY-MM-DD
        Example: "2000-01-28"

    topic: The topic to search for information. Defaults to "general".
    include_domains: List[str] of domains to restrict the search to.

    Verify the sources and use only reliable results.
    """
    if not tavily_client:
        raise ValueError("Tavily client not initialized. You cannot make the tool calls now.")
    search_results = tavily_client.search(
        query=query,
        topic=topic,
        search_depth="advanced",
        max_results=5,
        start_date=start_date,
        end_date=end_date,
        include_raw_content=True,
        include_answer=False,
        include_domains=include_domains,
    )
    return search_results

@tool
@search_context_processor
def news_search(query: str, days: int = 365) -> Dict:
    """
    Use this tool to search internet for recent news articles.
    query: The query to search for news articles. Must be specific and detailed.
    days: Number of days to restrict the search to. Defaults to 365 days.
    Use this tool to get news articles, not general information.
    Verify the sources and use only reliable results.
    """
    query = f"{query}"
    if not tavily_client:
        raise ValueError("Tavily client not initialized. You cannot make the tool calls now.")
    search_results = tavily_client.search(
        query=query,
        topic="news",
        search_depth="advanced",
        max_results=3,
        days=days,
        include_raw_content=True,
        include_answer=False,
    )

    return search_results


@tool
def crawl_url(url: str, max_chars: int = 12000) -> str:
    """
    Extract page content from a known URL.

    Args:
        url: The full URL to crawl.
        max_chars: Maximum number of characters to return from extracted content.
    Returns:
        Extracted page content and basic metadata as plain text.
    """
    target_url = (url or "").strip()
    if not target_url:
        return "Error: 'url' must be a non-empty string."
    if max_chars <= 0:
        return "Error: 'max_chars' must be greater than 0."

    async def _crawl():
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(target_url)

    try:
        result = _run_coro_blocking(_crawl())
    except ModuleNotFoundError:
        return (
            "Error: crawl4ai is not installed. Install it with "
            "`pip install crawl4ai` and run `crawl4ai-setup`."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: Crawl4AI failed for '{target_url}': {exc}"

    if not getattr(result, "success", False):
        status = getattr(result, "status_code", "unknown")
        error_message = getattr(result, "error_message", "Unknown crawl error")
        return (
            f"Error: Crawl failed for '{target_url}' "
            f"(status: {status}). {error_message}"
        )

    title = ""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        title = str(metadata.get("title", "")).strip()

    markdown = getattr(result, "markdown", None)
    content = ""
    if isinstance(markdown, str):
        content = markdown
    elif markdown is not None:
        fit_markdown = getattr(markdown, "fit_markdown", None)
        raw_markdown = getattr(markdown, "raw_markdown", None)
        content = fit_markdown or raw_markdown or ""

    if not content:
        content = (
            getattr(result, "cleaned_html", None)
            or getattr(result, "html", None)
            or ""
        )

    content = content.strip()
    if len(content) > max_chars:
        content = content[:max_chars].rstrip() + "\n...[truncated]"

    header_title = title or target_url
    return (
        f"<start [Crawl Result]>\n"
        f"SOURCE: {header_title} ({target_url})\n"
        f"CONTENT: {content}\n"
        "<end [Crawl Result]>\n"
    )

def get_search_context() -> List[Dict]:
    """Get all search results as context"""
    return search_context.get_all_context()


# ----------------------------------------------------------------------
# Register built-in tools with the global Toolkit registry
# ----------------------------------------------------------------------
# This allows tools to be referenced by name in configuration files, e.g.:
#   tools: [web_search, news_search]
#
# Registration is idempotent and will not override user-registered tools
# with the same name unless explicitly requested.
register_tools([
    web_search, 
    news_search, 
    search_historical,
    crawl_url,
])
