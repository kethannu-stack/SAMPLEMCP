

# tools.py
# This file defines all the "tools" that the AI can call.
# Think of these as capabilities the LLM can choose to use.
# This is the heart of the MCP (Model Context Protocol) pattern.

import httpx
import os

# ─────────────────────────────────────────────
# TOOL 1: Web Search via Tavily API
# ─────────────────────────────────────────────
async def web_search(query: str) -> dict:
    """
    Searches the web using Tavily API.
    Returns a list of results with title, content, and URL.
    """
    api_key = os.getenv("TAVILY_API_KEY", "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")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": 5,
                    "include_answer": True,
                },
                timeout=15.0
            )
            data = response.json()

            # Extract clean results from Tavily response
            results = []
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:600],  # Limit content size
                    "url": r.get("url", ""),
                    "score": r.get("score", 0),
                })

            return {
                "source": "web_search",
                "query": query,
                "answer": data.get("answer", ""),
                "results": results,
                "reliability": "medium",  # Web is medium reliability
            }

        except Exception as e:
            return {
                "source": "web_search",
                "query": query,
                "error": str(e),
                "results": [],
                "reliability": "low",
            }


# ─────────────────────────────────────────────
# TOOL 2: Wikipedia Summary
# ─────────────────────────────────────────────
async def wikipedia_search(topic: str) -> dict:
    """
    Fetches a summary from Wikipedia using their public REST API.
    No API key needed. Very reliable for factual definitions.
    """
    # Wikipedia REST API — completely free and reliable
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)

            if response.status_code == 200:
                data = response.json()
                return {
                    "source": "wikipedia",
                    "title": data.get("title", topic),
                    "summary": data.get("extract", "No summary available."),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "reliability": "high",  # Wikipedia = high reliability
                }
            else:
                # Try a search if direct page not found
                search_url = f"https://en.wikipedia.org/w/api.php"
                search_resp = await client.get(search_url, params={
                    "action": "opensearch",
                    "search": topic,
                    "limit": 1,
                    "format": "json"
                }, timeout=10.0)
                search_data = search_resp.json()
                
                # If we found a matching page, fetch it
                if search_data[1]:
                    best_match = search_data[1][0]
                    return await wikipedia_search(best_match)
                
                return {
                    "source": "wikipedia",
                    "title": topic,
                    "summary": "No Wikipedia article found for this topic.",
                    "url": "",
                    "reliability": "low",
                }

        except Exception as e:
            return {
                "source": "wikipedia",
                "title": topic,
                "error": str(e),
                "summary": "",
                "reliability": "low",
            }


# ─────────────────────────────────────────────
# TOOL REGISTRY — MCP Pattern
# ─────────────────────────────────────────────
# This is the KEY MCP concept: tools are registered in a standard format.
# The LLM reads these schemas and decides which tool to call.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for recent, broad, or practical information about a topic. "
                "Use this when you need current examples, applications, or explanations "
                "not covered by Wikipedia."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query (e.g., 'how does photosynthesis work explained simply')"
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Fetch a reliable summary from Wikipedia about a topic. "
                "Use this first for definitions, historical facts, scientific concepts, "
                "or anything requiring factual accuracy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The Wikipedia page title or close approximation (e.g., 'Photosynthesis')"
                    }
                },
                "required": ["topic"],
            },
        },
    },
]


# ─────────────────────────────────────────────
# TOOL DISPATCHER
# Calls the right function based on the tool name the LLM chose.
# ─────────────────────────────────────────────
async def dispatch_tool(tool_name: str, tool_args: dict) -> dict:
    """
    Given the tool name and arguments (chosen by the LLM),
    call the actual Python function and return the result.
    """
    if tool_name == "web_search":
        return await web_search(tool_args["query"])
    elif tool_name == "wikipedia_search":
        return await wikipedia_search(tool_args["topic"])
    else:
        return {"error": f"Unknown tool: {tool_name}"}
