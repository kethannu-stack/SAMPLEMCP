
# agent.py
# This is the brain of the system.
# It implements 3 logical agents using role prompts + Groq LLM.
# Function calling lets the LLM decide WHICH tools to use — not us.

import json
import os
from groq import AsyncGroq
from tools import TOOL_SCHEMAS, dispatch_tool

# Initialize Groq client (Groq is fast — perfect for hackathons)
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "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"))
MODEL = "llama-3.1-8b-instant"


# ══════════════════════════════════════════════════════════════
# AGENT 1: RESEARCH AGENT
# Role: Decides which tools to call and gathers raw information.
# This agent has access to tools and uses function calling.
# ══════════════════════════════════════════════════════════════
async def research_agent(topic: str, mode: str, reasoning_steps: list) -> dict:
    """
    Research Agent: Autonomously searches for information using tools.
    
    MCP Pattern: The agent registers tools, sends them to the LLM,
    and the LLM decides when/which tool to call via function calling.
    """
    reasoning_steps.append("🔍 Research Agent activated — planning tool usage...")

    # System prompt tells the agent its role and constraints
    system_prompt = f"""You are a Research Agent for a student study assistant.
Your job is to gather comprehensive information about the given topic.

You have access to two tools:
- wikipedia_search: for factual, reliable definitions and overviews
- web_search: for examples, applications, and detailed explanations

Strategy:
1. ALWAYS start with Wikipedia for the core definition/overview
2. Then use web_search for examples, recent developments, or exam-focused content
3. You may call tools multiple times if needed
4. Gather enough to cover: core concepts, how it works, real examples, and importance

Topic: {topic}
Study Mode: {mode} ({'brief overview' if mode == 'quick' else 'comprehensive deep dive'})
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research the topic: '{topic}'. Gather all relevant information for a student exam preparation guide."}
    ]

    # Collected raw data from all tool calls
    raw_data = []

    # ─────────────────────────────────────────────
    # AGENTIC LOOP: Keep calling tools until the LLM says it's done
    # This is the core of MCP-style function calling
    # ─────────────────────────────────────────────
    max_iterations = 6  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Send messages + tool schemas to LLM
        # The LLM will either: (a) call a tool, or (b) return final text
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,          # ← Register tools with LLM
            tool_choice="auto",          # ← LLM decides when to use tools
            max_tokens=1000,
        )

        assistant_message = response.choices[0].message

        # Check if LLM wants to call a tool
        if assistant_message.tool_calls:
            # Add assistant's tool-calling message to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute each tool the LLM requested
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Log which tool is being called (visible to user in UI)
                if tool_name == "wikipedia_search":
                    reasoning_steps.append(f"📖 Fetching Wikipedia: '{tool_args.get('topic', '')}'")
                elif tool_name == "web_search":
                    reasoning_steps.append(f"🌐 Web searching: '{tool_args.get('query', '')}'")

                # Actually call the tool function
                tool_result = await dispatch_tool(tool_name, tool_args)
                raw_data.append(tool_result)

                # Feed tool result back into message history
                # This is "context passing" — each agent sees all prior context
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })

        else:
            # LLM finished — no more tool calls needed
            reasoning_steps.append("✅ Research Agent — data collection complete")
            break

    return {
        "raw_data": raw_data,
        "messages": messages,  # Full context for next agent
    }


# ══════════════════════════════════════════════════════════════
# AGENT 2: SYNTHESIS AGENT
# Role: Takes raw data and converts it into structured study notes.
# ══════════════════════════════════════════════════════════════
async def synthesis_agent(topic: str, mode: str, raw_data: list, reasoning_steps: list) -> str:
    """
    Synthesis Agent: Converts messy raw research into clean study notes.
    No tool calling here — pure language generation.
    """
    reasoning_steps.append("🧠 Synthesis Agent — converting research into study notes...")

    # Combine all raw data into a readable summary for the LLM
    data_summary = json.dumps(raw_data, indent=2)[:4000]  # Limit token size

    if mode == "quick":
        format_instruction = """
Create CONCISE study notes with:
- A 2-3 sentence overview
- 5 bullet points covering key facts
- 3 key terms with one-line definitions
Keep it SHORT — this is Quick Revision Mode.
"""
    else:
        format_instruction = """
Create COMPREHENSIVE study notes in Markdown with:
## Overview
(Detailed explanation, 3-4 paragraphs)

## Core Concepts
(Each concept with explanation)

## How It Works
(Step-by-step or mechanism)

## Real-World Applications
(Practical examples)

## Key Formulas / Facts
(If applicable)

Be thorough — this is Deep Dive Mode.
"""

    system_prompt = f"""You are a Synthesis Agent. 
Your job is to transform raw research data into well-structured, student-friendly study notes.
Write in clear, simple language that a first-year engineering student can understand.
{format_instruction}
"""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nRaw Research Data:\n{data_summary}\n\nGenerate structured study notes now."}
        ],
        max_tokens=2000,
    )

    notes = response.choices[0].message.content
    reasoning_steps.append("📝 Synthesis Agent — study notes generated")
    return notes


# ══════════════════════════════════════════════════════════════
# AGENT 3: EVALUATION AGENT
# Role: Reviews the notes and adds exam features + confidence score.
# ══════════════════════════════════════════════════════════════
async def evaluation_agent(topic: str, study_notes: str, raw_data: list, reasoning_steps: list) -> dict:
    """
    Evaluation Agent: Reviews the study notes and generates:
    - Key concepts list
    - Exam questions
    - Confidence score (0-100)
    - Important topics to focus on
    - Improvement suggestions
    - Source reliability ranking
    """
    reasoning_steps.append("📊 Evaluation Agent — scoring confidence and generating exam questions...")

    # Calculate source reliability from raw_data
    sources = []
    reliability_scores = {"high": 3, "medium": 2, "low": 1}
    total_reliability = 0

    for item in raw_data:
        source_entry = {
            "source": item.get("source", "unknown"),
            "reliability": item.get("reliability", "low"),
            "url": item.get("url", item.get("results", [{}])[0].get("url", "") if item.get("results") else ""),
        }
        sources.append(source_entry)
        total_reliability += reliability_scores.get(item.get("reliability", "low"), 1)

    # Normalized confidence score based on source reliability + notes length
    source_score = min(total_reliability * 15, 60)  # Max 60 from sources
    notes_score = min(len(study_notes) / 50, 40)    # Max 40 from notes richness
    confidence_score = int((source_score * 0.7) + (notes_score * 0.3))
    confidence_score = max(30, min(confidence_score, 95))  # Clamp between 30-95

    # Ask the LLM to extract structured evaluation data
    system_prompt = """You are an Evaluation Agent — an expert exam coach.
Analyze the provided study notes and return ONLY valid JSON (no extra text).

Return this exact JSON structure:
{
  "key_concepts": ["concept1", "concept2", ...],
  "exam_questions": ["Question 1?", "Question 2?", ...],
  "important_topics": ["topic1", "topic2", ...],
  "weak_sections": ["section that needs more detail", ...],
  "improvement_suggestions": ["suggestion1", ...]
}

Rules:
- key_concepts: 5-8 core ideas
- exam_questions: 5 likely exam questions
- important_topics: 3-5 most testable topics
- weak_sections: areas where notes seem thin or vague
- improvement_suggestions: how to make the notes better
"""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nStudy Notes:\n{study_notes}"}
        ],
        max_tokens=800,
        temperature=0.3,  # Lower temperature for consistent JSON output
    )

    raw_eval = response.choices[0].message.content

    # Safely parse the JSON response
    try:
        # Strip any markdown code fences if present
        clean = raw_eval.strip().replace("```json", "").replace("```", "").strip()
        eval_data = json.loads(clean)
    except Exception:
        # Fallback if JSON parsing fails
        eval_data = {
            "key_concepts": ["Core definition", "Main mechanism", "Applications"],
            "exam_questions": [f"What is {topic}?", f"Explain the mechanism of {topic}.", f"Give 3 real-world applications of {topic}."],
            "important_topics": [topic],
            "weak_sections": [],
            "improvement_suggestions": ["Add more real-world examples"],
        }

    reasoning_steps.append("✅ Evaluation complete — confidence score calculated")

    return {
        "key_concepts": eval_data.get("key_concepts", []),
        "exam_questions": eval_data.get("exam_questions", []),
        "important_topics": eval_data.get("important_topics", []),
        "weak_sections": eval_data.get("weak_sections", []),
        "improvement_suggestions": eval_data.get("improvement_suggestions", []),
        "confidence_score": confidence_score,
        "sources": sources,
    }


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR: Runs all 3 agents in sequence
# This is what main.py calls — the full pipeline.
# ══════════════════════════════════════════════════════════════
async def run_study_pipeline(topic: str, mode: str = "deep") -> dict:
    """
    Main orchestrator. Runs agents in order:
    1. Research Agent → gathers data
    2. Synthesis Agent → creates notes
    3. Evaluation Agent → scores and enriches
    
    Context passes forward between agents (MCP pattern).
    """
    reasoning_steps = []
    reasoning_steps.append(f"🚀 Starting study pipeline for: '{topic}' [{mode} mode]")

    # ── Agent 1: Research ──
    research_result = await research_agent(topic, mode, reasoning_steps)

    # ── Agent 2: Synthesis ──
    study_notes = await synthesis_agent(
        topic, mode, research_result["raw_data"], reasoning_steps
    )

    # ── Agent 3: Evaluation ──
    evaluation = await evaluation_agent(
        topic, study_notes, research_result["raw_data"], reasoning_steps
    )

    reasoning_steps.append("🎓 Pipeline complete — study material ready!")

    # ── Final Structured Output ──
    return {
        "reasoning_steps": reasoning_steps,
        "study_notes": study_notes,
        "key_concepts": evaluation["key_concepts"],
        "exam_questions": evaluation["exam_questions"],
        "confidence_score": evaluation["confidence_score"],
        "important_topics": evaluation["important_topics"],
        "weak_sections": evaluation["weak_sections"],
        "improvement_suggestions": evaluation["improvement_suggestions"],
        "sources": evaluation["sources"],
    }
