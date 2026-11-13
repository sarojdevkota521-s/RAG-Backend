import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import httpx
from app.core.config import get_settings
from app.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)
settings = get_settings()

BOOKING_TOOL_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "book_interview",
        "description": "Book or schedule an interview when the candidate's name, email, date, and time are all provided.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full name of the candidate"
                },
                "email": {
                    "type": "string",
                    "description": "Email address of the candidate"
                },
                "date": {
                    "type": "string",
                    "description": "Date for the interview (e.g. '2026-11-20' or 'next Tuesday')"
                },
                "time": {
                    "type": "string",
                    "description": "Time for the interview (e.g. '14:00' or '2:00 PM')"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes or position applied for"
                }
            },
            "required": ["name", "email", "date", "time"]
        }
    }
}


class LLMService:
    """
    Modular LLM service using httpx to interface with OpenAI-compatible endpoints.
    Direct HTTP integration avoids native DLL/jiter compatibility issues on Windows.
    Supports OpenAI, OpenRouter, Groq, xAI, vLLM, Ollama.
    Handles Tool Calling for interview bookings and document RAG generation.
    """

    def __init__(self):
        api_key, base_url, model = settings.effective_llm_credentials
        self.api_key = api_key
        # Ensure base_url ends cleanly without duplicate slashes
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_configured(self) -> bool:
        """Check if a valid LLM API key is present."""
        return bool(self.api_key and self.api_key != "your_api_key")

    async def generate_response(
        self,
        user_query: str,
        conversation_history: List[ChatMessage],
        context_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """
        Generates conversational response using LLM with tool calling support.
        Returns:
            Tuple of (assistant_text, booking_data_dict_or_none, action_type)
            action_type is one of: 'interview_booking', 'interview_clarification', 'rag_query', 'general_chat'
        """
        if not self.is_configured():
            logger.info("LLM API key not configured or placeholder. Using intelligent local engine.")
            return self._fallback_pipeline(user_query, conversation_history, context_chunks)

        messages = self._build_prompt_messages(
            user_query=user_query,
            history=conversation_history,
            context_chunks=context_chunks,
            system_instruction=system_instruction
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [BOOKING_TOOL_DEFINITION],
            "tool_choice": "auto",
            "temperature": 0.2
        }

        endpoint = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(endpoint, json=payload, headers=headers)

            if res.status_code != 200:
                logger.warning("LLM API returned status %s: %s. Falling back to local engine.", res.status_code, res.text)
                return self._fallback_pipeline(user_query, conversation_history, context_chunks)

            data = res.json()
            choice = data["choices"][0]["message"]

            # Check if LLM invoked the book_interview tool
            tool_calls = choice.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    func = tool_call.get("function", {})
                    if func.get("name") == "book_interview":
                        try:
                            raw_args = func.get("arguments", "{}")
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            confirm_text = (
                                f"I have successfully scheduled your interview for {args.get('date')} at {args.get('time')}. "
                                f"A confirmation email will be sent to {args.get('email')}."
                            )
                            return confirm_text, args, "interview_booking"
                        except Exception as parse_err:
                            logger.error("Failed to parse tool call arguments: %s", parse_err)

            content = choice.get("content") or ""

            # Classify action type based on content & context
            action_type = "rag_query" if context_chunks else "general_chat"
            booking_indicators = ["interview", "schedule", "book", "date", "time", "email", "name"]
            if any(term in content.lower() for term in booking_indicators) and any(
                term in user_query.lower() for term in ["interview", "schedule", "book", "slot"]
            ):
                action_type = "interview_clarification"

            return content, None, action_type

        except Exception as e:
            logger.error("LLM API call failed: %s. Falling back to local RAG engine.", e)
            return self._fallback_pipeline(user_query, conversation_history, context_chunks)

    def _build_prompt_messages(
        self,
        user_query: str,
        history: List[ChatMessage],
        context_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Constructs prompt messages for chat completion with context & tool guidelines."""
        context_text = ""
        if context_chunks:
            context_text = "\n\n---\n\n".join(
                f"[Source: {c['payload'].get('source_filename', 'Doc')} | Page: {c['payload'].get('page_number', 'N/A')}]\n"
                f"{c['payload'].get('text', '')}"
                for c in context_chunks
            )

        base_system_prompt = (
            "You are an intelligent AI assistant capable of answering questions from provided documents "
            "and booking candidate interviews.\n\n"
            "GUIDELINES FOR INTERVIEW BOOKING:\n"
            "- You can book interviews. To complete a booking, you MUST gather all 4 required fields:\n"
            "  1. Candidate's Full Name\n"
            "  2. Email Address\n"
            "  3. Date (e.g., 2026-11-20 or specific date)\n"
            "  4. Time (e.g., 14:00 or 2:00 PM)\n"
            "- Check previous messages in the conversation to see if any fields were already provided.\n"
            "- If any required field is missing, ask the user politely for the missing field(s).\n"
            "- Once all 4 fields are present, invoke the 'book_interview' function call.\n\n"
            "GUIDELINES FOR DOCUMENT RAG:\n"
            "- Answer questions truthfully based strictly on the provided Context below.\n"
            "- If the context does not contain enough information to answer, state clearly that the uploaded documents "
            "do not contain that information.\n"
            "- Do not hallucinate facts."
        )

        if context_text:
            base_system_prompt += f"\n\nCONTEXT FROM RETRIEVED DOCUMENTS:\n{context_text}"

        if system_instruction:
            base_system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{system_instruction}"

        messages: List[Dict[str, Any]] = [{"role": "system", "content": base_system_prompt}]

        for msg in history[-8:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_query})
        return messages

    def _fallback_pipeline(
        self,
        user_query: str,
        history: List[ChatMessage],
        context_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """
        Deterministic, offline-capable fallback pipeline.
        Extracts interview booking details across multi-turn history using regex patterns,
        or summarizes retrieved context for RAG queries.
        """
        combined_text = " ".join([m.content for m in history] + [user_query])

        booking_keywords = ["interview", "book", "schedule", "slot", "appointment", "meeting"]
        has_booking_intent = any(kw in combined_text.lower() for kw in booking_keywords)

        if has_booking_intent:
            extracted = self._extract_booking_details(combined_text, user_query)
            missing = []
            if not extracted.get("name"):
                missing.append("full name")
            if not extracted.get("email"):
                missing.append("email address")
            if not extracted.get("date"):
                missing.append("interview date")
            if not extracted.get("time"):
                missing.append("interview time")

            if not missing:
                confirm = (
                    f"I have successfully scheduled an interview for {extracted['name']} "
                    f"on {extracted['date']} at {extracted['time']}. "
                    f"A confirmation email has been sent to {extracted['email']}."
                )
                return confirm, extracted, "interview_booking"
            else:
                missing_str = ", ".join(missing)
                prompt_text = (
                    f"I would be glad to schedule your interview! To proceed, please provide the following missing detail(s): {missing_str}."
                )
                return prompt_text, None, "interview_clarification"

        if context_chunks:
            snippets = [c['payload'].get('text', '') for c in context_chunks if c['payload'].get('text')]
            if snippets:
                doc_name = context_chunks[0]['payload'].get('source_filename', 'uploaded document')
                ans = f"Based on {doc_name}:\n\n" + "\n\n".join(snippets[:2])
                return ans, None, "rag_query"

        return (
            "I am your Conversational RAG Assistant. You can upload documents to ask questions, "
            "or tell me if you would like to book an interview!",
            None,
            "general_chat"
        )

    def _extract_booking_details(self, full_history_text: str, latest_query: str) -> Dict[str, Any]:
        """Extract name, email, date, and time using regex patterns across conversation history."""
        details: Dict[str, Any] = {}

        # 1. Email extraction
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_history_text)
        if email_match:
            details["email"] = email_match.group(0).lower()

        # 2. Date extraction
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
            r'\b(tomorrow|today)\b',
            r'\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4})?)\b'
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, full_history_text, re.IGNORECASE)
            if date_match:
                details["date"] = date_match.group(0).strip()
                break

        # 3. Time extraction
        time_pattern = r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}:\d{2})\b'
        time_matches = re.findall(time_pattern, full_history_text)
        if time_matches:
            details["time"] = time_matches[-1].strip()

        # 4. Name extraction - precise matching stopping at connective words
        # Try capitalized full name first (e.g. "John Doe")
        cap_match = re.search(r'(?:my name is|i am|name is|name:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', full_history_text)
        if cap_match:
            candidate = cap_match.group(1).strip()
            if candidate.lower() not in ["an interview", "a meeting"]:
                details["name"] = candidate

        if "name" not in details:
            # Fallback for case-insensitive match with boundary lookahead
            name_match = re.search(r'(?:my name is|i am|name is|name:)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)+?)(?=\s+(?:and|my|email|on|at|with|\.|\,)|$)', full_history_text, re.IGNORECASE)
            if name_match:
                candidate = name_match.group(1).strip()
                candidate = re.sub(r'\s+(?:and|my|email|the|is)$', '', candidate, flags=re.IGNORECASE)
                if candidate.lower() not in ["an interview", "a meeting"]:
                    details["name"] = candidate

        if "name" not in details:
            book_for_match = re.search(r'(?:interview for|booking for|schedule for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', full_history_text)
            if book_for_match:
                candidate = book_for_match.group(1).strip()
                if candidate.lower() not in ["an interview", "a meeting"]:
                    details["name"] = candidate

        if "name" not in details:
            candidate_names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', latest_query)
            filtered = [
                n for n in candidate_names
                if not any(day in n for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                and not any(m in n for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
            ]
            if filtered:
                details["name"] = filtered[0]

        return details
