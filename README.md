# Job-Fit Copilot

An AI assistant that helps you prepare for interviews by answering questions about your resume and a target job description — honestly.

## What it does

Upload your resume and a job description. Then ask specific questions, like you're quizzing yourself before an interview:

- **"Do I have experience with React?"**
- **"What did I do at my last job?"**
- **"What's the salary range for this role?"**
- **"Does my experience match the JD's requirement of 4-7 years?"**
- **"Am I qualified based on my Node.js background?"**

The assistant answers using only what's actually in your resume and the job description. If something isn't there, it says so directly instead of guessing or making you sound more qualified than you are.

If you ask something the resume and JD genuinely can't answer — like a company's current stock price or recent news — it asks your permission before searching the web, then gives you a real, up-to-date answer.

**What it's for:** rapid-fire, specific questions you'd actually rehearse before an interview — not a one-shot "write my whole prep plan" tool. Ask it one thing at a time and get a straight answer.

## How it works

**Step 1 — Answer from your documents (default path)**
```
Your question → figure out if it needs the resume, the JD, or both
             → find the most relevant parts of those documents
             → generate an answer using only that content
             → double-check the answer is actually grounded before showing it to you
```

**Step 2 — Search the web (only when needed, and only with your OK)**
```
If your documents can't answer the question →
    ask you: "Should I search the web for this?"
    → if you say yes, search and answer
    → if you say no, it tells you honestly that it doesn't know
```

The assistant defaults to your own documents. It only reaches beyond them when it's confident it should, and even then it checks with you first, rather than deciding on its own.

## Tech stack

- **AI models:** Gemini and Mistral, for answering, classifying, and double-checking answers
- **Orchestration:** LangGraph — manages the step-by-step decision-making, including the "ask permission before web search" behavior
- **Search & retrieval:** Supabase (vector search) to find relevant parts of your resume/JD; Tavily for live web search
- **Tool access:** an MCP server exposes the resume/JD search as a tool that other AI apps (like Claude Desktop) can use directly
- **Infra:** deployed on AWS

## Status

Working end to end: document upload, question answering, groundedness checking, web search fallback with approval, and a connected MCP tool. Single user, no login required.
