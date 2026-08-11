# Apex Hackathon AI 🚀

An autonomous, multi-agent graph system built with **LangGraph** and **Python**. It is designed to ingest multiple hackathon problem statements, autonomously select the most viable one, and ruthlessly evaluate it to produce winning technical specifications and pitch decks.

## Architecture & Features

This system runs a bounded cyclic graph that evaluates ideas on a strict $0-budget constraint.

*   **🧠 Problem Selector (New):** Ingests an array of hackathon problem statements, evaluates them against a strict rubric, and autonomously selects the single best idea based on $0-budget viability and novelty potential.
*   **🕵️‍♂️ Adversarial Researcher:** Scrapes competitor weaknesses via live web search and identifies true innovation gaps.
*   **🏗️ Shoestring Architect:** Enforces a mandatory $0-budget constraint, utilizing only open-source tools, free-tier infrastructure, and local/quantized models.
*   **⚖️ Red Team Judge:** An adversarial feedback loop that grades designs on Novelty (>=7), Feasibility (>=8), and Scale (>=7). If a design fails, it forces the Architect or Researcher to iterate.
*   **📦 Deliverables Master:** Generates a 6-slide SIH-formatted pitch deck, a repository ASCII tree, and runnable MVP scaffold files.

---

## Setup & Installation

Follow these steps to get the environment ready on your local machine.

### 1. Clone the repository
```bash
git clone [https://github.com/Abiranonymous/apex-hackathon-ai.git](https://github.com/Abiranonymous/apex-hackathon-ai.git)
cd apex-hackathon-ai

```

### 2. Install Dependencies

This project requires specific libraries to run the LangGraph agents and connect to the LLMs. Run this in your terminal:

```bash
pip install langgraph langchain-core pydantic python-dotenv langchain-google-genai ddgs

```

### 3. Configure Your API Key

The AI relies on a language model to power its agents (defaulting to Google Gemini). In the root folder of this project, create a new file named exactly `.env` and add your API key:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_ai_studio_api_key_here

```

---

## How to Run the Agent

Once your dependencies are installed and your `.env` file is ready, execute the Python script from your terminal:

```bash
python scripts/apex_hackathon_ai.py

```

### The Interactive Prompt

The terminal will launch an interactive intake loop.

1. Paste a hackathon problem statement and press **Enter**.
2. Paste as many candidate problem statements as you want to be evaluated.
3. Once you have added all candidates, type **DONE** and press **Enter**.

The AI will then evaluate all candidates, pick the winner, and run it through the multi-agent feedback loop.

---

## Collecting Your Output

Once the system successfully completes its iterations (or halts at the forced-exit limit), it will automatically generate an `apex_output` folder in your directory containing:

* `PITCH_DECK.md`: A fully outlined 6-slide pitch deck (SIH format) ready to be copied into Canva/PowerPoint.
* `README.md`: A tailored readme for your specific generated project.
* **MVP Scaffold Files:** Syntactically valid, runnable boilerplate code matching the approved $0-budget tech stack so you can start coding immediately.

```

```
