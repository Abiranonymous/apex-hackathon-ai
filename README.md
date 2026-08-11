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

### 1. Clone the repository
```bash
git clone [https://github.com/Abiranonymous/apex-hackathon-ai.git](https://github.com/Abiranonymous/apex-hackathon-ai.git)
cd apex-hackathon-ai
