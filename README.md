
# XML Driven LLM-RAG Algorithm

## Introduction

The LLM-RAG (Large Language Model with Retrieval-Augmented Generation) enhances conversational AI through advanced memory management and data retrieval. It integrates PersistentStore, HazyMemory, and a Conversational Memory Buffer for efficient, context-aware interactions.

## Components

### PersistentStore
- Stores entity relationships and states in an XML format for graph-like compression and flexible data representation.
- Optimizes data compression and querying, ideal for real-time interactions.

### HazyMemory
- A vector database providing broader context from past interactions, improving response relevance and coherence.

### Conversational Memory Buffer
- Maintains the 50 most recent dialogues for continuous conversation flow.

## Setup

1. **Prepare API Key:** Place your OpenAI API key in `secrets.txt` within the project's root directory.
2. **Install Dependencies:** Run `pip install -r requirements.txt`.
3. **Start the Bot:** Execute `python3 main.py` or `python main.py` if Python 3 is not explicitly required.

## Application

Designed for gaming and simulation, this architecture allows for dynamic entity interaction and state management, enabling immersive experiences.

## Running the Code

Ensure your OpenAI API key is set in `secrets.txt`. Install required libraries with `pip install -r requirements.txt`, then start the chat bot using `python3 main.py`.

## Conclusion

The LLM-RAG architecture sets new standards in conversational AI, leveraging innovative approaches in memory management and data storage for responsive and context-aware interactions.

