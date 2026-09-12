# RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for your own documents, built with
[LangChain](https://www.langchain.com/), [Chroma](https://www.trychroma.com/) as the vector
store, and Google's Gemini API for embeddings and chat completion.

You feed it markdown documents, it splits and embeds them into a local vector database, and then
you can ask natural-language questions that get answered using only the retrieved context —
along with the source document(s) used.

## How it works

1. **`create_database.py`** loads every `.md` file in `data/books/`, splits them into overlapping
   chunks, embeds each chunk with Gemini embeddings, and persists everything into a local Chroma
   vector database (`chroma/`).
2. **`query_data.py`** embeds your question, retrieves the most relevant chunks from Chroma,
   stuffs them into a prompt template as context, and asks a Gemini chat model to answer based
   only on that context. It prints the answer plus which source file(s) it came from.
3. **`test_rag.py`** contains automated evaluation tests: it asks known questions about the sample
   documents and uses an LLM to judge whether the RAG pipeline's answer matches the expected
   answer.
4. **`compare_embeddings.py`** is a small standalone script to inspect embedding vectors and
   compare the similarity between two words/phrases.

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Gemini API key**

   Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey), then open the
   `.env` file in the project root and set:

   ```
   GOOGLE_API_KEY=AIza...
   ```

3. **Build the vector database**

   ```bash
   python create_database.py
   ```

   This reads the markdown files in `data/books/` and creates the `chroma/` folder. Re-run this
   any time you add, remove, or edit documents in `data/books/`.

4. **Ask a question**

   ```bash
   python query_data.py "What are the five constant factors in The Art of War?"
   python query_data.py "How does castling work in chess?"
   ```

## Using your own documents

Drop your own `.md` files into `data/books/` (replacing or alongside the sample ones) and re-run
`create_database.py`. If you need to load other formats (PDF, .txt, .docx, etc.), swap out the
`DirectoryLoader` in `create_database.py` for the appropriate LangChain document loader.

## Sample data

Two original sample documents are included so the pipeline works out of the box:

- `data/books/art_of_war.md` — a condensed, paraphrased edition of Sun Tzu's *The Art of War*.
- `data/books/chess_rules.md` — an original written reference for the standard rules of chess.

## Running the evaluation tests

```bash
pytest test_rag.py
```

Each test asks a question, runs it through the RAG pipeline, and then uses a chat model to judge
whether the actual answer matches the expected answer (true/false). This is a common pattern for
evaluating RAG/LLM output where exact string matching isn't reliable.

## Project structure

```
rag-document-chatbot/
├── data/
│   └── books/
│       ├── art_of_war.md
│       └── chess_rules.md
├── chroma/                 # generated vector DB (git-ignored)
├── create_database.py      # build/rebuild the vector store
├── query_data.py           # ask questions against the vector store
├── compare_embeddings.py   # inspect/compare embedding vectors
├── test_rag.py             # LLM-judged evaluation tests
├── requirements.txt
└── .env                    # holds GOOGLE_API_KEY (not committed)
```

## Notes

- The vector database (`chroma/`) and `.env` are git-ignored — never commit your API key.
- `create_database.py` wipes and rebuilds `chroma/` from scratch every time it runs.
- If `query_data.py` reports "Unable to find matching results," the top match's relevance score
  was below the `0.5` threshold — try rephrasing the question or lowering the threshold in
  `query_data.py`.
- Model names: `create_database.py`/`query_data.py` use `models/gemini-embedding-001` for
  embeddings and `gemini-3.6-flash` for chat. If Google retires/renames these later, update the
  model strings — an error message from the API will usually name the current replacement.
