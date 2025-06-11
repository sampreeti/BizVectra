🗞️ README.md
---------------------------------------------------------------------

# 📈 BizVectra: AI-Powered Business Insights Engine

**BizVectra** is an advanced AI-driven analytics assistant that leverages Large Language Models (LLMs), vector databases, and interactive dashboards to generate business insights from structured sales data. It fuses natural language processing, real-time visualization, and memory-aware contextual reasoning, providing analysts and decision-makers with an intelligent, conversational analytics layer.

Designed using **LangChain**, **FAISS**, **OpenAI**, and **Streamlit**, BizVectra bridges the gap between data and decision-making.

---


## 💡Core Capabilities

- **Data-Aware RAG Pipeline**: Embeds structured business summaries using OpenAI embeddings and retrieves context with FAISS-based vector search.
- **Conversational Memory**: Maintains multi-turn awareness using `ConversationBufferMemory` for coherent, contextually grounded responses.
- **Structured Insight Generation**: Summarizes KPIs with clear metrics, statistical breakdowns, and actionable business intelligence.
- **Interactive Visualizations**: Dynamic visual exploration of business data via Matplotlib.
- **LLM Evaluation Suite**: Built-in semantic evaluation framework using LangChain's `QAEvalChain` for accuracy scoring and feedback logging.

---


## 🛠️ Tech Stack

| Layer              | Stack / Tool                                                                 |
|------------------  |------------------------------------------------------------------------------|
| • **Frontend UI**  | [Streamlit](https://streamlit.io)                                            |
| • **LLM Backbone** | [OpenAI GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5)      |
| • **Orchestration**| [LangChain](https://python.langchain.com/), LangChain Agents & Tools         |
| • **Vector Search**| [FAISS](https://github.com/facebookresearch/faiss)                           |
| • **Embeddings**   | `OpenAIEmbeddings` (text-to-vector for summaries)                            |
| • **Prompting**    | `ChatPromptTemplate`, with agent-based RAG injection                         |
| • **Visualization**| `Matplotlib`, `hvPlot`                                                       |
| • **Evaluation**   | LangChain’s `QAEvalChain` for automated scoring                              |

---


## 🧭 How It Works

### 🧾 Data Preparation
1. Sales CSV is parsed and grouped by product, region, and month.
2. Each chunk is converted into a human-readable summary.
3. Summaries are embedded via OpenAI’s embedding model and stored in FAISS.

### ✏️ Query Processing
- User query is routed through a LangChain agent equipped with:
  - FAISS retriever
  - Context-aware prompt
  - Memory buffer for dialogue history
  - Final output is rendered with structure: `Summary`, `Figures`, `Insights`.

### 📊 Visualization
- Sidebar options allow users to select from 10+ business intelligence charts:
  - Monthly/Yearly trends
  - Product/regional performance
  - Satisfaction breakdown
  - Demographic distributions

### 🔍 LLM Evaluation
- Last AI response is scored against a ground truth or heuristics.
- Uses prompt-based evaluators and logs reasoning, scores, timestamps.

### 🧠 Memory & Agent Design
- The assistant wraps an LLM agent around:
	A Tool that runs a RAG pipeline: memory + retrieval + prompting
	An agent type: "conversational-react-description" to simulate expert consultation
	A full memory store injected via RunnableLambda

---


## 📁 Project Structure

```bash
📦 bizVectra/
├── bizVectra_app.py           # Main Streamlit UI and business flow logic
├── bizVectra_APIs.py          # Core NLP, charting, and LLM-RAG functionality
├── bar-chart-purple.png       # UI image asset
├── bar-chart-blue.png         # UI image asset
├── sales_data.csv             # Sample sales data file
├── README.md                  # You're here


🖥️ Getting Started
________________________________________
pip install streamlit
pip install langchain
pip install faiss-cpu
pip install openai
pip install pandas
pip install matplotlib
pip install -r requirements.txt

streamlit run bizVectra_app.py



👩‍💻 Author
________________________________________
Sampreeti Alam

LangChain | LLM | pandas | streamlit | AI for Real-Time Insights


📝 License
________________________________________
MIT License – free to use, extend, and commercialize with attribution.
