import streamlit as st
import os
import re
import time
import holoviews as hv
import csv
import pandas as pd
import hvplot.pandas
import holoviews as hv
hv.extension('bokeh') 
from datetime import datetime
import matplotlib.pyplot as plt #' or 'plotly'
import matplotlib.cm as cm
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
from langchain.chains import ConversationChain
from langchain.schema.output_parser import StrOutputParser

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.evaluation.qa import QAEvalChain


# Get the API keys
OPENAI_API_KEY = "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"#db.secrets.get(name="OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load data-set
## This code fetches the data from a CSV file, parses dates, and sets appropriate column names ##
csv_file_path = r"C:\Users\sampr\OneDrive\Documents\AI Literacy\Agentic Frameworks\AGAI_lesson_2_demos\AGAI_lesson_2_demos\Hurr\sales_data.csv"

data = pd.read_csv(
    csv_file_path,
    skiprows=[0, 0],
    parse_dates=True,
    date_format="%y%m%d %H",
    names=["Date", "Product", "Region", "Sales", "Customer_Age", "Customer_Gender", "Customer_Satisfaction"],
).reset_index()

df = pd.DataFrame(data)

## ............ Chunk the Data and Embed Statistics ........... ##
# Convert date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Chunk the Data Logically (Group by Product/Year/Region). Rather than raw row-by-row splitting, chunk it into semantic summaries:
# Group by product and year-month
df["year_month"] = df["Date"].dt.to_period("M").astype(str)
df_grouped = df.groupby(["Product", "year_month"])

# Summarize Each Chunk into Natural Language - Convert each group to a summary string with structured metadata:
sales_documents = []

for (product, year_month), group in df_grouped:
    total_sales = group["Sales"].sum()
    product = group["Product"].iloc[0]  
    region = group["Region"].iloc[0]    
    customer_age = group["Customer_Age"].iloc[0]
    customer_gen = group["Customer_Gender"].iloc[0]
    customer_sat = group["Customer_Satisfaction"].iloc[0] # Assume consistent within group
    avg_age = group["Customer_Age"].mean()
    avg_satisfaction = group["Customer_Satisfaction"].mean()
    gender_distribution = group["Customer_Gender"].value_counts(normalize=True).round(2).to_dict()

    summary = (
        f"Sales Summary\n"
        f"Product: {product}\n"
        f"Date: {year_month}\n"
        f"Region: {region}\n"
        f"Total Sales: ${total_sales:,.2f}\n"
        f"Average Customer Age: {avg_age:.1f}\n"
        f"Average Satisfaction: {avg_satisfaction:.2f}/5.0\n"
        f"Gender Distribution: {gender_distribution}\n"
    )
    
    #print("TEXTTTTTTTT:", text)
    sales_documents.append(Document(
        page_content=summary,
        metadata={
            "product": product,
            "date": year_month,
            "region": region,
            "sales" : total_sales,
        }
    ))

# Embed the text to vector embeddings
embeddings = OpenAIEmbeddings()
# Create the vector database to save the vector embeddings/representation of the structured data incorporating 'FAISS'
vectordb = FAISS.from_documents(sales_documents, embeddings)
# --- Save index locally (optional) ---
vectordb.save_local("sales_faiss_index")
# Use FAISS retriever
base_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0)       # Define Model
# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a prompt template using 'ChatPromptTemplate' imported from LangChain
template_string ="""You are an AI-powered Business assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. 
Use only 2 sentences to answer the question.
Keep the answer clear, accurate and contextually relevant at the same time.
Question: {question}
Context: {context}
Chat History: {chat_history}
Answer in a clear and structured format:
- Summary
- Any figures or statistics
- Insights
"""
prompt_template_with_memory = ChatPromptTemplate.from_template(template_string)
output_parser = StrOutputParser()


# ========================= FUNCTIONS ===========================

def user_request(query):
    # Initialize chat history via st.session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Add the user's query to the chat history
    st.session_state.messages.append({"role": "Human", "content": query})
    # Display the user's query in the chat message
    with st.chat_message("Human"):
        message = st.session_state.messages[-1]
        st.markdown(message["content"])
    
    return process_messages(query)


def process_messages(query):
    # Initializing Memory via st.session_state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Style the chat message container to display the query and response
    st.markdown(
            """
            <style>
            .stChatMessage {
                background-color: #111322;
                length: 100%;
                width: 100%;
            }
             .stChatMessage > div:nth-child(1) {
                background-color: #12222f;
            }
            .stMarkdown > div:nth-child(1) {
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    return get_response(query)


# Create a Memory Adapter (inject chat history and pull the memory at runtime)
def with_memory(x):
    chat_history = memory.load_memory_variables({})["chat_history"]
    return {
        "question": x,
        "chat_history": chat_history
    }
memory_adapter = RunnableLambda(with_memory)


def get_response(query):    
    # Build the final RAG pipeline by chaining the pre-designed prompt template, chat model and output parser together
    rag_context = lambda x: "\n\n".join(doc.page_content for doc in base_retriever.get_relevant_documents(x["question"]))
    RAG_pipeline = (
        memory_adapter
        | RunnableMap({
            "context": rag_context,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        })
        | prompt_template_with_memory
        | llm
        | output_parser   
    )
    
    # Invoke the pipeline to respond to multiple queries with memory integrated to retain previous conversation
    def rag_with_memory(query: str) -> str:
        # Get the answer from the chain
        start = time.time()
        response = RAG_pipeline.invoke(query)
        end = time.time()
        memory.save_context({"input":query}, {"output": response})
        print(response)
        return response

    # Initialize tools
    # Wrap the RAG System into a Tool: Lets the agent call the full RAG + memory pipeline like an API.
    RAG_tool = Tool(
        name="sales_data_qa",
        func=rag_with_memory,
        description="Useful for answering questions about sales data and trends"
    )

    # Initialize an Agent with Memory + Tools
    business_agent = initialize_agent(
        tools=[RAG_tool],
        llm=llm,
        agent="conversational-react-description",
        memory=memory,
        verbose=True
    )

    
    # Generate the AI's response
    with st.container():
        st.markdown(
        """
        <style>
        div.stContainer{
        'overflow-y: scroll; height: 500px; padding: 10px; border: 1px solid #ddd; border-radius: 8px'}
        </style>""",unsafe_allow_html=True,
        )
        with st.chat_message("AI"):            
            # Set up the Streamlit callback handler
            st_callback = StreamlitCallbackHandler(st.container())
            message_placeholder = st.empty()
            full_response = ""
            ai_response = business_agent.run(query, callbacks=[st_callback])

            # Simulate a streaming response with a slight delay
            for chunk in ai_response.split():
                full_response += chunk + " "
                time.sleep(.03)

                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        
            # Display the full response
            message_placeholder.info(full_response)

        # Add the AI's response to the chat history
        st.session_state.messages.append({"role": "AI", "content": full_response})

        # Display the Human's query and the AI's response in the chat message history 
        for i in range(len(st.session_state.messages)-3, -1,-2):
            message_Human = st.session_state.messages[i-1]
            message_AI = st.session_state.messages[i]
            with st.chat_message(message_Human["role"]):
                    st.markdown(message_Human["content"])
            with st.chat_message(message_AI["role"]):
                    st.markdown(message_AI["content"])

        # Store everything in session_state for evaluation later
    if st.session_state.messages != []:
        st.session_state["latest_interaction"] = {
            "query": st.session_state.messages[-2],  
            "result": st.session_state.messages[-1], 
            "context": rag_context,          # <- Save retrieved context
            "answer": ""                     # <- Optional ground truth (if known)
        }
    
    return


def evaluate_model():
    latest = st.session_state.get("latest_interaction", None)
    if not latest:
        return "N/A", "No query to evaluate."

    example = {
        "query": latest["query"],
        "context": latest["context"],
        "answer": latest["answer"]
    }

    prediction = {
        "result": latest["result"]
    }

    eval_prompt = PromptTemplate.from_template(
    """You are a strict evaluator of a question-answering system.
    Compare the following predicted answer with the reference (ground truth).
    Rate it on a scale from 0 (completely incorrect) to 10 (perfectly correct).

    Question: {query}
    Reference Answer: {answer}
    Predicted Answer: {result}

    Score (0-10):
    Reasoning:"""
    )
    
    eval_chain = QAEvalChain.from_llm(llm = llm, prompt = eval_prompt)
    result = eval_chain.evaluate([example], [prediction])[0]
    #st.write("Evaluation Result:", result["results"])

    eval_text = result["results"]
    
    # Extract score
    score_match = re.search(r"Score\s*:\s*(\d+)", eval_text)
    score_e = score_match.group(1) if score_match else "N/A"

    # Extract reasoning (everything after “Reasoning:”)
    reasoning_match = re.search(r"Reasoning\s*:\s*(.*)", eval_text, re.DOTALL)
    reasoning_e = reasoning_match.group(1).strip() if reasoning_match else "N/A"
    
    # Log the evaluation result
    st.session_state.eval_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Query": latest["query"]["content"],
        "Prediction": latest["result"]["content"],
        "Score": score_e,
        "Reasoning": reasoning_e
    })
    return score_e, reasoning_e


def display_response_to_evaluate():
    # Style the chat message container to display the query and response
    st.markdown(
            """
            <style>
            .stChatMessage {
                background-color: #111322;
                length: 100%;
                width: 100%;
            }
            .stChatMessage > div:nth-child(1) {
                background-color: #12222f;
            }
            .stMarkdown > div:nth-child(1) {
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    if st.session_state["latest_interaction"] != []:
        with st.chat_message(st.session_state["latest_interaction"]["query"]["role"]):
            st.markdown(st.session_state["latest_interaction"]["query"]["content"])
        with st.chat_message(st.session_state["latest_interaction"]["result"]["role"]):
            st.markdown(st.session_state["latest_interaction"]["result"]["content"])
    
    return


def plot_sales_insights(sub_category: str):
    """Plot charts based on the selected sales data category."""
    if df is None or df.empty:
        st.warning("Sales data not loaded.")
        return
    
    # Plots by sub-category
    if sub_category == "Monthly Sales Trends":
            df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)
            monthly_sales = df.groupby("Month")["Sales"].sum()
            fig, ax = plt.subplots(figsize=(10, 4))
            monthly_sales.plot(ax=ax, marker='o', color = "#2FB969")
            ax.set_title("Monthly Sales Trends")
            ax.set_ylabel("Sales")
            ax.set_xlabel("Month")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    elif sub_category == "Yearly Sales Trends":
        if {"Date"}.issubset(df.columns):
            df["Year"] = pd.to_datetime(df["Date"]).dt.year
            yearly_sales = df.groupby("Year")["Sales"].sum()
            fig, ax = plt.subplots(figsize=(7, 4))
            yearly_sales.plot(kind='bar', ax=ax, color="#6ee4b3")
            ax.set_title("Yearly Sales Trends")
            ax.set_ylabel("Total Sales")
            ax.set_xlabel("Year")
            plt.xticks(rotation=0)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("Year and Month columns are required.")
    
    elif sub_category == "Yearly Performance by Percentage":
        # Ensure the Date column is datetime and Extract year
        df["Date"] = pd.to_datetime(df["Date"])    
        df["Year"] = df["Date"].dt.year
        # Group by Year and calculate total sales
        yearly_sales = df.groupby("Year")["Sales"].sum().reset_index()
        yearly_sales["Percentage"] = (yearly_sales["Sales"] / yearly_sales["Sales"].sum()) * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = cm.get_cmap("Blues", len(yearly_sales))
        colors = [cmap(i) for i in range(len(yearly_sales))]
        ax.pie(
            yearly_sales["Percentage"],
            labels=yearly_sales["Year"].astype(str),
            autopct="%1.1f%%",
            startangle=90,
            colors=colors
        )
        ax.set_title("Percentage of Sales Performance Over the Years", fontsize=10)
        ax.axis('equal')  # Equal aspect ratio to ensure pie is circular
        st.pyplot(fig)

    elif sub_category == "Top Products by Sales":
        top_products = df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        top_products.plot(kind='bar', ax=ax, color="#7274D1")
        ax.set_title("Top Products by Sales")
        ax.set_ylabel("Total Sales")
        ax.set_xlabel("Product")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif sub_category == "Product-wise Satisfaction":
        avg_satisfaction = df.groupby("Product")["Customer_Satisfaction"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        avg_satisfaction.plot(kind='bar', color="#1dffff", ax=ax)
        ax.set_title("Average Satisfaction by Product")
        ax.set_ylabel("Satisfaction Score")
        ax.set_xlabel("Product")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif sub_category == "Sales by Region":
            if "Region" in df.columns:
                region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 3))
                region_sales.plot(kind="bar", ax=ax, color="#6deed8")
                ax.set_ylabel("Total Sales")
                ax.set_xlabel='Region',
                ax.set_title("Sales by Region")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Region column not found in the dataset.")

    elif sub_category == "Satisfaction by Region":
        region_satisfaction = df.groupby("Region")["Customer_Satisfaction"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 3))
        region_satisfaction.plot(kind='barh', ax=ax, color="#a97df1")
        ax.set_title("Average Satisfaction by Region")
        ax.set_xlabel("Satisfaction Score")
        st.pyplot(fig)

    elif sub_category == "Gender Distribution":
        gender_counts = df["Customer_Gender"].value_counts()
        fig, ax = plt.subplots(figsize=(2, 2))
        gender_counts.plot(kind="pie", autopct="%0.01f%%", startangle=90, ax=ax, colors=["#f3cfff", "#9cffff"])
        ax.set_ylabel("")
        ax.set_title("Gender Distribution", fontsize=5)
        st.pyplot(fig)

    elif sub_category == "Age Distribution":
        fig, ax = plt.subplots(figsize=(7, 4))
        # Create histogram data
        n, bins, patches = ax.hist(df["Customer_Age"].dropna(), bins=20, edgecolor="white")
        cmap = cm.get_cmap("Greens", len(patches))
        for i, patch in enumerate(patches):
            patch.set_facecolor(cmap(i))
        ax.set_title("👤Customer Age Distribution", fontsize=11)
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    elif sub_category == "Customer Satisfaction":
        if "Customer_Satisfaction" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            satisfaction_scores = df["Customer_Satisfaction"].dropna()
            ax.hist(satisfaction_scores, bins=range(1, 8), edgecolor='white', color= "#5da2ff", align='left', rwidth=0.8)
            # Compute mean and median
            mean_score = satisfaction_scores.mean()
            median_score = satisfaction_scores.median()
            # Add lines for mean and median
            ax.axvline(mean_score, color='blue', linestyle='dashed', linewidth=1.5, label=f"Mean: {mean_score:.2f}")
            ax.axvline(median_score, color='red', linestyle='dashed', linewidth=1.5, label=f"Median: {median_score:.2f}")
            ax.set_title("Customer Satisfaction Distribution")
            ax.set_xlabel("Satisfaction Rating")
            ax.set_ylabel("Number of Customers")
            ax.set_xticks(range(1, 7))  # Adjust based on your scale
            st.pyplot(fig)
        else:
            st.warning("No 'Customer_Satisfaction' column found in dataset.")
    
    else:
        st.info("Please select a valid category from the sidebar.")

    

def color_score(score):
    try:
        score = int(score)
        if score >= 8:
            return f"<span style='color: green; font-weight: bold;'>Score: {score} ✅</span>"
        elif 5 <= score < 8:
            return f"<span style='color: orange; font-weight: bold;'>Score: {score} ⚠️</span>"
        else:
            return f"<span style='color: red; font-weight: bold;'>Score: {score} ❌</span>"
    except:
        return f"<span style='color: gray;'>Score: {score}</span>"

   