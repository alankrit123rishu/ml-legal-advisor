#!/usr/bin/env python
import streamlit as st
import pymongo
import faiss
import re
import numpy as np
import sys
import os
import pandas as pd
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="ML Legal Advisor Chat with Accuracy Metrics")

# --- Initialize Session State Variables ---
# Initialize ALL session state variables FIRST before any function definitions or other code
if "db_status" not in st.session_state:
    st.session_state.db_status = {"connected": False, "error": None}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI Legal Advisor. How can I assist you today?"}]

if "accuracy_metrics" not in st.session_state:
    st.session_state.accuracy_metrics = {
        "total_queries": 0,
        "relevant_queries": 0,
        "user_satisfaction": [],
        "mrr_scores": [],
        "ndcg_scores": [],
        "precision_at_k": [],
        "recall_at_k": []
    }

if "current_query" not in st.session_state:
    st.session_state.current_query = {"query": "", "results": []}

if "show_metrics_report" not in st.session_state:
    st.session_state.show_metrics_report = False

# Ensure pip is installed and up to date
try:
    from sentence_transformers import SentenceTransformer
    import spacy
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.stop()

# --- MongoDB Connection ---
# Use an environment variable for the MongoDB URI, with a fallback for local testing
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
@st.cache_resource(show_spinner="Connecting to Database...")
def get_mongo_collection():
    """Connects to MongoDB and returns the collection or uses demo data if connection fails."""
    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        #client.admin.command('ping')
        db = client["mllegaladvisordb"]
        collection = db["bns_sections"]
        
        # Initialize metrics collection if it doesn't exist
        if "query_metrics" not in db.list_collection_names():
            db.create_collection("query_metrics")
        
        print("Successfully connected to MongoDB.")
        # Update db_status to indicate successful connection
        st.session_state.db_status = {"connected": True, "error": None}
        return collection, db["query_metrics"]
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        # Update db_status to indicate connection failure
        st.session_state.db_status = {"connected": False, "error": str(e)}
        
        # Instead of stopping the app, create mock collections for demo mode
        from pymongo.collection import Collection
        from bson.objectid import ObjectId
        import datetime
        
        # Sample legal sections data
        sample_sections = [
            {
                "_id": ObjectId(),
                "BNS_Section": "123",
                "IPC_Section": "456",
                "Description": "Property rights and boundaries between neighboring properties."
            },
            {
                "_id": ObjectId(),
                "BNS_Section": "234",
                "IPC_Section": "567",
                "Description": "Contract law and enforcement of written agreements between parties."
            },
            {
                "_id": ObjectId(),
                "BNS_Section": "345",
                "IPC_Section": "678",
                "Description": "Liability for damages caused by negligence or intentional actions."
            }
        ]
        
        # Create mock collections
        class MockCollection:
            def __init__(self, name, sample_data=None):
                self.name = name
                self.data = sample_data or []
            
            def find(self, query=None):
                return self.data
            
            def insert_one(self, document):
                document["_id"] = ObjectId()
                self.data.append(document)
                return document
            
            def update_one(self, query, update, upsert=False):
                # Simple implementation for demo
                return True
                
            def find_one(self, query):
                for doc in self.data:
                    # Very simple query matching for demo
                    if all(doc.get(k) == v for k, v in query.items() if k in doc):
                        return doc
                return None
        
        print("Running in demo mode with sample data...")
        return MockCollection("bns_sections", sample_sections), MockCollection("query_metrics")

# --- Resource Loading ---
@st.cache_resource(show_spinner="Loading AI models and data...")
def load_models_and_data(_collection):
    """Loads data, Sentence Transformer, spaCy model, and creates FAISS index."""
    try:
        documents = list(_collection.find())
    except Exception as e:
        st.error(f"Failed to fetch documents from MongoDB: {e}")
        st.stop()

    if not documents:
        st.warning("No documents found in the 'bns_sections' collection.")
        st.stop()

    valid_docs = [doc for doc in documents if doc.get("Description")]
    if not valid_docs:
        st.warning("No documents with valid 'Description' fields found for analysis.")
        st.stop()
    print(f"Found {len(valid_docs)} documents with descriptions.")

    descriptions = [doc["Description"] for doc in valid_docs]

    # --- Load Sentence Transformer Model ---
    # Use a relative path that assumes the fine-tuned model is in a 'models' folder
    model_path = os.path.join(os.getcwd(), 'models', 'fine_tuned_model')
    try:
        if os.path.exists(model_path):
            print(f"Loading Sentence Transformer model from: {model_path}")
            model = SentenceTransformer(model_path)
        else:
            print(f"Custom model not found at {model_path}. Using pre-trained model instead.")
            model = SentenceTransformer('all-MiniLM-L6-v2')  # A good general-purpose model
        print("Sentence Transformer model loaded successfully.")
    except Exception as e:
        st.error(f"Fatal Error: Failed to load Sentence Transformer model: {e}")
        st.error("Attempting to use default model instead...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Default Sentence Transformer model loaded successfully.")
        except Exception as inner_e:
            st.error(f"Still failed to load default model: {inner_e}")
            st.stop()

    # --- Load spaCy Model ---
    # The spaCy model is now installed via requirements.txt, so we just need to load it.
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded successfully.")
    except Exception as e:
        st.error(f"Fatal Error: Failed to load spaCy model: {e}")
        st.stop()

    # --- Create Embeddings and FAISS Index ---
    try:
        print(f"Creating embeddings for {len(descriptions)} descriptions...")
        progress_bar = st.progress(0, text="Generating document embeddings...")
        embeddings = model.encode(
            descriptions,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        progress_bar.progress(1.0, text="Embeddings generated.")
        time.sleep(0.5)
        progress_bar.empty()

        embeddings_np = np.array(embeddings).astype("float32")

        if embeddings_np.ndim != 2 or embeddings_np.shape[0] != len(descriptions):
            st.error(f"Embedding shape mismatch: Got {embeddings_np.shape}")
            st.stop()

        d = embeddings_np.shape[1]
        print(f"Embedding dimension: {d}")
        print("Building FAISS index...")
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_np)
        print(f"FAISS index created successfully with {index.ntotal} vectors.")
        st.success(f"AI models and FAISS index ready ({index.ntotal} documents indexed).")

    except Exception as e:
        st.error(f"Fatal Error: Failed to create embeddings or FAISS index: {e}")
        st.stop()

    return valid_docs, model, index, nlp

# --- Accuracy Metrics Functions ---

def calculate_mrr(relevance_list):
    """
    Calculate Mean Reciprocal Rank
    relevance_list: List of boolean values indicating if a result is relevant
    """
    for i, rel in enumerate(relevance_list):
        if rel:
            return 1.0 / (i + 1)
    return 0.0

def calculate_ndcg(relevance_scores, k=None):
    """
    Calculate Normalized Discounted Cumulative Gain
    relevance_scores: List of relevance scores (higher = more relevant)
    k: Number of results to consider
    """
    if not relevance_scores:
        return 0.0
    
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    dcg = relevance_scores[0]
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 2)
    
    # Calculate ideal DCG
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = ideal_relevance[0]
    for i in range(1, len(ideal_relevance)):
        idcg += ideal_relevance[i] / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(relevant_items, retrieved_items, k):
    """
    Calculate precision@k
    relevant_items: Set of relevant item IDs
    retrieved_items: List of retrieved item IDs
    k: Number of results to consider
    """
    retrieved_k = retrieved_items[:k]
    if not retrieved_k:
        return 0.0
    
    relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
    return relevant_retrieved / len(retrieved_k)

def recall_at_k(relevant_items, retrieved_items, k):
    """
    Calculate recall@k
    relevant_items: Set of relevant item IDs
    retrieved_items: List of retrieved item IDs
    k: Number of results to consider
    """
    if not relevant_items:
        return 0.0
    
    retrieved_k = retrieved_items[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
    return relevant_retrieved / len(relevant_items)

def log_query_metrics(query, results, user_feedback=None, relevant_sections=None):
    """
    Log metrics for each query to MongoDB
    query: User's query string
    results: List of returned section IDs
    user_feedback: User's relevance feedback (1-5)
    relevant_sections: List of actually relevant section IDs (for ground truth if available)
    """
    metrics_entry = {
        "timestamp": datetime.now(),
        "query": query,
        "results": [str(r) for r in results],
        "user_feedback": user_feedback,
        "relevant_sections": relevant_sections
    }
    
    # Calculate MRR if we have ground truth
    if relevant_sections:
        relevance_list = [r in relevant_sections for r in results]
        metrics_entry["mrr"] = calculate_mrr(relevance_list)
        
        # Calculate relevance scores (1 if relevant, 0 if not)
        relevance_scores = [1.0 if r in relevant_sections else 0.0 for r in results]
        metrics_entry["ndcg"] = calculate_ndcg(relevance_scores)
        
        # Calculate precision and recall at different k values
        for k in [1, 3, 5]:
            if k <= len(results):
                metrics_entry[f"precision_at_{k}"] = precision_at_k(relevant_sections, results, k)
                metrics_entry[f"recall_at_{k}"] = recall_at_k(relevant_sections, results, k)
    
    # Insert metrics into MongoDB
    try:
        metrics_collection.insert_one(metrics_entry)
    except Exception as e:
        print(f"Failed to log metrics: {e}")

# --- Helper Functions ---

def preprocess_query(query_text):
    """Preprocesses the query: lowercase, remove special chars, lemmatize, remove stop words."""
    if not query_text or not isinstance(query_text, str):
        return ""
    query_text = re.sub(r"[^a-zA-Z0-9\s]", "", query_text.lower())
    doc = nlp(query_text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
    processed_query = " ".join(lemmas)
    return processed_query if processed_query.strip() else query_text

def generate_advice(user_query, top_docs, distances):
    """Generates a conversational advice string based on top matching documents."""
    if not top_docs:
        return "I couldn't find specific legal sections closely related to your query based on the available data.", []
    
    # Collect section IDs for metrics tracking
    result_ids = [doc.get('_id') for doc in top_docs]
    
    advice = f"Based on your query regarding '{user_query}', here are some areas that might be relevant according to the information I have:\n\n"
    for i, (doc, dist) in enumerate(zip(top_docs, distances)):
        bns = doc.get('BNS_Section', 'N/A')
        ipc = doc.get('IPC_Section', 'N/A')
        desc = doc.get("Description", "No description available.")
        similarity_score = 1 / (1 + dist + 1e-6)
        advice += (
            f"{i+1}. Sections {bns} (BNS) / {ipc} (IPC): "
            f"This section pertains to: \"{desc[:200]}...\" "
            f"(Relevance score: {similarity_score:.2f})\n"
        )

    advice += "\n---\n*Disclaimer:* I am an AI assistant. This information is based on matching your query to legal section descriptions and is not legal advice. Laws are complex and nuanced. You should consult with a qualified legal professional for advice tailored to your specific situation."
    
    # Add feedback request for metrics collection
    advice += "\n\n*Was this helpful?* Please rate the relevance of these results (1-5) using the feedback buttons in the sidebar."
    
    return advice, result_ids

# --- Initialize Application ---
# Now that session state is initialized, we can get the collections
collection, metrics_collection = get_mongo_collection()
documents, model, index, nlp = load_models_and_data(collection)

# --- Streamlit UI ---

st.title("⚖ Conversational ML Legal Advisor with Accuracy Metrics")
st.caption("Ask a question about a legal situation, and I'll try to identify potentially relevant sections.")

# Check database status and show warning if needed
if not st.session_state.db_status["connected"]:
    st.warning(f"⚠ Database connection issue: {st.session_state.db_status['error']}. Some features may be limited.")
    st.info("The app will run in demo mode with sample data.")

# Main area and sidebar layout
main_content, metrics_sidebar = st.columns([3, 1])

with main_content:
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your legal question?"):
        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query and generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analysing your query...")

            try:
                # Track metrics
                st.session_state.accuracy_metrics["total_queries"] += 1
                
                # 1. Preprocess Query
                cleaned_query = preprocess_query(prompt)
                
                if not cleaned_query.strip():
                    response = "Your query seems empty after processing. Could you please provide more details or rephrase your question?"
                    message_placeholder.markdown(response)
                    result_ids = []
                else:
                    # 2. Embed Query
                    query_embedding = model.encode(cleaned_query).astype("float32")

                    # Ensure embedding is 2D for FAISS search
                    if query_embedding.ndim == 1:
                        query_embedding = query_embedding.reshape(1, -1)

                    # 3. Search FAISS Index
                    k = 5  # Increased to 5 for better metrics analysis
                    distances, indices = index.search(query_embedding, k=k)

                    # 4. Retrieve Matching Documents
                    top_docs_data = []
                    top_distances_data = []
                    if indices.size > 0 and distances.size > 0:
                        for i, idx in enumerate(indices[0]):
                            doc_index = int(idx)
                            if doc_index != -1 and 0 <= doc_index < len(documents):
                                top_docs_data.append(documents[doc_index])
                                top_distances_data.append(distances[0][i])

                    # 5. Generate Advice
                    response, result_ids = generate_advice(prompt, top_docs_data, top_distances_data)
                    message_placeholder.markdown(response)

                # Store current query info for feedback
                st.session_state.current_query = {
                    "query": prompt,
                    "results": result_ids
                }

                # Add assistant's final response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                print(f"Error during query processing: {e}")
                st.exception(e)
                error_message = f"Sorry, I encountered an error while processing your request: {e}. Please try again."
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Metrics Sidebar ---
with metrics_sidebar:
    st.header("Metrics Dashboard")
    
    # User feedback collection
    st.subheader("Rate the Results")
    cols = st.columns(5)
    for i in range(5):
        if cols[i].button(f"{i+1} ⭐", key=f"rating_{i+1}"):
            if "current_query" in st.session_state:
                # Record user feedback
                st.session_state.accuracy_metrics["user_satisfaction"].append(i+1)
                
                # Log complete metrics
                log_query_metrics(
                    st.session_state.current_query["query"],
                    st.session_state.current_query["results"], 
                    user_feedback=i+1
                )
                
                st.success(f"Thank you for rating the results as {i+1}/5!")

    # Display accuracy metrics summary
    st.subheader("System Performance")
    metrics_summary = st.empty()
    
    # Function to display metrics
    def show_metrics_summary():
        total = st.session_state.accuracy_metrics["total_queries"]
        satisfaction = st.session_state.accuracy_metrics["user_satisfaction"]
        avg_rating = sum(satisfaction) / len(satisfaction) if satisfaction else 0
        
        metrics_df = pd.DataFrame({
            "Metric": ["Total Queries", "Rated Queries", "Avg Rating"],
            "Value": [total, len(satisfaction), f"{avg_rating:.2f}/5"]
        })
        
        metrics_summary.dataframe(metrics_df, use_container_width=True, hide_index=True)

    show_metrics_summary()
    
    # Button to view detailed metrics in a separate tab
    if st.button("View Detailed Metrics Report"):
        st.session_state.show_metrics_report = True
    
    st.divider()
    
    # Reset button
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I help you?"}]
        st.rerun()
    
    # About section (collapsible)
    with st.expander("About This System"):
        st.markdown("""
        This application uses AI (Sentence Transformers and FAISS) to find potentially relevant legal sections based on your query description.

        Metrics tracked:
        - User satisfaction ratings
        - Mean Reciprocal Rank (MRR)
        - Normalized Discounted Cumulative Gain (NDCG)
        - Precision and Recall @ K

        Disclaimer:
        This tool provides informational suggestions only and does not constitute legal advice. Always consult a qualified legal professional for help with specific legal matters.
        """)

# --- Metrics Report Tab ---
if st.session_state.get("show_metrics_report", False):
    st.title("Detailed Accuracy Metrics Report")
    
    # Handle metrics display in a try-except block for robustness
    try:
        # Fetch all metrics from MongoDB - with error handling for empty database
        try:
            all_metrics = list(metrics_collection.find())
        except Exception as db_error:
            st.warning(f"Could not retrieve metrics from database: {db_error}")
            st.info("Using sample metrics for demonstration instead.")
            # Sample metrics for demonstration when database is empty or unavailable
            all_metrics = [
                {"timestamp": datetime.now(), "query": "Sample query 1", "user_feedback": 4, 
                 "mrr": 0.75, "ndcg": 0.82, "precision_at_3": 0.67, "recall_at_3": 0.5},
                {"timestamp": datetime.now(), "query": "Sample query 2", "user_feedback": 3, 
                 "mrr": 0.5, "ndcg": 0.65, "precision_at_3": 0.33, "recall_at_3": 0.4}
            ]
        
        # Display summary statistics
        if all_metrics:
            st.header("Summary Statistics")
            
            # Calculate metrics with error handling for missing values
            try:
                # Calculate average user feedback
                feedback_scores = [m.get("user_feedback") for m in all_metrics if m.get("user_feedback")]
                avg_feedback = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
                
                # Calculate average MRR and NDCG
                mrr_scores = [m.get("mrr") for m in all_metrics if m.get("mrr") is not None]
                avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
                
                ndcg_scores = [m.get("ndcg") for m in all_metrics if m.get("ndcg") is not None]
                avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
                
                # Calculate average precision and recall
                precision_at_3 = [m.get("precision_at_3") for m in all_metrics if m.get("precision_at_3") is not None]
                avg_precision = sum(precision_at_3) / len(precision_at_3) if precision_at_3 else 0
                
                recall_at_3 = [m.get("recall_at_3") for m in all_metrics if m.get("recall_at_3") is not None]
                avg_recall = sum(recall_at_3) / len(recall_at_3) if recall_at_3 else 0
            except Exception as calc_error:
                st.error(f"Error calculating metrics: {calc_error}")
                # Default values if calculation fails
                avg_feedback = avg_mrr = avg_ndcg = avg_precision = avg_recall = 0
            
            # Display metrics
            try:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Queries", len(all_metrics))
                col2.metric("Avg. User Rating", f"{avg_feedback:.2f}/5")
                col3.metric("Avg. MRR", f"{avg_mrr:.3f}")
                col4.metric("Avg. P@3", f"{avg_precision:.3f}")
                col5.metric("Avg. R@3", f"{avg_recall:.3f}")
            except Exception as display_error:
                st.error(f"Error displaying metrics: {display_error}")
            
            # Visualizations - with additional error handling
            try:
                st.subheader("Metrics Visualizations")
                
                # Prepare data for charts
                viz_data = pd.DataFrame([
                    {"Metric": "User Rating (÷5)", "Value": avg_feedback/5, "Max": 1.0},  # Normalize to 0-1 scale
                    {"Metric": "MRR", "Value": avg_mrr, "Max": 1.0},
                    {"Metric": "NDCG", "Value": avg_ndcg, "Max": 1.0},
                    {"Metric": "Precision@3", "Value": avg_precision, "Max": 1.0},
                    {"Metric": "Recall@3", "Value": avg_recall, "Max": 1.0}
                ])
                
                # Create horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x="Value", y="Metric", data=viz_data, ax=ax)
                for i, v in enumerate(viz_data["Value"]):
                    ax.text(v + 0.02, i, f"{v:.3f}", va="center")
                ax.set_xlim(0, 1.1)  # Scale from 0 to 1 for all metrics
                ax.set_title("System Performance Metrics")
                st.pyplot(fig)
            except Exception as viz_error:
                st.error(f"Error creating visualizations: {viz_error}")
                st.info("Unable to display metrics visualizations due to an error.")
            
            # Query metrics table for the last 10 queries
            try:
                st.subheader("Recent Query Metrics")
                if len(all_metrics) > 0:
                    # Convert to DataFrame handling ObjectId issues
                    recent_metrics = pd.DataFrame([{k: str(v) if k == '_id' else v 
                                                 for k, v in m.items()} 
                                                for m in all_metrics[-10:]])
                    
                    # Select columns safely
                    display_cols = []
                    for col in ["timestamp", "query", "user_feedback", "mrr", "ndcg", "precision_at_3"]:
                        if col in recent_metrics.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(recent_metrics[display_cols], use_container_width=True)
                    else:
                        st.dataframe(recent_metrics, use_container_width=True)
                    
                    # Download button with error handling
                    try:
                        csv_data = pd.DataFrame([{k: str(v) if k == '_id' else v 
                                               for k, v in m.items()} 
                                              for m in all_metrics]).to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download All Metrics Data (CSV)",
                            csv_data,
                            "legal_advisor_metrics.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    except Exception as csv_error:
                        st.error(f"Error preparing CSV download: {csv_error}")
                else:
                    st.info("No query metrics available yet.")
            except Exception as table_error:
                st.error(f"Error displaying metrics table: {table_error}")
                
        else:
            st.info("No metrics data available yet. Start using the system to generate metrics.")
            
    except Exception as e:
        st.error(f"Error handling metrics report: {e}")
        st.info("Please try refreshing the page or check the database connection.")
    
    # Button to return to main chat
    if st.button("Return to Chat"):
        st.session_state.show_metrics_report = False
        st.rerun()

# --- Export/Import Ground Truth Data ---
if st.sidebar.checkbox("Show Ground Truth Management"):
    st.sidebar.subheader("Ground Truth Management")
    
    # Import ground truth data
    ground_truth_file = st.sidebar.file_uploader("Import Ground Truth Data (JSON)", type=["json"])
    if ground_truth_file and st.sidebar.button("Import Ground Truth"):
        try:
            ground_truth_data = json.load(ground_truth_file)
            # Store in a specialized collection
            for entry in ground_truth_data:
                metrics_collection.update_one(
                    {"query": entry["query"]},
                    {"$set": {"relevant_sections": entry["relevant_sections"]}},
                    upsert=True
                )
            st.sidebar.success(f"Imported {len(ground_truth_data)} ground truth entries.")
        except Exception as import_error:
            st.sidebar.error(f"Failed to import ground truth data: {import_error}")

    # Export ground truth data
    if st.sidebar.button("Export Ground Truth"):
        try:
            ground_truth_entries = list(metrics_collection.find(
                {"relevant_sections": {"$exists": True, "$ne": []}},
                {"query": 1, "relevant_sections": 1}
            ))
            
            if ground_truth_entries:
                # Convert to JSON-friendly format
                export_data = [{
                    "query": entry["query"],
                    "relevant_sections": entry["relevant_sections"]
                } for entry in ground_truth_entries]
                
                # Create downloadable JSON
                json_data = json.dumps(export_data, indent=2)
                st.sidebar.download_button(
                    label="Download Ground Truth",
                    data=json_data,
                    file_name="ground_truth.json",
                    mime="application/json"
                )
            else:
                st.sidebar.warning("No ground truth data available for export.")
        except Exception as export_error:
            st.sidebar.error(f"Export failed: {export_error}")
