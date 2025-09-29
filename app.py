# Job-AiD: AI-Powered Job Matching Platform
import streamlit as st
import pandas as pd
import re
from groq import Groq

# ----------------------------
# Page Setup & Branding
# ----------------------------
st.set_page_config(
    page_title="Job-AiD - AI-Powered Job Matching",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def simple_text_similarity(query, text):
    query_words = set(preprocess_text(query).split())
    text_words = set(preprocess_text(text).split())
    if not query_words or not text_words:
        return 0
    return len(query_words & text_words) / len(query_words | text_words)

class RAGJobSearch:
    def __init__(self, job_data):
        self.job_data = job_data

    def search(self, query, top_k=10):
        results = []
        for _, row in self.job_data.iterrows():
            search_text = f"{row['JobTitle']} {row['Company']} {row['Location']} {row['EmploymentType']}"
            score = simple_text_similarity(query, search_text)
            if score > 0:
                job = row.copy()
                job['SIMILARITY_SCORE'] = score
                results.append((score, job))
        results.sort(key=lambda x: x[0], reverse=True)
        return pd.DataFrame([job for _, job in results[:top_k]])

# ----------------------------
# Groq AI Chatbot Function
# ----------------------------
def get_ai_response(user_message, jobs_df=None):
    """Get AI response using Groq API"""
    try:
        client = Groq(
            api_key="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"  # ✅ Your API key
        )
        
        context = ""
        if jobs_df is not None and not jobs_df.empty:
            total_jobs = len(jobs_df)
            job_types = jobs_df['EmploymentType'].value_counts().to_dict()
            locations = jobs_df['Location'].value_counts().to_dict()
            top_titles = jobs_df['JobTitle'].value_counts().head(3).to_dict()
            
            context = f"""
            Current Philippine Job Market Context:
            - Total available jobs: {total_jobs}
            - Job types: {', '.join([f'{k} ({v})' for k, v in job_types.items()])}
            - Locations: {', '.join([f'{k} ({v})' for k, v in locations.items()])}
            - Popular roles: {', '.join([f'{k} ({v})' for k, v in top_titles.items()])}
            """

        system_prompt = f"""You are Job-AiD, an AI career assistant specialized in the Philippine job market. 
        {context}
        Provide helpful, specific advice about job searching, resumes, interviews, and career development.
        Be concise but thorough in your responses."""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=st.session_state.selected_model,   # ✅ Use selected model dynamically
            temperature=0.7,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"I apologize, but I'm experiencing technical issues. Error: {str(e)}"

# ----------------------------
# Initialize Session State
# ----------------------------
if "jobs_df" not in st.session_state:
    st.session_state.jobs_df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_input" not in st.session_state:
    st.session_state.current_input = ""

if "is_typing" not in st.session_state:
    st.session_state.is_typing = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"

# ----------------------------
# Sidebar: Controls
# ----------------------------
st.sidebar.markdown("## ⚙️ Controls")

# Model selector in sidebar
st.session_state.selected_model = st.sidebar.selectbox(
    "🤖 Choose AI Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0
)

if st.sidebar.button("📂 Load Dataset"):
    try:
        st.session_state.jobs_df = pd.read_csv("philjobnet_jobs_dataset.csv")
        st.sidebar.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading dataset: {e}")

# ----------------------------
# Main Layout
# ----------------------------
st.markdown('<h1 style="text-align:center; color:#4FC3F7;">🤖 Job-AiD</h1>', unsafe_allow_html=True)
st.markdown("### Smart AI-powered job matching and career support")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Job Search", "📋 Job Listings", "📊 Analytics", "💬 AI Career Assistant"])

# ----------------------------
# TAB 1: Job Search (RAG)
# ----------------------------
with tab1:
    st.subheader("AI-Powered Job Search")
    if st.session_state.jobs_df is None:
        st.info("⚠️ Please load the dataset from the sidebar first.")
    else:
        search_query = st.text_input("Describe your ideal job:", placeholder="e.g., Data Analyst in Cebu, Full-time")

        if st.button("Search Jobs"):
            rag_search = RAGJobSearch(st.session_state.jobs_df)
            results = rag_search.search(search_query, top_k=5)

            if not results.empty:
                for _, job in results.iterrows():
                    st.markdown(f"""
                        <div style="background:#f1f1f1;padding:15px;border-radius:10px;margin:10px 0;">
                            <h4>{job['JobTitle']} at {job['Company']}</h4>
                            <p><b>📍 Location:</b> {job['Location']}<br>
                            <b>💼 Employment Type:</b> {job['EmploymentType']}<br>
                            <b>💰 Salary:</b> {job['SalaryRange']}<br>
                            <b>🗓️ Posted:</b> {job['PostedDate']}<br>
                            <b>⭐ Match:</b> {job['SIMILARITY_SCORE']*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No matches found. Try different keywords.")

# ----------------------------
# TAB 2: Job Listings
# ----------------------------
with tab2:
    st.subheader("Job Listings & Filters")
    if st.session_state.jobs_df is None:
        st.info("⚠️ Please load the dataset from the sidebar first.")
    else:
        df = st.session_state.jobs_df

        col1, col2 = st.columns(2)
        with col1:
            job_type_filter = st.selectbox("Job Type", ["All"] + df["EmploymentType"].unique().tolist())
        with col2:
            location_filter = st.selectbox("Location", ["All"] + df["Location"].unique().tolist())

        filtered_df = df.copy()
        if job_type_filter != "All":
            filtered_df = filtered_df[filtered_df["EmploymentType"] == job_type_filter]
        if location_filter != "All":
            filtered_df = filtered_df[filtered_df["Location"] == location_filter]

        st.markdown(f"### 📋 {len(filtered_df)} Matching Jobs")

        for _, job in filtered_df.iterrows():
            st.markdown(f"""
                <div style="background:#f9f9f9;padding:15px;border-radius:10px;margin:10px 0;border:1px solid #ddd;">
                    <h4>💼 {job['JobTitle']} at {job['Company']}</h4>
                    <p><b>📍 Location:</b> {job['Location']}<br>
                    <b>💼 Employment Type:</b> {job['EmploymentType']}<br>
                    <b>💰 Salary:</b> {job['SalaryRange']}<br>
                    <b>🗓️ Posted:</b> {job['PostedDate']}</p>
                </div>
            """, unsafe_allow_html=True)

# ----------------------------
# TAB 3: Analytics
# ----------------------------
with tab3:
    st.subheader("📊 Job Market Analytics")
    if st.session_state.jobs_df is None:
        st.info("⚠️ Please load the dataset from the sidebar first.")
    else:
        df = st.session_state.jobs_df
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Jobs by Employment Type")
            st.bar_chart(df["EmploymentType"].value_counts())
        with col2:
            st.subheader("Jobs by Location")
            st.bar_chart(df["Location"].value_counts())

# ----------------------------
# TAB 4: Dark Minimal Chat Assistant
# ----------------------------
with tab4:
    st.markdown("""
    <style>
    .chat-box {
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 80px;
        padding-right: 10px;
    }
    .chat-message {
        display: flex;
        margin: 12px 0;
        max-width: 75%;
    }
    .chat-message.user {
        justify-content: flex-end;
        margin-left: auto;
    }
    .chat-message.assistant {
        justify-content: flex-start;
        margin-right: auto;
    }
    .bubble {
        padding: 12px 16px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }
    .user .bubble {
        background: #4a90e2;
        color: white;
        border-bottom-right-radius: 5px;
    }
    .assistant .bubble {
        background: #2c2c2c;
        color: #eaeaea;
        border-bottom-left-radius: 5px;
    }
    .typing {
        font-size: 14px;
        color: #bbb;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("<h2 style='text-align:center; color:#4FC3F7;'>Welcome to Job-AiD</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#bbb;'>Ask me about jobs, resumes, interviews, or career advice to get started.</p>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            role = message["role"]
            css_class = "user" if role == "user" else "assistant"
            st.markdown(f"""
                <div class="chat-message {css_class}">
                    <div class="bubble">{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)
        if st.session_state.is_typing:
            st.markdown('<p class="typing">Job-AiD is typing...</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input bar
    with st.container():
        col1, col2 = st.columns([8,1])
        with col1:
            user_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed")
        with col2:
            if st.button("Send", key="send_btn", use_container_width=True) and user_input.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.is_typing = True
                st.rerun()

    if st.session_state.is_typing and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        response = get_ai_response(st.session_state.chat_history[-1]["content"], st.session_state.jobs_df)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.is_typing = False
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.is_typing = False
            st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("### 💡 Tips for using Job-AiD:")
st.markdown("- Load the dataset first from the sidebar")
st.markdown("- Use the sidebar to switch models (speed vs quality)")
st.markdown("- Ask specific questions for better career advice")
st.markdown("- The AI assistant uses real job market data from the Philippines")
