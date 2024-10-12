# If trying this app locally, comment out these 3 lines
# __import__("pysqlite3")
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import streamlit as st
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain_cerebras import ChatCerebras
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback

from chain import initialize_chain, generate_suggestions

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]["API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "Short Video Master"

# Enhanced Streamlit chatbot interface
st.sidebar.header("✨ Short Video Content Master")
st.sidebar.markdown(
    "This app answers your questions about short video content strategy and execution.  "
    "To use this app, you'll need to provide a Cerebras API key, which you can obtain for free [here](https://cloud.cerebras.ai/platform/org_nxh29kc28dt5rvrcphxv54et/apikeys)..")
st.sidebar.write("### Instructions")
st.sidebar.write(":pencil: Enter your video topic or theme, target audience, unique selling points, and question you want to ask.")
st.sidebar.write(":point_right: Click 'Generate Suggestions' to get inspired.")
st.sidebar.write(":heart_decoration: Tell me your thoughts and feedback about the App.")

st.sidebar.image("assets/logo01.jpg", use_column_width=True)

# ask user for their OpenAI API key via `st.text_input`.
cerebras_api_key = st.text_input("Cerebras API Key", type="password", placeholder="Your Cerebras API Key here...")
if not cerebras_api_key:
    st.info("Please add your Groq API key to continue.", icon="🗝️")
else:
    # Input fields for video details
    st.header("Enter Your Video Details")
    video_topic_input = st.text_input("Enter a Video Topic or Theme", placeholder="Organic Soup")
    target_audience_input = st.text_input("Enter the Target Audience", placeholder="For busy working adults")
    selling_point_input = st.text_input("Enter Unique Selling Points or Opinion", placeholder="Ready in 5 minutes")
    question_input = st.text_input("Enter Your Question", placeholder="How can I make the video hook?")

    # Initialize Cerebras model
    model = ChatCerebras(model="llama3.1-70b", temperature=0.8, api_key=cerebras_api_key)

    # Initialize LangSmith client
    langchain_endpoint = "https://api.smith.langchain.com"
    client = Client(api_url=langchain_endpoint, api_key=st.secrets["LANGCHAIN_API_KEY"]["API_KEY"])

    # Initialize state
    if "trace_link" not in st.session_state:
        st.session_state.trace_link = None
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = "some_initial_value"

    # System prompt setup
    system_prompt = ""  # Add your system prompt here if needed
    system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")

    # Initialize memory
    memory = ConversationBufferMemory(
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
        return_messages=True,
        memory_key="chat_history",
    )

    # Initialize chain
    chain = initialize_chain(_llm=model, system_prompt=system_prompt, _memory=memory)

    # Clear message history button
    if st.sidebar.button("Clear message history"):
        memory.clear()
        st.session_state.trace_link = None
        st.session_state.run_id = None

    # Helper function to get OpenAI message type
    def _get_openai_type(msg):
        if msg.type == "human":
            return "user"
        if msg.type == "ai":
            return "assistant"
        if msg.type == "chat":
            return msg.role
        return msg.type

    # Display chat history
    for msg in st.session_state.langchain_messages:
        streamlit_type = _get_openai_type(msg)
        avatar = "🤖" if streamlit_type == "assistant" else None
        with st.chat_message(streamlit_type, avatar=avatar):
            st.markdown(msg.content)

    # Set up run collector and config
    run_collector = RunCollectorCallbackHandler()
    runnable_config = RunnableConfig(
        callbacks=[run_collector],
        tags=["Streamlit Chat"],
    )

    # Display trace link if available
    if st.session_state.trace_link:
        st.sidebar.markdown(
            f'<a href="{st.session_state.trace_link}" target="_blank"><button>Latest Trace: 🛠️</button></a>',
            unsafe_allow_html=True,
        )

    # Function to reset feedback
    def _reset_feedback():
        st.session_state.feedback_update = None
        st.session_state.feedback = None

    # Character limit for input
    MAX_CHAR_LIMIT = 500

    # Generate suggestions button
    if st.button("Generate Suggestions"):
        if not all([video_topic_input, target_audience_input, selling_point_input, question_input]):
            st.error("Please fill out all fields to generate suggestions.", icon="🚫")
        else:
            prompt = f"Video Topic: {video_topic_input}\nTarget Audience: {target_audience_input}\nUnique Selling Points: {selling_point_input}\nQuestion: {question_input}"
            
            if len(prompt) > MAX_CHAR_LIMIT:
                st.warning(f"⚠️ Your input is too long! Please limit your input to {MAX_CHAR_LIMIT} characters.")
            else:
                st.chat_message("user").write(prompt)
                _reset_feedback()
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    
                    # Use the generate_suggestions function
                    suggestions = generate_suggestions(_llm=model, video_topic=video_topic_input, target_audience=target_audience_input, selling_point=selling_point_input, question=question_input)
                    
                    message_placeholder.markdown(suggestions)
                    memory.save_context({"input": prompt}, {"output": suggestions})

                    # Process run information
                    run = run_collector.traced_runs[0] if run_collector.traced_runs else None
                    if run:
                        run_collector.traced_runs = []
                        st.session_state.run_id = run.id
                        wait_for_all_tracers()
                        url = client.share_run(run.id)
                        st.session_state.trace_link = url
    
    # Feedback section
    has_chat_messages = len(st.session_state.get("langchain_messages", [])) > 0

    if has_chat_messages:
        feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type=feedback_option,
                optional_text_label="[Optional] Please provide an explanation",
                key=f"feedback_{st.session_state.run_id}",
            )

            score_mappings = {
                "thumbs": {"👍": 1, "👎": 0},
                "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
            }

            scores = score_mappings[feedback_option]

            if feedback:
                score = scores.get(feedback["score"])

                if score is not None:
                    feedback_type_str = f"{feedback_option} {feedback['score']}"

                    feedback_record = client.create_feedback(
                        st.session_state.run_id,
                        feedback_type_str,
                        score=score,
                        comment=feedback.get("text"),
                    )
                    st.session_state.feedback = {
                        "feedback_id": str(feedback_record.id),
                        "score": score,
                    }
                else:
                    st.warning("Invalid feedback score.")