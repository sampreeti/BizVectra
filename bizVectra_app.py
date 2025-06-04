import streamlit as st
import json
import pandas as pd
import base64
from bizVectra_APIs import user_request, plot_sales_insights, display_response_to_evaluate, evaluate_model, color_score

    
# ----------------- STREAMLIT APP ----------------- #

if "eval_history" not in st.session_state:
        st.session_state.eval_history = []

# Main Streamlit Function
def run_streamlit_app():
    
    st.set_page_config(page_title="Sales Assistant", layout='wide', page_icon="📊")
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .stContainer {
            background-color: #000000;
            color: #ffffff;
            position: top;
        }
        .stTitle {
            color: #ffffff;
            font-size: 2rem;
            font-weight: bold;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    
    # Read and encode image
    with open("bar-chart-purple.png", "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    # Embed image in markdown
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{encoded}" width="40" style="margin-right: 10px;" />
            <h1 style="margin: 0;">BizVectra</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(":violet[**_From Data to Decision, AI with Business Vision!_**]")
    st.markdown("""
        <style>
        [data-testid="stHeader"] {
            background: #000000;
        }
        </style>""",
        unsafe_allow_html=True) 
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000;
        }
        .stTitle {
        color: #ffffff;
        font-size: 2rem;
        font-weight: bold;
        position: fixed;
        }
        </style>""",
        unsafe_allow_html=True,
    )


    ## Sidebar 
    m = st.markdown("""
        <style>
            [data-testid=stSidebar] {
                background-color: rgb(0, 30, 40);
            }
            div[data-baseweb="listbox"] > ul {
                background-color: green;
                border-color: #2d408d;
            }
            div[data-baseweb="select"] > div {
                background-color: #12222f;
                color: #ffffff;
            }
        </style>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
            # Read and encode image
            with open("bar-chart-blue.png", "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()

            # Embed image in markdown
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{encoded}" width="20" style="margin-right: 10px;"EWW />
                    <span style="font-size: 20px; font-weight: bold;">Explore Sales Data</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.sidebar.markdown(":lightgrey[Analyze your business performance with AI-powered insights.]")
            st.markdown("<br><br>", unsafe_allow_html=True)     # Pushes the selectbox down by Two lines
            
            category_options = [
                    "Sales Performance ▶",
                    "  Monthly Sales Trends",
                    "  Yearly Sales Trends",
                    "  Yearly Performance by Percentage",
                    "Product Performance ▶",
                    "  Top Products by Sales",
                    "  Product-wise Satisfaction",
                    "Regional Analysis ▶",
                    "  Sales by Region",
                    "  Satisfaction by Region",
                    "Customer Demographics and Segmentation ▶>",
                    "  Gender Distribution",
                    "  Age Distribution",
                    "  Customer Satisfaction"
                ]
            selected_category = st.sidebar.selectbox("Select a chart to display:", category_options)
                
            chat_prompt = f'''
            You are an AI ChatBot intended to help users with sales data.
            \nTOPIC: {category_options} 
            \nUSER MESSAGE: "Get me the updates on: {category_options}"
        '''
            m = st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: rgb(33, 55, 77);
                    color: white;
                    border-radius:10px 10px 10px 10px;
                }
                .css-1aumxhk {
                text-shadow: 1px 1px 2px #00000033;}
                </style>""", unsafe_allow_html=True)
            st.subheader("Plot Sales Data")
            if "show_plot" not in st.session_state:
                st.session_state.show_plot = False

            # Only when a plot category is selected
            if selected_category and st.button("💹 Generate Chart"):
                    st.session_state.show_plot = True
            if st.sidebar.button("🔄 Reset"):
                st.session_state.show_plot = False
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button(":blue-background[Clear Chat]"):
                st.session_state.messages = []
                st.session_state["latest_interaction"] = []
    
    
    main_col, right_col = st.columns([9,1])
    with right_col:
        if "evaluate_response" not in st.session_state:
            st.session_state.evaluate_response = False
        #with st.container():
        st.markdown("""
        <style>
            .sticky-btn {
                position: sticky;
                top: 10px;
                z-index: 999;
            }
            .css-1aumxhk {
            text-shadow: 1px 1px 2px #00000033;}
        </style>""", unsafe_allow_html=True)
        if st.button("✅Evaluate"):
            if "latest_interaction" in st.session_state:
                st.session_state.evaluate_response = True
            else:
                st.warning("Ask a question first!") 
        

    with main_col:         
         m = st.markdown(
            """
            <style>
            .stChatInput > div{
            background-color: #122222;
            }
            </style>""", unsafe_allow_html=True) 
         
         # ⌨️ Input UI   
         user_input = st.chat_input("Question about sales data...")
         if user_input:
             get_answer = user_request(user_input)

         if st.session_state.evaluate_response == True:
            #st.session_state.evaluate_response = False
            display_response_to_evaluate()

         #print("SELECTEDDDDD CATEGORY:", selected_category)
         parent_category, child_category = selected_category, None
         if not selected_category.startswith(" "):
             parent_category = selected_category
         else:
             child_category = selected_category.strip()
         if child_category and st.session_state.show_plot == True:
            st.session_state.show_plot = False
            with st.container():
                with st.expander("Sales Chart Dashboard: {}".format(selected_category), expanded=True):
                    col1, col2, col3 = st.columns([1, 5, 1])
                    with col2:
                        #print("PARENT AND CHILD SSSSUUUBBBY:", parent_category, child_category)
                        plot_sales_insights(child_category)
         
        
        # Display Evaluation Panel with evaluation Result and Log    
         st.markdown("</div>", unsafe_allow_html=True)   
         if "latest_interaction" in st.session_state and st.session_state.evaluate_response == True:# st.session_state.latest_interaction:
            score, reasoning = evaluate_model()
            st.markdown("#### Evaluation Result:")
            st.markdown(color_score(score), unsafe_allow_html=True)
            st.markdown(f"""
            - **Score:** {score}
            - **Reasoning:** {reasoning}
            """)
     
         with st.expander("📝 Evaluation Log", expanded=False):
            if st.session_state.eval_history:
                #st.dataframe(pd.DataFrame(st.session_state.eval_history))
                log_df = pd.DataFrame(st.session_state.eval_history)
                # Optional: Clean or reorder columns for export
                export_df = log_df[["Timestamp", "Query", "Prediction", "Score", "Reasoning"]]
                # Convert to CSV bytes
                csv_data = export_df.to_csv(index=False).encode("utf-8")
                # Download button just above the table
                st.download_button(
                    label="📥 Download Evaluation",
                    data=csv_data,
                    file_name="evaluation_log.csv",
                    mime="text/csv",
                )
                table_html = log_df.to_html(index=False, classes='styled-table')
                st.markdown("""
                    <style>
                    .styled-table {
                        border-collapse: collapse;
                        margin: 0 auto;
                        font-size: 0.9em;
                        font-family: sans-serif;
                        min-width: 400px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }
                    .styled-table thead tr {
                        background-color: #054436;
                        color: #ffffff;
                        text-align: left;
                    }
                    .styled-table th,
                    .styled-table td {
                        padding: 12px 15px;
                    }
                    .styled-table tbody tr {
                        border-bottom: 1px solid #dddddd;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown(table_html, unsafe_allow_html=True)

            else:
                st.info("No evaluations yet.")


    return


if __name__ == "__main__":
    run_streamlit_app()
