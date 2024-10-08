import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.decisionAgent import DecisionAgent, DecisionCase

# Initialize the DecisionAgent
decision_agent = DecisionAgent('gpt-4o-mini')

st.title("Decision Making Assistant")

# Input for the decision case description
description = st.text_area("Describe your problem or situation:", height=150)

if st.button("Analyze"):
    if description:
        try:
            # Perform structured evaluation
            case, choice_analysis, cost_evaluation = decision_agent.evaluate_structured(description)
            
            st.subheader("Problem")
            st.write(case.problem)
            
            st.subheader("Options")
            for i, option in enumerate(case.options, 1):
                st.write(f"{i}. {option.description}")
            
            st.subheader("Analysis")
            st.write(choice_analysis.analysis)
            
            st.subheader("Recommended Decision")
            st.write(choice_analysis.decision)
            
            st.subheader("Opportunity Costs")
            for cost in cost_evaluation.costs:
                st.write(f"**Option:** {cost.choice}")
                st.write(f"**Opportunity Cost:** {cost.opportunity_cost}")
                st.write("---")
            
            # Store the case in session state for later use
            st.session_state.current_case = case
            
        except Exception as e:
            st.error(f"Error in analyzing the description: {str(e)}")
    else:
        st.warning("Please provide a description of your problem or situation.")

# Only show the decision input and outcome prediction if we have a current case
if 'current_case' in st.session_state:
    decision = st.text_input("What is your final decision?")
    
    # Add option selection for outcome prediction
    option_index = st.selectbox("Select an option to predict its outcome:", 
                                range(len(st.session_state.current_case.options)),
                                format_func=lambda i: st.session_state.current_case.options[i].description)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Predict Outcome"):
            selected_option = st.session_state.current_case.options[option_index]
            prediction = decision_agent.predict_outcome(st.session_state.current_case, selected_option)
            st.subheader("Predicted Outcome")
            st.write(prediction.prediction)
    
    with col2:
        if st.button("Record Decision"):
            if decision:
                try:
                    updated_case = decision_agent.record_decision(st.session_state.current_case, decision)
                    st.success("Decision recorded successfully!")
                    st.write("Updated Decision Case:")
                    st.json(updated_case.dict())
                    
                    # Clear the current case from session state
                    del st.session_state.current_case
                except Exception as e:
                    st.error(f"Error in recording the decision: {str(e)}")
            else:
                st.warning("Please enter your final decision.")

# Display decision history
if st.button("Show Decision History"):
    history = decision_agent.decision_history
    if history:
        for i, case in enumerate(history, 1):
            st.subheader(f"Decision Case {i}")
            st.write(f"Problem: {case.problem}")
            st.write("Options:")
            for j, option in enumerate(case.options, 1):
                st.write(f"  {j}. {option.description}")
            st.write(f"Final Decision: {case.final_decision}")
            st.write("---")
    else:
        st.info("No decisions have been recorded yet.")