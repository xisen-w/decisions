import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from agents.basicAgents import LLMAgent
from pydantic import BaseModel, Field
from typing import List, Optional

class ChoiceAnalysis(BaseModel):
    analysis: str = Field(..., description="Detailed analysis of all provided choices")
    decision: str = Field(..., description="The final decision made based on the analysis")

class OpportunityCost(BaseModel):
    choice: str = Field(..., description="The choice being evaluated")
    opportunity_cost: str = Field(..., description="The opportunity cost of making this choice")

class CostEvaluation(BaseModel):
    costs: List[OpportunityCost] = Field(..., description="List of opportunity costs for each choice")

class DecisionOption(BaseModel):
    description: str = Field(..., description="Description of the decision option")

class DecisionCase(BaseModel):
    problem: str = Field(..., description="Description of the problem or situation")
    options: List[DecisionOption] = Field(default_factory=list, description="List of decision options")
    final_decision: Optional[str] = Field(None, description="The final decision made, if any")

class OutcomePrediction(BaseModel):
    option: str = Field(..., description="The option being predicted")
    prediction: str = Field(..., description="Detailed prediction of the outcome")

class DecisionAgent(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.decision_history: List[DecisionCase] = []

    def choose(self, choices_description: str) -> ChoiceAnalysis:
        """
        Analyze the given choices and make a decision.

        :param choices_description: A string describing the choices to be analyzed
        :return: ChoiceAnalysis object containing the analysis and final decision
        """
        user_prompt = f"Please analyze the following choices and make a decision:\n\n{choices_description}"
        system_prompt = """
        You are a decision-making AI assistant. Your task is to:
        1. Carefully analyze all the choices presented.
        2. Provide a detailed analysis of each choice, considering pros and cons.
        3. Make a final decision based on your analysis.
        4. Explain the reasoning behind your decision.
        Be thorough in your analysis and clear in your final decision.
        """

        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=ChoiceAnalysis
        )

        return response

    def evaluate_cost(self, choices_description: str) -> CostEvaluation:
        """
        Evaluate the opportunity costs for each given choice.

        :param choices_description: A string describing the choices to be analyzed
        :return: CostEvaluation object containing a list of opportunity costs for each choice
        """
        user_prompt = f"Please evaluate the opportunity costs for each of the following choices:\n\n{choices_description}"
        system_prompt = """
        You are an AI assistant specializing in economic analysis. Your task is to:
        1. Identify each choice presented in the given description.
        2. For each choice, determine the opportunity cost - what is being given up by making that choice.
        3. Provide a clear and concise description of the opportunity cost for each choice.
        Be thorough in your analysis and consider both tangible and intangible costs.
        """

        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=CostEvaluation
        )

        return response

    def suggest_options(self, situation: str) -> List[str]:
        """
        Suggest a list of options based on the given situation.

        :param situation: A string describing the situation
        :return: List of suggested options
        """
        user_prompt = f"Please suggest a list of options for the following situation:\n\n{situation}"
        system_prompt = """
        You are an AI assistant specializing in brainstorming and option generation. Your task is to:
        1. Analyze the given situation.
        2. Generate a list of 3-5 relevant and diverse options.
        3. Ensure each option is clear, concise, and directly related to the situation.
        Be creative and think outside the box while remaining practical.
        """

        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=List[str]
        )

        return response

    def suggest_automate(self, situation: str) -> ChoiceAnalysis:
        """
        Suggest options and then automatically choose the best one.

        :param situation: A string describing the situation
        :return: ChoiceAnalysis object containing the analysis and final decision
        """
        options = self.suggest_options(situation)
        choices_description = "\n".join(f"{i+1}. {option}" for i, option in enumerate(options))
        return self.choose(choices_description)

    def create_decision_case(self, description: str) -> DecisionCase:
        """
        Create a DecisionCase from a string description.

        :param description: A string describing the problem and potential options
        :return: DecisionCase object
        """
        user_prompt = f"Please analyze the following description and extract the problem and potential options:\n\n{description}"
        system_prompt = """
        You are an AI assistant specializing in problem analysis. Your task is to:
        1. Identify the main problem or situation from the given description.
        2. Extract or generate 3-5 potential options to address the problem.
        3. Format your response as a JSON object with 'problem' and 'options' fields.
        Be concise and clear in your analysis.
        """

        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=DecisionCase
        )

        return response

    def record_decision(self, case: DecisionCase, decision: str) -> DecisionCase:
        """
        Record the final decision for a DecisionCase and store it in history.

        :param case: The DecisionCase to update
        :param decision: The final decision made
        :return: Updated DecisionCase
        """
        case.final_decision = decision
        self.decision_history.append(case)
        return case

    def _choose_structured(self, case: DecisionCase) -> ChoiceAnalysis:
        """
        Analyze the given DecisionCase and make a decision.

        :param case: DecisionCase object containing the problem and options
        :return: ChoiceAnalysis object containing the analysis and final decision
        """
        choices_description = f"Problem: {case.problem}\n\nOptions:\n" + "\n".join(f"{i+1}. {option.description}" for i, option in enumerate(case.options))
        
        user_prompt = f"Please analyze the following problem and options, then make a decision:\n\n{choices_description}"
        system_prompt = """
        You are a decision-making AI assistant. Your task is to:
        1. Carefully analyze the problem and all the options presented.
        2. Provide a detailed analysis of each option, considering pros and cons.
        3. Make a final decision based on your analysis.
        4. Explain the reasoning behind your decision.
        Be thorough in your analysis and clear in your final decision.
        """

        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=ChoiceAnalysis
        )

        return response

    def evaluate_cost_structured(self, case: DecisionCase) -> CostEvaluation:
        """
        Evaluate the opportunity costs for each option in the given DecisionCase.

        :param case: DecisionCase object containing the problem and options
        :return: CostEvaluation object containing a list of opportunity costs for each option
        """
        choices_description = f"Problem: {case.problem}\n\nOptions:\n" + "\n".join(f"{i+1}. {option.description}" for i, option in enumerate(case.options))
        
        user_prompt = f"Please evaluate the opportunity costs for each of the following options:\n\n{choices_description}"
        system_prompt = """
        You are an AI assistant specializing in economic analysis. Your task is to:
        1. Identify each option presented in the given description.
        2. For each option, determine the opportunity cost - what is being given up by making that choice.
        3. Provide a clear and concise description of the opportunity cost for each option.
        Be thorough in your analysis and consider both tangible and intangible costs.
        """

        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=CostEvaluation
        )

        return response

    def evaluate_structured(self, description: str) -> tuple[DecisionCase, ChoiceAnalysis, CostEvaluation]:
        """
        Perform a structured evaluation of a decision case.

        :param description: A string describing the problem and potential options
        :return: A tuple containing the DecisionCase, ChoiceAnalysis, and CostEvaluation
        """
        # Create a DecisionCase from the description
        case = self.create_decision_case(description)

        # Analyze and choose the best option
        choice_analysis = self._choose_structured(case)

        # Evaluate the costs
        cost_evaluation = self.evaluate_cost_structured(case)

        return case, choice_analysis, cost_evaluation

    def predict_outcome(self, case: DecisionCase, option: DecisionOption) -> OutcomePrediction:
        """
        Predict the outcome of choosing a specific option in a given DecisionCase.

        :param case: DecisionCase object containing the problem and options
        :param option: The specific DecisionOption to predict the outcome for
        :return: OutcomePrediction object containing the option and its predicted outcome
        """
        context = f"""
        Problem: {case.problem}

        All options:
        {chr(10).join(f"- {opt.description}" for opt in case.options)}

        Selected option: {option.description}
        """

        user_prompt = f"Please predict the outcome if the following option is chosen:\n\n{context}"
        system_prompt = """
        You are an AI assistant specializing in predicting outcomes based on decisions. Your task is to:
        1. Carefully analyze the given problem and the selected option.
        2. Predict the most likely outcome if this option is chosen.
        3. Provide a vivid, detailed description of the predicted outcome.
        4. Include potential consequences, both positive and negative.
        5. Consider short-term and long-term effects.
        6. Use your imagination to fill in details, but ensure they logically follow from the given information.
        7. Make your prediction specific, engaging, and realistic.

        Your prediction should be comprehensive, engaging, and paint a clear picture of the potential future. 
        Use sensory details, potential dialogues, and specific events to make your prediction come alive. 
        Ensure that your prediction is coherent and makes logical sense given the context of the problem and the chosen option.
        """

        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=OutcomePrediction
        )

        return response

# Example usage in main
if __name__ == "__main__":
    # ... existing code ...

    # Testing DecisionAgent
    decision_agent = DecisionAgent('gpt-4o-mini')
    choices = """
    1. Go to the beach for a relaxing day
    2. Stay home and catch up on work
    3. Visit a museum to learn something new
    """
    
    try:
        result = decision_agent.choose(choices)
        print("\nDecision Agent Test:")
        print(f"Choices:\n{choices}")
        print(f"Analysis:\n{result.analysis}")
        print(f"Decision: {result.decision}")
    except Exception as e:
        print(f"Error in Decision Agent: {str(e)}")

    # Testing evaluate_cost method
    choices = """
    1. Go to the beach for a relaxing day
    2. Stay home and catch up on work
    3. Visit a museum to learn something new
    """
    
    try:
        cost_evaluation = decision_agent.evaluate_cost(choices)
        print("\nCost Evaluation Test:")
        print(f"Choices:\n{choices}")
        for cost in cost_evaluation.costs:
            print(f"Choice: {cost.choice}")
            print(f"Opportunity Cost: {cost.opportunity_cost}\n")
    except Exception as e:
        print(f"Error in Cost Evaluation: {str(e)}")

    # ... rest of the existing code ...