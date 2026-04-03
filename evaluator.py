import os

# We will lazily load Langchain inside the class ONLY if the connection isn't mocked.
# This prevents crashes when users have DNS/offline errors blocking pip installs!

class ThinkingEvaluator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy-key")
        self.chain = None
        
        if self.api_key != "dummy-key" and self.api_key:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import PromptTemplate
                from langchain_core.output_parsers import JsonOutputParser
                from pydantic import BaseModel, Field

                class EvaluationResult(BaseModel):
                    logical_reasoning: float = Field(description="Score from 0 to 10 for logical reasoning depth")
                    concept_quality: float = Field(description="Score from 0 to 10 for concept explanation quality")
                    originality: float = Field(description="Score from 0 to 10 for originality and unique perspective")
                    feedback: str = Field(description="Constructive feedback for the student on how to deepen their argument")

                os.environ["OPENAI_API_KEY"] = self.api_key
                self.llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
                self.parser = JsonOutputParser(pydantic_object=EvaluationResult)

                self.prompt = PromptTemplate(
                    template="You are an expert academic evaluator. Assess the following student answer for critical thinking.\n\n"
                             "{format_instructions}\n\n"
                             "Student Answer:\n{answer}\n\n"
                             "Provide strict, rubric-based scores (0.0 to 10.0). Focus on logic, clarity, and unique insights.",
                    input_variables=["answer"],
                    partial_variables={"format_instructions": self.parser.get_format_instructions()},
                )
                self.chain = self.prompt | self.llm | self.parser
            except ImportError as e:
                print(f"Warning: Could not load Language model components: {e}")
                self.chain = None

    def evaluate(self, text):
        if self.chain is None or self.api_key == "dummy-key":
             # Return mock values if there's no actual API key or if pip install failed
             return {
                 "logical_reasoning": 7.5,
                 "concept_quality": 8.0,
                 "originality": 6.0,
                 "feedback": "This is a demonstration evaluation because internet dependencies were disrupted (LangChain not installed). To use live OpenAI evaluation, connect your internet, run pip install langchain-openai, and provide an API Key!"
             }
        try:
            return self.chain.invoke({"answer": text})
        except Exception as e:
            return {"error": str(e)}
