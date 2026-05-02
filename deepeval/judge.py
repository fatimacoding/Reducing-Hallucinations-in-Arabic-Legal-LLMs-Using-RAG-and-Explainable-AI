"""
Simple Judge
Uses GPT-4o to evaluate answers
"""
from openai import OpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    HallucinationMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
import os

load_dotenv()


class SimpleJudge(DeepEvalBaseLLM):
    """GPT-4o Judge"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
    
    def load_model(self):
        return self.client
    
    def generate(self, prompt: str) -> str:#spetial prompt for the judge, differ from the test LLM's prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return self.model


def evaluate_answer(question, answer, context):
    """
    Evaluate an answer using 3 metrics
    
    Args:
        question: The question
        answer: Generated answer
        context: The context
    
    Returns:
        dict with scores
    """
    print(f"\n    Evaluating with GPT-4o...")
    
    # Create judge
    judge = SimpleJudge()
    
    # Create test case
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=[context],
        retrieval_context=[context]
    )
    
    # Initialize metrics
    metrics = [
        HallucinationMetric(model=judge),
        FaithfulnessMetric(model=judge),
        AnswerRelevancyMetric(model=judge)
    ]
    
    # Evaluate with better error handling
    results = {}
    for metric in metrics:
        name = metric.__class__.__name__
        try:
            metric.measure(test_case)
            results[name] = {
                'score': round(metric.score, 3),
                'passed': metric.is_successful()
            }
            print(f"{name}: {results[name]['score']}")
        except Exception as e:
            # Handle JSON errors gracefully
            if "invalid JSON" in str(e):
                print(f"{name}: JSON error, retrying...")
                try:
                    # Retry once
                    import time
                    time.sleep(5)
                    metric.measure(test_case)
                    results[name] = {
                        'score': round(metric.score, 3),
                        'passed': metric.is_successful()
                    }
                    print(f"{name}: {results[name]['score']} (retry success)")
                except:
                    # If retry fails, use default score
                    results[name] = {
                        'score': None,
                        'passed': False,
                        'error': 'JSON_ERROR'
                    }
                    print(f"{name}: JSON error")
            else:
                results[name] = {
                    'score': 0.0,
                    'passed': False,
                    'error': str(e)[:100]
                }
                print(f"{name}: {str(e)[:50]}...")
    
    return results


if __name__ == "__main__":
    # Test
    print("="*60)
    print("TESTING JUDGE")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print(" No OpenAI API key!")
    else:
        print(" API key found")