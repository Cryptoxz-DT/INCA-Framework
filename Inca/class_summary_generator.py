from typing import List
import openai

class ClassSummaryGenerator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.summaries = {}
        
    def generate_summary(self, class_id: str, examples: List[str]) -> str:
        """Generate class summary using LLM"""
        prompt = f"""
        Create a concise summary for class '{class_id}' using these examples:
        {examples}
        
        Keep it under 2 sentences. Focus on key patterns.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        
        summary = response.choices[0].message.content.strip()
        self.summaries[class_id] = summary
        return summary

# if __name__ == "__main__":
#     # Test with your API key
#     csg = ClassSummaryGenerator(api_key="your_api_key_here")
#     examples = ["Check balance", "View transactions"]
#     print("Summary:", csg.generate_summary("account_management", examples))