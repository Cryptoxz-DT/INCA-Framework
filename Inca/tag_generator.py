from typing import List, Optional
import openai

class TagGenerator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate_tags(self, text: str, examples: Optional[List[str]] = None) -> List[str]:
        """Generate semantic tags using LLM"""
        prompt = f"""
        Analyze the following text and generate 3-5 semantic tags. 
        Return only comma-separated values.
        
        Text: "{text}"
        Example Format: tag1, tag2, tag3
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip().split(", ")

# if __name__ == "__main__":
#     # Test with your API key
#     tg = TagGenerator(api_key="your_api_key_here")
#     test_text = "How do I transfer money between accounts?"
#     print("Generated tags:", tg.generate_tags(test_text))

