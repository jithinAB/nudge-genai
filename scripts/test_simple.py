#!/usr/bin/env python3
"""Simple test of the LM Studio API with a shorter prompt"""

import asyncio
import aiohttp
import json

async def test_simple():
    url = "http://localhost:1234/v1/chat/completions"

    # Simple test prompt
    prompt = """Generate a simple JSON with 2 conversations for a person from Kerala who believes in traditional values.

Return as JSON:
{
  "Conversations": {
    "career_advice": [
      {"role": "person", "message": "..."},
      {"role": "AI", "message": "..."}
    ]
  }
}"""

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    async with aiohttp.ClientSession() as session:
        try:
            print("Sending request...")
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    print("Response received:")
                    print(content)

                    # Try to parse JSON
                    try:
                        # Remove markdown code blocks if present
                        if "```json" in content:
                            start = content.find("```json") + 7
                            end = content.find("```", start)
                            content = content[start:end].strip()
                        elif "```" in content:
                            start = content.find("```") + 3
                            end = content.find("```", start)
                            content = content[start:end].strip()

                        parsed = json.loads(content)
                        print("\nParsed successfully!")
                        print(json.dumps(parsed, indent=2))
                    except Exception as e:
                        print(f"\nFailed to parse JSON: {e}")
                else:
                    print(f"Error: {response.status}")
        except asyncio.TimeoutError:
            print("Request timed out")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple())