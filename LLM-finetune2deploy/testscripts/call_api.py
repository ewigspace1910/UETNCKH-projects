import requests

# Define the URL of your LLM host
LLM_URL = "http://localhost:8000/v1/chat/completions"

# Define the payload
payload = {
    "model": "llama3-8b-ieltsmarking-hf",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      },
      {
        "role":"assistant",
        "content":"Hi! I am a IETLS WRITING MARKER"
      },
      {
        "role":"user",
        "content":"Please rate this essay in term of 'Coherence & Cohesion', 'Lexical Resource', 'Grammatical Range & Accuracy' on a scale of 9. The essay is : ```Urbanization is a big change that is happening in many places. More people are coming to cities to find jobs and a better life. This can be good, but it also has problems. I think the bad things about cities are more than the good things. One good thing is jobs. Cities have more business, so more jobs. People can find work easier there. Also, cities have more services like hospitals and schools. This is good for people. But cities also have bad things. One big problem is that cities are crowded. There are too many people. This makes houses more expensive. Poor people can't find a place to live. Also, cities make more dirty. All the cars and factories make bad air. This makes people sick. Another problem is crime. Cities have more people, so more crime. It is not safe to live in cities. I think city governments need to fix these problems. They need to make more cheap houses. They need to clean the air. They need to make cities safe. If cities can fix these problems, they will be better. But for now, I think the problems are more big than the good things.```"
      }
    ],
    #"prompt": "John buys 10 packs of magic cards. Each pack has 20 cards and 1/4 of those cards are uncommon. How many uncommon cards did he get?",
    "max_tokens": 400,
    "n": 3,
}

# Set the headers
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# Function to send a POST request to the LLM host
def get_llm_response():
    try:
        response = requests.post(LLM_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the response in JSON format
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Test the function
if __name__ == "__main__":
    result = get_llm_response()
    print("LLM Response:")
    print(result)