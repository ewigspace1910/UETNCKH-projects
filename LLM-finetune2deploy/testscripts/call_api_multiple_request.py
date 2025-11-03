import requests
from concurrent.futures import ThreadPoolExecutor
import time

# Define the URL of your LLM host
LLM_URL = "http://localhost:8000/v1/chat/completions"
#LLM_URL = "http://34.142.198.21:8000//v1/chat/completions"
MODELNAME= "meta/llama-3.1-8b-instruct"
MODELNAME= "ducanh"
import random 
questions = [
"what is 1+1",
"what is the biggest country",
"""Given the question as: ``` <strong>TASK 1</strong><br /> <br /> You should spend about 20 minutes on this task.<br />   <table border="3" cellpadding="5" cellspacing="1" style="width:500px;"> <tbody> <tr> <td style="text-align: justify;">The tables below show the sales of organic fruits and vegetables in four European countries in 2015 and 2020.<br /> Summarize the information by selecting and reporting the main features, and make comparisons where relevant.</td> </tr> </tbody> </table> <br /> Write at least 150 words.<br /> <br /> <br /> <strong>Sales of Organic Fruits and Vegetables (2015 & 2020)</strong><br />   <table border="3" cellpadding="5" cellspacing="1" style="width:500px;"> <tbody> <tr> <td><strong>Fruits</strong></td> <td><strong>2015 (millions of euros)</strong></td> <td><strong>2020 (millions of euros)</strong></td> </tr> <tr> <td>Germany</td> <td>4.5</td> <td>12</td> </tr> <tr> <td>France</td> <td>1.8</td> <td>8.5</td> </tr> <tr> <td>Netherlands</td> <td>2.0</td> <td>6.3</td> </tr> <tr> <td>Spain</td> <td>3.2</td> <td>4.1</td> </tr> </tbody> </table>   <table border="3" cellpadding="5" cellspacing="1" style="width:500px;"> <tbody> <tr> <td><strong>Vegetables</strong></td> <td><strong>2015 (millions of euros)</strong></td> <td><strong>2020 (millions of euros)</strong></td> </tr> <tr> <td>Germany</td> <td>3.5</td> <td>10</td> </tr> <tr> <td>France</td> <td>1.6</td> <td>3.2</td> </tr> <tr> <td>Netherlands</td> <td>2.8</td> <td>5.5</td> </tr> <tr> <td>Spain</td> <td>1.3</td> <td>7.5</td> </tr> </tbody> </table> <br /> Note: Organic products are those grown without synthetic pesticides, chemical fertilizers, or genetically modified organisms.<br />   
Chart description: You can assume a chart with information related to the charts ``` .
Please rate this essay in term of 'Coherence & Cohesion', 'Lexical Resource', 'Grammatical Range & Accuracy' on a scale of 9. 
The essay is : 
```The tables show how much money was made from selling organic fruits and vegetables in some places. It says what happened in 2015 and 2020. Germany did good with fruits. France also. Netherlands too. Spain not so much. Vegetables, Germany again. France went up a little. Netherlands, yeah. Spain, they did better. Its about money. Organic is when they don't use bad stuff for the plants. So, its more healthy maybe. I like fruits and vegetables. They are good for you. Money is important to.```""",
]

rubric = """
Rubric: Task Response (Max Score: 9)
Score Type: fixed
Breakdown:
  - Score: 9
    Description: - The prompt is appropriately addressed and explored in depth.
  - Score: 8
    Description: - The prompt is appropriately and sufficiently addressed.
- A clear and well-developed position is presented in response to the question/s.
- Ideas are relevant, well extended and supported.
- There may be occasional omissions or lapses in content.
  - Score: 7
    Description: - The main parts of the prompt are appropriately addressed.
- A clear and developed position is presented.
- Main ideas are extended and supported but there may be a tendency to over-generalise or there may be a lack of focus and precision in supporting ideas/material.
  - Score: 6
    Description: - The main parts of the prompt are addressed (though some may be more fully covered than others). An appropriate format is used.
- A position is presented that is directly relevant to the prompt, although the conclusions drawn may be unclear, unjustified or repetitive.
- Main ideas are relevant, but some may be insufficiently developed or may lack clarity, while some supporting arguments and evidence may be less relevant or inadequate.
  - Score: 5
    Description: - The main parts of the prompt are incompletely addressed. The format may be inappropriate in places.
- The writer expresses a position, but the development is not always clear.
- Some main ideas are put forward, but they are limited and are not sufficiently developed and/or there may be irrelevant detail. 
- There may be some repetition.
  - Score: 4
    Description: - The prompt is tackled in a minimal way, or the answer is tangential, possibly due to some misunderstanding of 
the prompt. The format may be inappropriate.
- A position is discernible, but the reader has to read carefully to find it.
- Main ideas are difficult to identify and such ideas that are identifiable may lack relevance, clarity and/or 
support.
- Large parts of the response may be repetitive.
  - Score: 3
    Description: - No part of the prompt is adequately addressed, or the prompt has been misunderstood.
- No relevant position can be identified, and/or there is little direct response to the question/s.
- There are few ideas, and these may be irrelevant or insufficiently developed.
  - Score: 2
    Description: - The content is barely related to the prompt. 
- No position can be identified.
- There may be glimpses of one or two ideas without 
development.
  - Score: 1
    Description: - Responses of 20 words or fewer are rated at Band 1.
- The content is wholly unrelated to the prompt.
- Any copied rubric must be discounted.
  - Score: 0
    Description: - Should only be used where a candidate did not attend or attempt the question in anyway, used a language other than English throughout, or where there is proof that a candidate's answer has been totally memorised.

===============================================

Rubric: Coherence & Cohesion (Max Score: 9)
Score Type: fixed
Breakdown:
  - Score: 9
    Description: - The message can be followed effortlessly.
- Cohesion is used in such a way that it very rarely attracts attention.
- Any lapses in coherence or cohesion are minimal.
- Paragraphing is skillfully managed
  - Score: 8
    Description: - The message can be followed with ease.
- Information and ideas are logically sequenced, and cohesion is well managed.
- Occasional lapses in coherence and cohesion may occur. 
- Paragraphing is used sufficiently and appropriately.
  - Score: 7
    Description: - Information and ideas are logically organised, and there is a clear progression throughout the response. (A few lapses may occur, but these are minor.)
- A range of cohesive devices including reference and substitution is used flexibly but with some inaccuracies or some over/under use.
- Paragraphing is generally used effectively to support overall coherence, and the sequencing of ideas within a paragraph is generally logical.
  - Score: 6
    Description: - Information and ideas are generally arranged coherently and there is a clear overall progression.
- Cohesive devices are used to some good effect but cohesion within and/or between sentences may be faulty or mechanical due to misuse, overuse or omission.
- The use of reference and substitution may lack flexibility or clarity and result in some repetition or error.
- Paragraphing may not always be logical and/or the central topic may not always be clear.
  - Score: 5
    Description: - Organisation is evident but is not wholly logical and there may be a lack of overall progression. 
- Nevertheless, there is a sense of underlying coherence to the response.
- The relationship of ideas can be followed but the sentences are not fluently linked to each other.
- There may be limited/overuse of cohesive devices with some inaccuracy.
- The writing may be repetitive due to inadequate and/or inaccurate use of reference and substitution.
- Paragraphing may be inadequate or missing.
  - Score: 4
    Description: - Information and ideas are evident but not arranged coherently and there is no clear progression within the response.
- Relationships between ideas can be unclear and/or inadequately marked. There is some use of basic cohesive
devices, which may be inaccurate or repetitive.
- There is inaccurate use or a lack of substitution or referencing.
- There may be no paragraphing and/or no clear main topic within paragraphs.
  - Score: 3
    Description: - There is no apparent logical organisation. Ideas are discernible but difficult to relate to each other.
- There is minimal use of sequencers or cohesive devices. 
- Those used do not necessarily indicate a logical relationship 
between ideas.
- There is difficulty in identifying referencing. 
- Any attempts at paragraphing are unhelpful.
  - Score: 2
    Description: - There is little relevant message, or the entire response may be off-topic.
- There is little evidence of control of organisational features.
  - Score: 1
    Description: - Responses of 20 words or fewer are rated at Band 1. 
- The writing fails to communicate any message and appears to be by a virtual non-writer.
  - Score: 0
    Description: - Should only be used where a candidate did not attend or attempt the question in anyway, used a language other than English throughout, or where there is proof that a candidate's answer has been totally memorised.

===============================================

Rubric: Lexical Resource (Max Score: 9)
Score Type: fixed
Breakdown:
  - Score: 9
    Description: - Full flexibility and precise use are widely evident.
- A wide range of vocabulary is used accurately and appropriately with very natural and sophisticated control of lexical features.
- Minor errors in spelling and word formation are extremely rare and have minimal impact on communication.
  - Score: 8
    Description: - A wide resource is fluently and flexibly used to convey precise meanings.
- There is skillful use of uncommon and/or idiomatic items when appropriate, despite occasional inaccuracies in word choice and collocation.
- Occasional errors in spelling and/or word formation may occur, but have minimal impact on communication.
  - Score: 7
    Description: - The resource is sufficient to allow some flexibility and precision.
- There is some ability to use less common and/or idiomatic items.
- An awareness of style and collocation is evident, though inappropriacies occur.
- There are only a few errors in spelling and/or word formation and they do not detract from overall clarity.
  - Score: 6
    Description: - The resource is generally adequate and appropriate for the task.
- The meaning is generally clear in spite of a rather restricted range or a lack of precision in word choice.
- If the writer is a risk-taker, there will be a wider range of vocabulary used but higher degrees of inaccuracy or inappropriacy.
- There are some errors in spelling and/or word formation, but these do not impede communication.
  - Score: 5
    Description: - The resource is limited but minimally adequate for the task.
- Simple vocabulary may be used accurately but the range does not permit much variation in expression.
- There may be frequent lapses in the appropriacy of word choice and a lack of flexibility is apparent in frequent simplifications and/or repetitions.
- Errors in spelling and/or word formation may be noticeable and may cause some difficulty for the reader.
  - Score: 4
    Description: - The resource is limited and inadequate for or unrelated to the task. Vocabulary is basic and may be used repetitively.
- There may be inappropriate use of lexical chunks (e.g. memorised phrases, formulaic language and/or language from the input material).
- Inappropriate word choice and/or errors in word formation and/or in spelling may impede meaning.
  - Score: 3
    Description: - The resource is inadequate (which may be due to the response being significantly underlength). Possible over-dependence on input material or memorised language.
- Control of word choice and/or spelling is very limited, and errors predominate. These errors may severely impede meaning.
  - Score: 2
    Description: - The resource is extremely limited with few recognisable strings, apart from memorised phrases.
- There is no apparent control of word formation and/or spelling.
  - Score: 1
    Description: - Responses of 20 words or fewer are rated at Band 1. 
- No resource is apparent, except for a few isolated words
  - Score: 0
    Description: - Should only be used where a candidate did not attend or attempt the question in anyway, used a language other than English throughout, or where there is proof that a candidate's answer has been totally memorised.

===============================================

Rubric: Grammatical Range & Accuracy (Max Score: 9)
Score Type: fixed
Breakdown:
  - Score: 9
    Description: - A wide range of structures is used with full flexibility and control.
- Punctuation and grammar are used appropriately throughout.
- Minor errors are extremely rare and have minimal impact on communication.
  - Score: 8
    Description: - A wide range of structures is flexibly and accurately used.
- The majority of sentences are error-free, and punctuation is well managed.
- Occasional, non-systematic errors and inappropriacies occur, but have minimal impact on communication.
  - Score: 7
    Description: - A variety of complex structures is used with some flexibility and accuracy.
- Grammar and punctuation are enerally well controlled, and error-free sentences are frequent.
- A few errors in grammar may persist, but these do not impede communication.
  - Score: 6
    Description: - A mix of simple and complex sentence forms is used but flexibility is limited.
- Examples of more complex structures are not marked by the same level of accuracy as in simple structures.
- Errors in grammar and punctuation occur, but rarely impede communication.
  - Score: 5
    Description: - The range of structures is limited and rather repetitive.
- Although complex sentences are attempted, they tend to be faulty, and the greatest accuracy is achieved on simple sentences.
- Grammatical errors may be frequent and cause some difficulty for the reader.
- Punctuation may be faulty.
  - Score: 4
    Description: - A very limited range of structures is used.
- Subordinate clauses are rare and simple sentences predominate.
- Some structures are produced accurately but grammatical errors are frequent and may impede meaning.
- Punctuation is often faulty or inadequate.
  - Score: 3
    Description: - Sentence forms are attempted, but errors in grammar and punctuation predominate (except in memorised
phrases or those taken from the input material). This prevents most meaning from coming through.
- Length may be insufficient to provide evidence of control of sentence forms.
  - Score: 2
    Description: - There is little or no evidence of sentence forms (except in memorised phrases)
  - Score: 1
    Description: - Responses of 20 words or fewer are rated at Band 1. 
- No rateable language is evident.
  - Score: 0
    Description: - Should only be used where a candidate did not attend or attempt the question in anyway, used a language other than English throughout, or where there is proof that a candidate's answer has been totally memorised.

===============================================


"""
# Define the payload
def generate_payload():
    payload = {
        "model": MODELNAME,
        "messages": [
          {
            "role":"assistant",
            "content":"Hi! I am a IETLS WRITING MARKER"
          },
          {
            "role":"user",
            "content":"What is the rubric?"
          },
          {
            "role":"assistant",
            "content":rubric
          },
          {
            "role":"user",
            "content":random.choice(questions)
          }
        ],
        "max_tokens": 400
    }
    payload = {
    "messages": [

    {
      "content": "What is 12+12*3",
      "role": "developer",
      "name": "Nana",
      "max_tokens": 400
    }
  ],
  "model": MODELNAME,
}
    return payload

# Define headers
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# Function to send a single request
def send_request(index):
    try:
        start_time = time.time()  # Start time for this request
        response = requests.post(LLM_URL, json=generate_payload(), headers=headers)
        response.raise_for_status()
        elapsed_time = time.time() - start_time  # Time taken for this request
        return f"Response {index}: {response.json()} (Time: {elapsed_time:.2f} seconds)"
    except requests.exceptions.RequestException as e:
        return f"Request {index} failed: {str(e)}"

# Function to send multiple requests concurrently and measure total time
def send_multiple_requests(total_requests):
    start_time = time.time()  # Start total timer
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        for future in futures:
            print(future.result())
    total_time = time.time() - start_time  # Total time for all requests
    print(f"\nTotal Time for {total_requests} Requests: {total_time:.2f} seconds")
    print(f"Average Time per Request: {total_time / total_requests:.2f} seconds")

# Trigger 100 simultaneous requests
if __name__ == "__main__":
    send_multiple_requests(20)