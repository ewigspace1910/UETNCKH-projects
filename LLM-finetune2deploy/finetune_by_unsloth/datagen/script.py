import pickle
from google import genai
import asyncio
import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import AsyncOpenAI
import math
# Function to save list of dictionaries into a JSONL file
import json
from google import genai
import openai
BASE_URL = "https://asia-southeast1-aiplatform.googleapis.com/v1/projects/heyhi-ai-model1/locations/asia-southeast1/endpoints/8534615962983333888"

def call_gemini(question):
  client = genai.Client(api_key="AIzaSyDjUqffrSF3fhauyv1TZu5OMduXG053y1M") #, base_url="https://asia-southeast1-aiplatform.googleapis.com/v1/projects/heyhi-ai-model1/locations/asia-southeast1/endpoints/8534615962983333888")
  response = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=question)

  return response.text


load_dotenv()

def save_to_jsonl(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for item in data:
                json.dump(item, file)
                file.write('\n')  # Write each dictionary as a separate line
        print(f"Data saved successfully into {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
def read_json(file_path):
    try:
        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("File loaded successfully!")
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#@title IELTS Rubric 1
rubric_1 = {
    "rubric_table": [
        # {
        #     "name": "Task Achievement",
        #     "max_score": 9,
        #     "score_type": "fixed",
        #     "breakdown": [
        #         {
        #             "id": 2963,
        #             "description": "- All the requirements of the task are fully and appropriately satisfied.",
        #             "score": 9
        #         },
        #         {
        #             "id": 2964,
        #             "description": "- The response covers all the requirements of the task appropriately, relevantly and sufficiently.\n- (Academic) Key features are skilfully selected, and clearly presented, highlighted and illustrated.\n- (General Training) All bullet points are clearly presented, and appropriately illustrated or extended.\n- There may be occasional omissions or lapses in content.",
        #             "score": 8
        #         },
        #         {
        #             "id": 2965,
        #             "description": "- The response covers the requirements of the task.\n- The content is relevant and accurate – there may be a few omissions or lapses. \n- The format is appropriate.\n- (Academic) Key features which are selected are covered and clearly highlighted but could be more fully or more appropriately illustrated or extended.\n- (Academic) It presents a clear overview, the data are appropriately categorised, and main trends or differences are identified.\n- (General Training) All bullet points are covered and clearly highlighted but could be more fully or more appropriately illustrated or extended. It presents a clear purpose. The tone is consistent and appropriate to the task. Any lapses are minimal.",
        #             "score": 7
        #         },
        #         {
        #             "id": 2966,
        #             "description": "- The response focuses on the requirements of the task and an appropriate format is used.\n- (Academic) Key features which are selected are covered and adequately highlighted. A relevant overview is attempted. Information is appropriately selected and supported using figures/data.\n- (General Training) All bullet points are covered and adequately highlighted. The purpose is generally clear. There may be minor inconsistencies in tone.\n- Some irrelevant, inappropriate or inaccurate information may occur in areas of detail or when illustrating or extending the main points.\n- Some details may be missing (or excessive) and further extension or illustration may be needed.",
        #             "score": 6
        #         },
        #         {
        #             "id": 2967,
        #             "description": "- The response generally addresses the requirements of the task. The format may be inappropriate in places.\n- (Academic) Key features which are selected are not adequately covered.  The recounting of detail is mainly mechanical. There may be no data to support the description.\n- (General Training) All bullet points are presented but one or more may not be adequately covered. The purpose may be unclear at times. The tone may be variable and sometimes inappropriate.\n- There may be a tendency to focus on details (without referring to the bigger picture).\n- The inclusion of irrelevant, inappropriate or inaccurate material in key areas detracts from the task achievement.\nThere is limited detail when extending and illustrating the main points.",
        #             "score": 5
        #         },
        #         {
        #             "id": 2968,
        #             "description": "- The response is an attempt to address the task.\n- (Academic) Few key features have been selected.\n- (General Training) Not all bullet points are presented.\n- (General Training) The purpose of the letter is not clearly explained and may be confused. The tone may be inappropriate.\n- The format may be inappropriate.\n- Key features/bullet points which are presented may be irrelevant, repetitive, inaccurate or inappropriate.",
        #             "score": 4
        #         },
        #         {
        #             "id": 2969,
        #             "description": "- The response does not address the requirements of the task (possibly because of misunderstanding of the data/diagram/situation).\n- Key features/bullet points which are presented may be largely irrelevant.\n- Limited information is presented, and this may be used repetitively.",
        #             "score": 3
        #         },
        #         {
        #             "id": 2970,
        #             "description": "- The content barely relates to the task.",
        #             "score": 2
        #         },
        #         {
        #             "id": 2971,
        #             "description": "- Responses of 20 words or fewer are rated at Band 1.\n- The content is wholly unrelated to the task.\n- Any copied rubric must be discounted.",
        #             "score": 1
        #         },
        #         {
        #             "id": 2972,
        #             "description": "- Should only be used where a candidate did not attend or attempt the question in any way, used a language other than English throughout, or where there is proof that a candidate’s answer has been totally memorised.",
        #             "score": 0
        #         }
        #     ]
        # },
        # {
        #     "name": "Coherence & Cohesion",
        #     "max_score": 9,
        #     "score_type": "fixed",
        #     "breakdown": [
        #         {
        #             "id": 2973,
        #             "description": "- The message can be followed effortlessly.\n- Cohesion is used in such a way that it very rarely attracts attention.\n- Any lapses in coherence or cohesion are minimal.\n- Paragraphing is skillfully managed.",
        #             "score": 9
        #         },
        #         {
        #             "id": 2974,
        #             "description": "- The message can be followed with ease.\n- Information and ideas are logically sequenced, and cohesion is well managed.\n- Occasional lapses in coherence or cohesion may occur.\n- Paragraphing is used sufficiently and appropriately.",
        #             "score": 8
        #         },
        #         {
        #             "id": 2975,
        #             "description": "- Information and ideas are logically organised and there is a clear progression throughout the response. A few lapses may occur.\n- A range of cohesive devices including reference and substitution is used flexibly but with some inaccuracies or some over/under use.",
        #             "score": 7
        #         },
        #         {
        #             "id": 2976,
        #             "description": "- Information and ideas are generally arranged coherently and there is a clear overall progression.\n- Cohesive devices are used to some good effect but cohesion within and/or between sentences may be faulty or mechanical due to misuse, overuse or omission.\n- The use of reference and substitution may lack flexibility or clarity and result in some repetition or error.",
        #             "score": 6
        #         },
        #         {
        #             "id": 2977,
        #             "description": "- Organisation is evident but is not wholly logical and there may be a lack of overall progression. Nevertheless, there is a sense of underlying coherence to the response.\n- The relationship of ideas can be followed but the sentences are not fluently linked to each other.\n- There may be limited/overuse of cohesive devices with some inaccuracy.\n- The writing may be repetitive due to inadequate and/or inaccurate use of reference and substitution.",
        #             "score": 5
        #         },
        #         {
        #             "id": 2978,
        #             "description": "- Information and ideas are evident but not arranged coherently, and there is no clear progression within the response.\n- Relationships between ideas can be unclear and/or inadequately marked. There is some \nuse of basic cohesive devices, which may be inaccurate or repetitive.\n- There is inaccurate use or a lack of substitution or referencing.",
        #             "score": 4
        #         },
        #         {
        #             "id": 2979,
        #             "description": "- There is no apparent logical organisation. Ideas are discernible but difficult to relate to each other.\n- Minimal use of sequencers or cohesive devices. Those used do not necessarily indicate a logical relationship between ideas.\n- There is difficulty in identifying referencing.",
        #             "score": 3
        #         },
        #         {
        #             "id": 2980,
        #             "description": "- There is little relevant message, or the entire response may be off-topic.\n- There is little evidence of control of organisational features.",
        #             "score": 2
        #         },
        #         {
        #             "id": 2981,
        #             "description": "- Responses of 20 words or fewer are rated at Band 1. \n- The writing fails to communicate any message and appears to be by a virtual non-writer.",
        #             "score": 1
        #         },
        #         {
        #             "id": 2982,
        #             "description": "- Should only be used where a candidate did not attend or attempt the question in any way, used a language other than English throughout, or where there is proof that a candidate’s answer has been totally memorised.",
        #             "score": 0
        #         }
        #     ]
        # },
        {
            "name": "Lexical Resouce",
            "max_score": 9,
            "score_type": "fixed",
            "breakdown": [
                {
                    "id": 2983,
                    "description": "- Full flexibility and precise use are evident within the scope of the task.\n- A wide range of vocabulary is used accurately and appropriately with very natural and sophisticated control of lexical features.\nMinor errors in spelling and word formation are extremely rare and have minimal impact on communication.",
                    "score": 9
                },
                {
                    "id": 2984,
                    "description": "- A wide resource is fluently and flexibly used to convey precise meanings within the scope of the task.\n- There is skilful use of uncommon and/or idiomatic items when appropriate, despite occasional inaccuracies in word choice and collocation.\n- Occasional errors in spelling and/or word formation may occur, but have minimal impact on communication.",
                    "score": 8
                },
                {
                    "id": 2985,
                    "description": "- The resource is sufficient to allow some flexibility and precision.\n- There is some ability to use less common and/or idiomatic items.\n -An awareness of style and collocation is evident, though inappropriacies occur.\n- There are only a few errors in spelling and/or word formation, and they do not detract from overall clarity.",
                    "score": 7
                },
                {
                    "id": 2986,
                    "description": "- The resource is generally adequate and appropriate for the task.\n- The meaning is generally clear in spite of a rather restricted range or a lack of precision in word choice.\n- If the writer is a risk-taker, there will be a wider range of vocabulary used but higher degrees of inaccuracy or inappropriacy.\n- There are some errors in spelling and/or word formation, but these do not impede communication.",
                    "score": 6
                },
                {
                    "id": 2987,
                    "description": "- The resource is limited but minimally adequate for the task.\n- Simple vocabulary may be used accurately but the range does not permit much variation in expression.\n- There may be frequent lapses in the appropriacy of word choice, and a lack of flexibility is apparent in frequent simplifications and/or repetitions.\n- Errors in spelling and/or word formation may be noticeable and may cause some difficulty for the reader.",
                    "score": 5
                },
                {
                    "id": 2988,
                    "description": "- The resource is limited and inadequate for or unrelated to the task. Vocabulary is basic and may be used repetitively.\n- There may be inappropriate use of lexical chunks (e.g. memorised phrases, formulaic language and/or language from the input material).\n- Inappropriate word choice and/or errors in word formation and/or in spelling may impede meaning.",
                    "score": 4
                },
                {
                    "id": 2989,
                    "description": "- The resource is inadequate (which may be due to the response being significantly underlength).\n- Possible over-dependence on input material or memorised language.\n- Control of word choice and/or spelling is very limited, and errors predominate. These errors may severely impede meaning.",
                    "score": 3
                },
                {
                    "id": 2990,
                    "description": "- The resource is extremely limited with few recognisable strings, apart from memorised phrases.\n- There is no apparent control of word formation and/or spelling.",
                    "score": 2
                },
                {
                    "id": 2991,
                    "description": "- Responses of 20 words or fewer are rated at Band 1. \n- No resource is apparent, except for a few isolated words.",
                    "score": 1
                },
                {
                    "id": 2992,
                    "description": "- Should only be used where a candidate did not attend or attempt the question in any way, used a language other than English throughout, or where there is proof that a candidate’s answer has been totally memorised.",
                    "score": 0
                }
            ]
        },
        
        # {
        #     "name": "Grammatical Range & Accuracy",
        #     "max_score": 9,
        #     "score_type": "fixed",
        #     "breakdown": [
        #         {
        #             "id": 2993,
        #             "description": "- A wide range of structures within the scope of the task is used with full flexibility and control.\n- Punctuation and grammar are used appropriately throughout.\n- Minor errors are extremely rare and have minimal impact on communication",
        #             "score": 9
        #         },
        #         {
        #             "id": 2994,
        #             "description": "- A wide range of structures within the scope of the task is flexibly and accurately used.\n- The majority of sentences are error-free, and punctuation is well managed.\n- Occasional, non-systematic errors and inappropriacies occur, but have minimal impact on communication.",
        #             "score": 8
        #         },
        #         {
        #             "id": 2995,
        #             "description": "- A variety of complex structures is used with some flexibility and accuracy.\n- Grammar and punctuation are generally well controlled, and error-free sentences are frequent.\n -A few errors in grammar may persist, but these do not impede communication.",
        #             "score": 7
        #         },
        #         {
        #             "id": 2996,
        #             "description": "- A mix of simple and complex sentence forms is used but flexibility is limited.\n- Examples of more complex structures are not marked by the same level of accuracy as in simple structures.\n- Errors in grammar and punctuation occur, but rarely impede communication.",
        #             "score": 6
        #         },
        #         {
        #             "id": 2997,
        #             "description": "- The range of structures is limited and rather repetitive.\n- Although complex sentences are attempted, they tend to be faulty, and the greatest accuracy is achieved on simple sentences.\n- Grammatical errors may be frequent and cause some difficulty for the reader.\n- Punctuation may be faulty.",
        #             "score": 5
        #         },
        #         {
        #             "id": 2998,
        #             "description": "- A very limited range of structures is used.\n- Subordinate clauses are rare and simple sentences predominate.\n- Some structures are produced accurately but grammatical errors are frequent and may impede meaning.\n- Punctuation is often faulty or inadequate.",
        #             "score": 4
        #         },
        #         {
        #             "id": 2999,
        #             "description": "- Sentence forms are attempted, but errors in grammar and punctuation predominate (except in memorised\nphrases or those taken from the input material). This prevents most meaning from coming through.\n- Length may be insufficient to provide evidence of control of sentence forms.",
        #             "score": 3
        #         },
        #         {
        #             "id": 3000,
        #             "description": "- There is little or no evidence of sentence forms (except in memorised phrases).",
        #             "score": 2
        #         },
        #         {
        #             "id": 3001,
        #             "description": "- Responses of 20 words or fewer are rated at Band 1. \n- No rateable language is evident.",
        #             "score": 1
        #         },
        #         {
        #             "id": 3002,
        #             "description": "- Should only be used where a candidate did not attend or attempt the question in any way, used a language other than English throughout, or where there is proof that a candidate’s answer has been totally memorised.",
        #             "score": 0
        #         }
        #     ]
        # }
    ],
    "question_statement": "<strong>TASK 1</strong><br />\r\n<br />\r\nYou should allocate approximately 20 minutes to this task.<br />\r\n \r\n<table border=\"1\" cellpadding=\"5\" cellspacing=\"1\" style=\"width:500px;\">\r\n\t<tbody>\r\n\t\t<tr>\r\n\t\t\t<td style=\"text-align: left;\">The hypothetical bar chart illustrates the educational attainment levels in the United States for the years 2015, 2019, and 2023. Analyze the chart to summarize key trends and compare the shifts in educational levels over these years.</td>\r\n\t\t</tr>\r\n\t</tbody>\r\n</table>\r\n<br />\r\nPlease write a minimum of 150 words.<br />\r\n<br />\r\n  <img style=\"width:100%\" src=\"https://static-contents-smartjen.s3.ap-southeast-1.amazonaws.com/img/questionImage/202408100014-66bc0fae32179.jpeg\"/> ",
    "student_composition": "<p>The bar chart illustrates the educational attainment levels in the United States from 2015 to 2023. In 2015, the highest percentage of the population had attained a high school diploma, while the bachelor's, master's, and doctorate degrees had lower percentages, respectively. By 2019, there was a noticeable increase in bachelor's and master's degree holders, with a slight decline in high school diploma holders. The doctorate level remained relatively stable. By 2023, the trend of increased higher education continued, with the number of people holding a bachelor's degree increasing further, and those with master's degrees also seeing a rise. The percentage of high school diploma holders dropped even more. This chart shows a clear trend towards higher educational attainment in the U.S., with more people pursuing advanced degrees over time.</p>",
    "model_composition": "",
    "student_class": ""
}

rubric_1_statement = """

"""
def convert_rubrics_to_text(rubrics):
    output = []
    for rubric in rubrics:
        output.append(f"Rubric: {rubric['name']} (Max Score: {rubric['max_score']})")
        output.append(f"Score Type: {rubric['score_type']}")
        output.append("Breakdown:")
        for item in rubric['breakdown']:
            output.append(f"  - Score: {item['score']}\n    Description: {item['description']}")
        output.append("\n===============================================\n")  # Add a blank line between rubrics
    return "\n".join(output)



if  __name__ == "__main__":
    # Generate structured text
    structured_text_rubric1 = convert_rubrics_to_text(rubric_1.get("rubric_table"))
    question_file = "/home/ducanh/nvidia-llm-pipeline/unslot/datagen/questions.json"
    
    questbank = read_json(question_file)

    import random

    results = []
    for k, orgquestion in questbank.items():
        if random.random()> 0.65: continue
        band = [(j/10, 1) for j in  range(10, 95, 10)]
        for b, c in band:
            for i in range(c):
                if b == 1 and random.random() > 0.2: continue
                if b == 2 and random.random() > 0.2: continue
                if b == 3 and random.random() > 0.2: continue
                if b == 4 and random.random() > 0.5: continue
                if b == 5 and random.random() > 0.5: continue
                if b == 6 and random.random() > 0.5: continue
                if b == 7 and random.random() > 0.8: continue
                
                # print(b, i, orgquestion, "\n\t\t=================\n")
                # question = random.choice(question)
                func     = random.choice([call_gemini,  call_gemini])
                question = f"""
                You are a example generator. Your task is to act as a student taking Ielts speaking test, then generate response coresponding to Lexical Resource Band {b} and as well as the question.
                Given a question and the score band, based on the rubric, you have to generate a response example which satisfy conditions in the rubric to match the given score band.
                The mistake in example should be native like human speaking response and diversifying from the rubric.
                
                ======================

                The Lexical Resource rubric includes:
                ```
                {structured_text_rubric1}

                ```

                =====================

                The question is:
                ```
                {orgquestion}
                ```
                
                Please generate the responses in band {b} of Lexical Resource Band.
                Your response should be in json format as:
                {{
                    "student_response": "....",
                    "Lexical Resource": "Score X.",
                    "Reason" : "<Specify and explain mistake to make your mark, especially, why the response cannot get more score or why the mistake could acceptable>"
                }}
                
                
                #Note that
                - student_response could answer in various formats and you should make the lexical resource, error also diverse to match with the band {b}
                For better understanding how output should be here are some real examples:
                #Example 1
                ## Part: IELTS speaking test part: Part 1
                ##Topic: Place you live in\nQuestions:\nTell me about the street where you live\nWhat are the public transports services near your street bike?\nWould you like to live on the street for a very long time?
                    ## Answers:\nOh, I live in a magnificent street. It's called Vitebskaya, and it's in the coolest neighborhood in St. Petersburg.\nThis is a great question, um, because, uh, actually we do not have many, um, opportunities to get to the city center very quickly from the area where I live, but I see it as an advantage, um, well, it makes the neighborhood quieter, and, um, I don't know, it, um, I walk more because of that.\nYeah, this is what I'm debating right now, to be honest. I've enjoyed living there, but I think it's essential to move, like, every three or four years. So I've lived there for two years and a half, so probably I should, yeah, should get going soon.
                ## Topic: School trips\nQuestions:\nWhat was your favorite school trip when you were a child?\nWhat makes it a good school trip?\nDid you take the same kinds of school trips as your parents?\nHave any types of school trips become more popular these days?
                    ## Answers:\nAh, with your school trip. Well, um, can I, can I tell about the university trip? So it was a hike. Um, we went to the mountains and we spent there four days. That was my first experience hiking and, uh, since then I've, I've really, um, yeah, I've done it a lot.\nIt was very well organized, I felt safe and I got all the equipment, all the instructions and also there were great people with us and great leaders.\nI don't think so. I don't think my parents have ever been to the mountains because, um, that wasn't in Russia, that wasn't in this area, it was in America. So my school was, um, like located pretty much at the footstep of a mountain, and, uh, my parents did not study next to mountains, so no.\nIn Russian? Yeah, I think these days schools have more opportunities to take the schoolchildren abroad and I definitely don't think that was like a common practice at the time when my parents were children.
                ## Topic: Fashion\nQuestions:\nDo you consider yourself to be a fashionable person?\nIs being trendy important in your country?\nAre the most popular people in life the most fashionable?\nIs it easy to be fashionable without spending a lot of money?
                    ## Answers:\nUm, yes and no. I think sometimes when you don't pay too much attention to fashion, you just appear to be fashionable. But when you kind of give it a lot of thought, it kind of seems a little bit, um, too, how to say, too enforced?\nUm, some people may say so. Um, in St. Petersburg where I live, there are many trendy people and there are many, um, shopping malls and, I don't know, like designer boutiques where you can get, um, pretty like interesting clothes. Also, people buy clothes abroad. Um, but being trendy is not that important to me personally.\nOf course not I don't think so why not well look at I don't know look at some of the look at some of the businessmen who who we see yeah in the media I think they dress really really simply and make a point out of it.\nAbsolutely, three stores rule."
                ## Overall Band score Lexical Resource: 9
                
                #Example 2
                ## Part \nIELTS speaking test part: Part 2
                ## Topic: -\nQuestion:\nDescribe a time where you forgot to do something important.
                ## Answer:\nYes, um, I wanted to tell a very simple story that happened to me, uh, literally yesterday. So, um, I teach and also, um, I learn music. So, um, my schedule changes pretty much every week and also the schedule of my music lessons, um, changes, uh, yeah, pretty, uh, pretty often. So, normally, uh, I keep, um, like, I keep, um, like, like a diary of, like, when I scheduled my lessons and when my music teacher comes to me. But yesterday, I forgot, um, to, um, to put my music lesson into my schedule, so I scheduled my music lesson at the same time as, um, my, um, language lessons. So, and I remembered about it pretty much last minute, so, um, yeah, I had to do a lot of apologizing, uh, and I really, um, it really made me think, like, okay, what is more important for me, to make money or to learn music? So, but I ended up, um, compromising, so, um, I rescheduled one of my students for earlier time, had to cancel another lesson, but still, um, I didn't do anything with, um, with my music lesson because it was very important to me. Um, so, yeah, anyway, I think it happened because I was quite absent-minded and, um, maybe because I was meeting with my friends yesterday, so, um, yeah, but I, I think I should be a bit more responsible with, with, with my work and with my hobbies and with the time management.
                ## Overall Band score Lexical Resource: 9"
                
                #Example 3
                ## Part \nIELTS speaking test part: Part 2
                ## Topic: -\nQuestion:\nDescribe about one big decision you have to make in your life
                ## Answer:\nSo I remember I was six years old when I left my city where I was born, which is Sicily, and I moved to Milan. For four years, no, I haven't seen Sicily for over four years, and one day my father said, let's go to Sicily on holiday. And it was amazing because firstly, I met all my relatives that I haven't seen them for a long time. Secondly, I had a first touch with my origin, I would say, and how can I say, one day my father said that we don't have to forget our origin because if you know where we are, where we were born, we understand where we are going to go. That's always, always. Unfortunately, I didn't enjoy much my relatives, as I explained before, because I lived in Milan, which is so far from Sicily, and the most beautiful thing I have is just the memory at the moment, remembering them. Yes, birthday, yes, Christmas together, it was a nice experience.
                ## Overall Band score Lexical Resource: 7
                
                #Example 4
                ##IELTS speaking test part: Part 3
                ## Topic: Farm\nQuestions:\nAre the number of farmlands across the globe declining?\nWhy do you think this is/is not happening?\nWhat are the other advantages of having a large farmlands?\nDo you think that farmers and agricultural workers should be given more facilities by the government?\nHow can we improve the skill of farmers in a region?
                ## Answers:\nNumber of land, I'm particularly, yeah, maybe in north of Iran, many land of, is there actually better land under and...\nCause feel free and make happy and actually made people happy to be there. Cause when you go to a farmer and see there, for me was that when I saw my grandfather to be farming things, they always be happy to this, you know, maybe they, he works until the night and go there for many things and he be happy to enjoy it.\nExcept expensive things, I mean, I think it makes so better to be a larger land to many things that you are have to do.\nYeah, I was thinking about this, I actually think maybe oh or yes, I don't think so actually, but it's better to let them be free and do whatever they want, I think this is a good idea.\nMaybe this part should go with the government. They can make a meeting with farmers and teach them a lot of things and prepare them to be a good farmer.
                ## Overall Band score Lexical Resource: 3
                """
                
                
                try:
                    response = func(question)
                    print(response)
                except : response = "ERR"
                exp =   {
                        "question": question,
                        "response": response,
                        "band": b,
                        'type': 1 if "Part 1" in orgquestion else 2 if "Part 2" in orgquestion else 3
                    }
                print(exp)
                print("===============@@@==============")
                results += [exp ]
                if random.random() < 0.2: 
                    print(exp)
                

    new_results = []
    for i in results:
        if not isinstance(i["response"], str) and not (i["response"]  is None):
            i["response"]=i["response"].text
        new_results += [i]
            

    # Call the function with your data and filename
    save_to_jsonl(new_results , 'raw_sample_for_lra.jsonl')