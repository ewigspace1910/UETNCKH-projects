from Modules.knowledgebase import knowledgebase
# Import the necessary modules
import asyncio

async def main():
    pdf_url = 'https://heyjen-bucket.s3.ap-southeast-1.amazonaws.com/a77221a9-f1f8-4553-a7c6-37d58ad06f7e.pdf'
    kb_id = await knowledgebase.parse_pdf(pdf_url)
    print(f"Generated knowledge base ID: {kb_id}")

if __name__ == "__main__":
    # Run the main function using asyncio to await the asynchronous call
    # asyncio.run(main())

    criteria = [
        {
            "criteria": "Claim and Focus",
            "description": "State a clear claim on the scientific topic and maintain a focus on it throughout.",
        },
        {
            "criteria": "Evidence",
            "description": "Use facts, definitions, and information from other sources to support and develop central idea about the issue or topic.",
        },
        {
            "criteria": "Reasoning",
            "description": "The response demonstrates reasoning and understanding of the scientific topic and/ or source(s). and sufficiently explains the relationship between claim and evidence.",
        },
        {
            "criteria": "Language",
            "description": "Communicate ideas clearly using vocabulary specific to the scientific topic.",
        },
    ]
    api = "http://127.0.0.1:8008/rubric-marking/get-score"
    statement = "Dengue fever is spread by the Aedes mosquito. In town Z, the number of dengue cases is affected by the amount of rainfall as shown in the graphs below. What is the relationship between the amount of rainfall and the number of dengue cases from March to July?"
    model_answer = "As the amount of rainfall increases, the number of dengue cases increases."
    student_answer = "They are correlated with each other."
    payload = {
        "high_level_criteria": criteria,
        "question": statement,
        "correct": model_answer,
        "student": student_answer,
    }
    statement = "Mei Ling found an organism B growing on a piece of bread. She wanted to find out if it would grow without sunlight. She placed the piece of bread in a dark cupboard. After 4 days, she noticed that the organism B was still alive and there was more of it growing on the bread. What could organism B be?"
    model_answer = "Organism B is a mould, a type of fungus."
    student_answer = "The organism is a type of mold"
    payload2 = {
        "high_level_criteria": criteria,
        "question": statement,
        "correct": model_answer,
        "student": student_answer,
    }
    # use asyncio to send requests asynchronously
    loop = asyncio.get_event_loop()
    # do not use requests because it is not asynchronous
    # use aiohttp instead
    import aiohttp
    # avoid this: DeprecationWarning: The object should be created within an async function
    # task = loop.create_task(
    #     aiohttp.ClientSession().post(api, json=payload)
    # )
    # task2 = loop.create_task(
    #     aiohttp.ClientSession().post(api, json=payload2)
    # )
    # use this instead
    async def post(api, payload):
        async with aiohttp.ClientSession() as session:
            async with session.post(api, json=payload) as response:
                return await response.json()

    task = loop.create_task(post(api, payload))
    task2 = loop.create_task(post(api, payload2))
    # wait for the tasks to finish
    loop.run_until_complete(asyncio.wait([task, task2]))
    # get the results
    result = task.result()
    result2 = task2.result()
    # print the content of the result
    print(result)
    print(result2)
