import openai
import os
from config import OPENAI_API_KEY_DICT
import json

# openai.api_key  = OPENAI_API_KEY

def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    client = openai.OpenAI(api_key=OPENAI_API_KEY_DICT['RUBRIC_MARKING'])
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=0.1,
    )
    return response.choices[0].message.content

structure = r"""
    {
        "score": <your score>,
        "explanation": <your explanation>,
    }
"""


def generate_criteria(statement, correct_answer, rubric_string, student_class):
    criteria_structure = r"""
        - Criteria name: <criteria name>
            - Contribution: <contribution>
            - Explanation: <explanation>
        - Criteria name: <criteria name>
        ...
    """

    prompt = """
    You are an educator for {student_class}. You are given the following Science question and the corresponding correct answer.

    Question is put inside the triple backticks below:
    ```{question}```
    Correct answer is put inside the triple backticks. In case there are multiple correct answers, they will be given as a list:
    ```{correct}```
    According to the marking guide, the following are the criteria for assessment:
    ```
    {rubric_string}
    ```
    Among those criteria, you need to choose from 2 to 5 criteria that are the most relevant to the question given and the correct answer. \
    When choosing the criteria, you must make sure that these criteria are always elaborated with the correct \
    answer as the reference. For example, what evidence should be used, What scientific phenomna to understand? \
    Always cite the correct answer to support your criteria. \
    Moreover, you also need to provide the contribution of each criteria to the final score. \
    The total contribution of all criteria should be 1. \
    A good contribution is one that focus on important knowledge from the question and correct answer.

    The language of the criteria and the explanation must be in the same language as the question and the correct answer. \

    Your output should be an object with the following structure:
    {structure}
    """.format(question=statement, correct=str(correct_answer), structure=criteria_structure, rubric_string=rubric_string, student_class=student_class)

    criteria = get_completion(prompt)
    return criteria

def generate_score(criteria, statement, answer, student_class):
    prompt = """
    You are an educator for {student_class}. Based on the following question and student answer, determine \
    the score for the student's answer. You should give a score between 0 and 1, where 0 is the worst answer \
    and 1 is the best answer. While grading, make sure that the student's answer is academically correct. For \
    each satisfied criteria, give a proof of how the student's answer satisfies the criteria.

    Use the following criteria to determine the score for the student's answer. It is put inside the triple backticks below:
    ```{criteria}```

    The question statement is put inside the triple backticks below:
    ```{question}```

    Student's answer is put inside the triple backticks. You need to decide the score of this answer:
    ```{answer}```


    Next, explain your grading in a paragraph. Your explanation paragraph must be in the same language as the student's answer. \
    If the student's answer is in Chinese, your explanation paragraph must be in Chinese. Your paragraph must point out the \
    student's strengths and weaknesses and how they can improve their answer. Provide concrete examples from the student's answer \
    to support your explanation.
    DO NOT mention the score in the explanation.

    You must answer in the following structure:
    {structure}

    """.format(criteria=criteria, question=statement, answer=answer, structure=structure, student_class=student_class)
    score = get_completion(prompt)
    return score

def get_avg_score(criteria, statement, answer, student_class, iterations=4):
    scores = []
    feedback_string = ""
    for i in range(iterations):
        score = generate_score(criteria, statement, answer, student_class)
        try:
            score_json = json.loads(score)
        except:
            continue
        scores.append(score_json['score'])
        feedback_string = score_json['explanation']
    return sum(scores)/len(scores), feedback_string

def convert_rubric_structure(rubric):
    """
        This function converts the rubric structure from the following format:
        [
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
        to a string format that can be used in the prompt.
        1. Claim and focus: State a clear claim on the scientific topic and maintain a focus on it throughout.
        2. Evidence: Use facts, definitions, and information from other sources to support and develop central idea about the issue or topic.
    """
    rubric_string = ""
    for i, criteria in enumerate(rubric):
        rubric_string += f"{i+1}. {criteria['criteria']}: {criteria['description']}\n"
    return rubric_string

def rubric_mark(high_level_criteria, statement, model_answer, student_answer, max_score, student_class="Primary Chinese"):
    high_level_criteria_string = convert_rubric_structure(high_level_criteria)
    criteria = generate_criteria(statement, model_answer, high_level_criteria_string, student_class)
    student_score, feedback = get_avg_score(criteria, statement, student_answer, student_class)
    model_score, _ = get_avg_score(criteria, statement, model_answer, student_class)
    try:
        modified_score = student_score / max(model_score, student_score)
    except:
        modified_score = 0

    return {
        "criteria": criteria,
        "student_score": student_score,
        "model_score": model_score,
        "modified_score": modified_score * max_score,
        "feedback": feedback
    }

if __name__ == "__main__":
    high_level_criteria = [
        {
            "criteria": "Criteria 1",
            "description": "因为表妹的英文成绩一直不好（1分）。",
        },
        {
            "criteria": "Criteria 2",
            "description": "乘着学校放假, 我也正好请假在家休息（1分）",
        },
        {
            "criteria": "Criteria 3",
            "description": "于是母亲便让她来我家跟我学英文（1分）",
        },
    ]
    paragraph = """
    阿里和阿布是两个同龄的年轻人，他们同时到一家店铺里工作，拿同样的薪水。
        可是，过了一段时间后，阿里得到老板的重用，而阿布的薪水却与刚进店时一样。阿布很不满意老板的不公正待遇，忍不住去找老板。老板一边耐心地听着他的抱怨，一边在心里盘算着怎样向他解释清楚他和阿里之间的差别。 
            “阿布，”老板开口说话了，“你现在到集市上去一下，看看有什么卖的。”
            阿布回来向老板报告说：“集市上只有一个农夫在卖马(mǎ)铃(líng)薯(shǔ)。”
            “有多少？”老板问。
            阿布赶快又跑到集市上，回来告诉老板，一共有四十袋马铃薯。
        “价格是多少？”
            阿布第三次跑到集市上问来了价格。
            “好吧。” 老板对他说，“现在请你坐到一边不要说话，看看阿里是怎么做的。”老板又布置了同样的任务给阿里。
    阿里很快就从集市上回来了，向老板报告：到现在为止只有一个农夫在卖马铃薯，一共四十袋，价格是多少。马铃薯质量不错，阿里还带回来一个让老板看看。这个农夫一个钟头以后还会弄来几箱番茄，阿里觉得价格非常合理。昨天他们铺子进的番茄卖得很快，剩的已经不多了。他想这么便宜的番茄，老板肯定会再进一些的，所以他把那个农夫带来了，那人现在正在外面等回话呢。
        老板听完，转向阿布，说: “现在你该知道为什么阿里的薪水比你高了吧？” 阿布心服口服，惭愧地说：“我的能力的确不如阿里。”
    """
    question_list = [
        {
            'question_statement': "文中形容“因为事情不如意，表示不满”的词语是。 （2分）",
            'model_answer': '抱怨',
            'student_answer': '抱怨',
            'score': 2,
        },
        {
            'question_statement': "文中形容“心里算计或筹划”的词语是。（2分）",
            'model_answer': '盘算',
            'student_answer': '盘算',
            'score': 2,
        },
        {
            'question_statement': "为什么阿布去找老板抱怨？（3分）",
            'model_answer': '因为阿布和阿里是两个同龄的年轻人，他们同时在一家店铺里工作，一开始，两人都拿同样的薪水。（1分）一段时间后，阿里得到老板的重用，而阿布的薪水却与刚进店时一样。（1分）阿布很不满意，认为老板不公正/待遇不公正。（1分）所以阿布去找老板抱怨。',
            'student_answer': '因为老板不公正待遇',
            'score': 3
        },
        {
            'question_statement': "老板分别让阿布和阿里到集市上看有什么东西卖，目的是什么？（1分）阿布和阿里的表现有什么不同？（2分）",
            'model_answer': '老板分别让阿布和阿里到集市上看有什么东西卖，目的是让阿布清楚他和阿里之间的差别/明白自己的不足（1分）。阿布去了三次才把事情解决（1分），阿里只去了一次，就把事情做好了（0.5分），而且还为老板带来了更多的有用信息（0.5分）。',
            'student_answer': '目的是看他们的能力',
            'score': 3
        },
        {
            'question_statement': "最后阿布听了老板的话，有什么感受？（1分）为什么？（3分）",
            'model_answer': '最后阿布听了老板的话，感到很惭愧（1分）。因为老板布置同样的工作分别让阿布和阿里去做（1分），阿布明白了自己的能力不如阿里（1分），他心服口服（1分）',
            'student_answer': '阿布感到羞愧，因为他的能力不如阿里',
            'score': 4
        },
        {
            'question_statement': "你认为老板是个怎样的人？试举例说明。（4分）",
            'model_answer': '老板为人公正（1分），他可以看清工人的能力，并根据能力公平地对待他们（1分）。老板也有智慧（1分），他用巧妙的方法，让工人明白自己的不足之处（1分）。',
            'student_answer': '老板是个善于发现员工潜力的人',
            'score': 4
        },
        {
            'question_statement': "在学习的道路上，你会以文中的哪一个人为榜样？为什么？（4分）",
            'model_answer': '在学习的道路上，我会以阿里为榜样。因为阿里做事积极主动（0.5分），肯动脑筋（0.5分），考虑周全（0.5分），这样可以把事情做得更好（0.5分），也会得到别人的赏识（1分）。如果我们在学习中有阿里这样的态度，就会学到更多的知识（1分）。',
            'student_answer': '我会以阿里为榜样，因为他能够主动发现商机并且善于预测市场需求。',
            'score': 4
        }
    ]
    for id, question in enumerate(question_list):
        thing = rubric_mark(high_level_criteria, paragraph + question['question_statement'], question['model_answer'], question['student_answer'], question['score'])
        print(thing['student_score'], thing['model_score'], thing['modified_score'], thing['feedback'])