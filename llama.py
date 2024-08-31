import re
import requests
import json
import csv
import time
real_num = 20
fake_num = 10

#generate real topics as per the domain
def generate_real_topics(domain):

    #Making a call to the API endpoint i.e. together.ai
    endpoint = 'https://api.together.xyz/v1/chat/completions'   #End point
    #Building a request body
    res = requests.post(endpoint, json={            
        "model": "meta-llama/Llama-3-8b-chat-hf",   #model used
        "max_tokens": 512,      #max number of tokens
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<|eot_id|>"
        ],
        "messages": [
            {
                "content": f"List {real_num} concepts related to {domain}, just names",     #prompt
                "role": "user"
            }
        ]
    }, headers={
        "Authorization": "Bearer f69ade01ffe6bfd246ea4ff05e23388d93eafa59059e93a5a86b15a80e023f34",     #authorization token
    })
    parsed_data = json.loads(res.content)       #loading the response content
    content = parsed_data["choices"][0]["message"]["content"]
    
    concept_regex = r"\d+\. (.+)"
    matches = re.findall(concept_regex, content)
    time.sleep(5)       #generates a prompt after a sleep of every 5 secs

    return matches

#Function to generate fake topics
def generate_fake_topics(domain):
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    res = requests.post(endpoint, json={        #request body
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "max_tokens": 512,      #max number of tokens
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<|eot_id|>"
        ],
        "messages": [
            {
                "content": f"List {fake_num} fake concepts related to {domain}, just names",        #prompt
                "role": "user"
            }
        ]
    }, headers={
        "Authorization": "Bearer f69ade01ffe6bfd246ea4ff05e23388d93eafa59059e93a5a86b15a80e023f34",
    })
    parsed_data = json.loads(res.content)
    content = parsed_data["choices"][0]["message"]["content"]       #fetching the content of the response from LLM
    
    concept_regex = r"\d+\. (.+)"
    matches = re.findall(concept_regex, content)
    time.sleep(5)

    return matches


#Function to fetch entities (meaningful terms, phrases,etc.) from the question
def get_entities_from_questions(question):
    prompt = f"""Given a question, extract all the meaningful and significant entities, phrases, or terms that are critical to understanding the question correctly. The entities should be specific and from the question, such as proper nouns (e.g., names of people, places, organizations), concepts, events, or any other important words or phrases directly from the question. Avoid extracting common words and concepts.\nReturn just the extracted entities no additional lines, each entity is between three single quotes. Here is an example output: '''entity'''\n{question}"""
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    res = requests.post(endpoint, json={
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<|eot_id|>"
        ],
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ]
    }, headers={
        "Authorization": "Bearer f69ade01ffe6bfd246ea4ff05e23388d93eafa59059e93a5a86b15a80e023f34",
    })
    parsed_data = json.loads(res.content)
    content = parsed_data["choices"][0]["message"]["content"]       #fetching the content of the response
    
    entities = content.split("'''")
    main_entities = []      #Grabbing the meaningful or main entities
    for entity in entities:
        text = entity.strip("\n")
        if len(text)>3:
            main_entities.append(text)

    time.sleep(5)       #making a call after every 5 secs
    
    return main_entities

#Function to generate fake topics related to a specific domain
def generate_fake_topics(domain):
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    res = requests.post(endpoint, json={
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<|eot_id|>"
        ],
        "messages": [
            {
                "content": f"List {fake_num} fake concepts related to {domain}, just names",
                "role": "user"
            }
        ]
    }, headers={
        "Authorization": "Bearer f69ade01ffe6bfd246ea4ff05e23388d93eafa59059e93a5a86b15a80e023f34",
    })
    parsed_data = json.loads(res.content)
    content = parsed_data["choices"][0]["message"]["content"]
    
    concept_regex = r"\d+\. (.+)"
    matches = re.findall(concept_regex, content)    #grabbing the text that matches regex defined
    time.sleep(5)

    return matches

#used for indexing purpose to create the dataset. Done to replicate the implementation of the paper
def add_token_with_index(data, index):
    sentences = data.split(".")
    combined_data = ""
    for i, sentence in enumerate(sentences, 1):
        if len(sentence) > 5:
            combined_data += f"{sentence.strip()}[^{index+1}^]. "
    return combined_data.strip()

#segregates the output into text and url
def get_text_and_url(text):
    texts = []
    urls = []
    for data in text.split("'''")[1:]:
        try:
            if data[:4] == "data":
                if 'url:' in data:
                    text, url = data[6:].split('url:')
                    texts.append(text.rstrip("\n"))
                    urls.append(url.rstrip("\n"))
                else:
                    texts.append(data[6:].rstrip("\n"))
        except Exception as e:
            continue
            
    combined_texts = " ".join(add_token_with_index(entry, i) for i, entry in enumerate(texts))
    combined_urls = " ".join(f"[{i}]: {entry}" for i, entry in enumerate(urls, 1))
    combined_data = " ".join(entry for entry in texts)
    
    return combined_texts, combined_urls, combined_data

#used for generating fake questions (for data augmenation purpose) as per a domain
def generate_fake_questions(topic, domain):
    #prompt
    prompt = f"give three questions on fake concept of {topic}, in the domain of {domain}, each entry should have a parameter named question and must be separate using three single quote. Here is an exaple of output '''question: '''"
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    
    #request body for the API
    res = requests.post(endpoint, json={
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<|eot_id|>"
        ],
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ]
    }, headers={
        "Authorization": "Bearer f69ade01ffe6bfd246ea4ff05e23388d93eafa59059e93a5a86b15a80e023f34",
    })
    parsed_data = json.loads(res.content)
    content = parsed_data["choices"][0]["message"]["content"]       #content extractiong from response
    
    pattern = r"'''question: (.+?)'''"
    questions = re.findall(pattern, content)
    time.sleep(5)
    
    all_entities = []
    #iterates through the generated questions and getches entities as per the function defined
    for question in questions:
        all_entities.append(get_entities_from_questions(question))
    
    return questions, all_entities



#Function to generate real questions
def generate_real_questions(combined_data):
    prompt = f"{combined_data}\nGenerate three unique questions using this data, each entry should have a parameter named question and each entry must be separated using three single quote. Here is an exaple of output '''question: '''"
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    res = requests.post(endpoint, json={
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<|eot_id|>"
        ],
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ]
    }, headers={
        "Authorization": "Bearer f69ade01ffe6bfd246ea4ff05e23388d93eafa59059e93a5a86b15a80e023f34",
    })
    parsed_data = json.loads(res.content)
    content = parsed_data["choices"][0]["message"]["content"]
    
    pattern = r"'''question: (.+?)'''"
    questions = re.findall(pattern, content)
    time.sleep(5)
    
    return questions

#Used for fetching info about a specific topic i.e. three entries (unique defn and two additional info beyond general defn)    
def get_topic_info(topic):
    prompt = f"Generate three entries about {topic}, each containing multiple sentences from a different website. One entry should include a definition of {topic} in one or more sentences. The other two entries should provide additional unique information beyond generalized definitions, spanning multiple sentences. For each entry, provide the content in a 'data' field with each sentence on a new line, and the source URL in a 'url' field. Enclose each entry in triple single quote for clear separation. Here is an example format: ''' data: This is a multi-sentence statement about choreography. url: https://example.com '''"
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    res = requests.post(endpoint, json={
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<|eot_id|>"
        ],
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ]
    }, headers={
        "Authorization": "Bearer f69ade01ffe6bfd246ea4ff05e23388d93eafa59059e93a5a86b15a80e023f34",
    })
    try:
        parsed_data = json.loads(res.content)
        content = parsed_data["choices"][0]["message"]["content"]
        
        data, urls, combined_data = get_text_and_url(content)
        time.sleep(5)
        # print(data, urls, combined_data)
        
        real_questions = generate_real_questions(combined_data)
        time.sleep(5)
        
        all_entities = []
        
        for question in real_questions:
            all_entities.append(get_entities_from_questions(question))
    except Exception as e:
        data = ""
        urls = ""
        real_questions=[]   
        all_entities=[]
        return data, urls, real_questions, all_entities
        
    # real_questions = ", ".join(question for question in real_questions)
    
    # print(real_questions)
    
    return data, urls, real_questions, all_entities
    
#Used for generating data to create the augmentated dataset which will be further used for evaluating the model
def generate_data(domain):
    data = []
    real_topics = generate_real_topics(domain)
    fake_topics = generate_fake_topics(domain)
    
    for topic in real_topics:
        print(topic)
        text, urls, real_questions, all_entities = get_topic_info(topic)
        d = {}
        d["Domain"] = domain
        d["Concept"] = topic
        d["Real"] = "TRUE"
        d["Label"] = "TRUE"
        d["Background"] = text
        d["Refs"] = urls
        d["Questions"] = real_questions
        d["Entities"] = all_entities
        data.append(d)
        time.sleep(5)
        
    for topic in fake_topics:
        questions, all_entities = generate_fake_questions(topic, domain)
        d = {}
        d["Concept"] = topic
        d["Domain"] = domain
        d["Real"] = "FALSE"
        d["Label"] = "FALSE"
        d["Background"] = ""
        d["Refs"] = ""
        d["Questions"] = questions
        d["Entities"] = all_entities
        data.append(d)
        
    file_path = "data.csv"

    # Extracting headers from the first dictionary
    headers = list(data[0].keys())

    # Save the data to a CSV file
    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # Write the header
        writer.writeheader()
        
        # Write each row
        for row in data:
            try:
                writer.writerow(row)
            except Exception as e:
                continue
    
    print(data)

#Real domains    
domains = [
    "Psychology",
    "Linguistics",
    "Sociology",
    "Anthropology",
    "Philosophy",
    "Environmental Science",
    "Geology",
    "Chemistry",
    "Biology",
    "Astronomy",
    "Mathematics",
    "Education",
    "Literature",
    "Linguistics",
    "Agriculture",
    "Architecture",
    "Aerospace",
    "Geography",
    "Political Science",
    "Economics",
    "Business Management",
    "Marketing",
    "Journalism",
    "Public Relations",
    "Communication Studies",
    "Film Studies",
    "Theater Arts",
    "Dance",
    "Religion/Theology",
    "Culinary Arts",
    "Fashion Design",
    "Graphic Design",
    "Industrial Design",
    "Interior Design",
    "Urban Planning",
    "Military Science",
    "Criminology",
    "Forensic Science",
    "Archaeology",
    "Paleontology",
    "Oceanography",
    "Materials Science",
    "Biotechnology",
    "Nanotechnology",
    "Robotics",
    "Computer Science",
    "Information Technology",
    "Software Engineering",
    "Game Design",
    "Renewable Energy",
]
    
for domain in domains:
    generate_data(domain)

            
    
    

    
