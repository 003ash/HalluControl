'''
Instructions to run:
* !git clone https://github.com/soap117/Self-evaluation.git
* cd Self-evaluation
* !pip install accelerate
'''

import os.path
import pickle
import time
import torch

# from myutils import *
import types
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import metrics
import numpy as np
import csv
import re
# from train_para import generate_parser
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import combinations
from transformers import StoppingCriteria, StoppingCriteriaList
from itertools import combinations
import pandas as pd
import requests
import torch
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import os
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Args:
    def __init__(self):
        self.exp_file = "7"
        self.model = 'openlm-research/open_llama_3b'
        self.baseline = 'BERTScore_baseline'
        self.unk_token = '...'
        self.stop_token = ''
        self.unk_token_id = 3346
        self.forward_search_length = 200
        self.backward_search_size = 30
        self.backward_search_length = 10

args = Args()


para_list = [
                ['openlm-research/open_llama_3b', '...', 856, ['</s>'], 200, 15],
]
common_english_words_file = './data/wiki-100k.txt'
common_english_words = []
with open(common_english_words_file, 'r', encoding='utf-8') as fr:
    for line in fr:
        if line[0] == '#':
            continue
        common_english_words.append(line.strip())
word_rare = {}
for id, word in enumerate(common_english_words):
    word_rare[word] = np.exp(-(id + 1) / 100)
common_english_words = common_english_words[0:10000]


data_file = 'data/data_knowledge_7_general_multi.csv'


para = para_list[0]
args.model = para[0]
args.stop_token = para[3]
args.unk_token = para[1]
args.unk_token_id = para[2]
args.forward_search_length = para[4]
args.backward_search_length = para[5]


#print(args)
max_memory = {0: "50GiB", 1: "40GiB", 2: "40GiB", "cpu": "40GiB"}
mpt = args.model.split('/')[-1]

# III. Transformers - Using transformer from the hugging face
model = AutoModelForCausalLM.from_pretrained("{mpt}".format(mpt=args.model), cache_dir='/data/cache',
                                              trust_remote_code=True, device_map="sequential",
                                              max_memory=max_memory, torch_dtype=torch.bfloat16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("{mpt}".format(mpt=args.model), cache_dir='/data/cache')

if args.unk_token_id is None:
    args.unk_token_id = tokenizer.vocab[args.unk_token]
tokenizer.unk_token_id = args.unk_token_id
tokenizer.unk_token = args.unk_token

stop_words = args.stop_token
stop_words_ids = [
    tokenizer.vocab[stop_word] for stop_word in stop_words]

#Defining stoping criteria to be used in concept guessing and concept inference
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.encounters = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
            if stop_count >= self.encounters:
                return True
        return False
    
s1 = StoppingCriteriaSub(stops=stop_words_ids, encounters=1)
print(stop_words_ids)
stopping_criteria = StoppingCriteriaList([s1])


# IV. Application - Function to give a paragraph on the given entity
def concept_guessing(entity, model, stopping_criteria, tokenizer, max_length=100):
    prompt1 = f"Explain the \"{entity}\" within one short paragraph."
    prompt2 = f"USER: {prompt1} ASSISTANT:"
    input_ids = tokenizer(prompt2, return_tensors="pt").input_ids.to(my_device)
    output = model.generate(input_ids, output_scores=True, return_dict_in_generate=True, max_length=max_length+input_ids.shape[-1], stopping_criteria=stopping_criteria)
    print("Original Question Output:\n{}".format(tokenizer.decode(output.sequences[0])))
    return output

#Function to mask the entity words from the prompt which is to be given to the concept inference function
def mask_input_words(prompt, tokenized_entities, tokenizer):
    # Generate variations of target words
    words_to_mask = [word for target_word in tokenized_entities
                    for word in {target_word, target_word.lower(), target_word.upper(), target_word.capitalize()}]

    # Sort target words by length in descending order
    words_to_mask.sort(key=lambda x: len(x), reverse=True)

    # Replace target words with tokenizer's unknown token
    for target_word in words_to_mask:
        prompt = prompt.replace(target_word, tokenizer.unk_token)

    return prompt


#Infer the entities which are masked from the concept guessing phase
def concept_inference(entity, tokenized_entities, domain, output_forward):
    concept_guessing_output = tokenizer.decode(output_forward.sequences[0], skip_special_tokens=True)
    concept_guessing_output = concept_guessing_output.rstrip()

    # Check if the last five characters are 'User ' and remove them if true
    if concept_guessing_output.endswith('User '):
        concept_guessing_output = concept_guessing_output[:-5]

    tokenizer.unk_token = args.unk_token
    prompt1 = f"\"{concept_guessing_output}\" is related to what?"
    prompt2 = f"USER: {prompt1} ASSISTANT:"

    prompt_masked = mask_input_words(prompt2, tokenized_entities, tokenizer)
    input_ids_mask = tokenizer(prompt_masked, return_tensors="pt").input_ids.to(my_device)

    target_entity = tokenized_entities[-1]

    target_word_variations = list(set([target_entity, target_entity.lower(), target_entity.upper(), ' '.join([x.capitalize() for x in target_entity.split()])]))

    target_word_variations = target_word_variations + [' ' + x for x in target_word_variations]

    force_words_ids = tokenizer(target_word_variations, add_special_tokens=False).input_ids

    try:
        #II. Probabilistic Model - Using beam search for the output entities
        outputs_constraint = model.generate(
            input_ids_mask,
            force_words_ids=[force_words_ids],
            output_scores=True,
            return_dict_in_generate=True,
            max_length=input_ids_mask.shape[-1]+len(force_words_ids)+args.backward_search_length,
            stopping_criteria=stopping_criteria,
            num_beams=args.backward_search_size,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        backward_text = tokenizer.decode(outputs_constraint.sequences[0])
        back_score = torch.exp(outputs_constraint.sequences_scores).item()
        print("Back Score", back_score)
    except Exception as e:
        back_score = 0
        backward_text = ""
        print(e)

    return back_score, backward_text

#I. Classification - Classifying word as rare or not
def entity_rare(entity):
    entity = entity.split(' ')
    rare = 1
    for word in entity:
        if word not in word_rare or word[0].isupper():
            rare *= np.exp(-len(word_rare) / 100)
        else:
            rare*=word_rare[word]
    return rare


#Preprocessing the entities to form combined entities
def merge_entities(entities, question):
    # Create a set to store final entities
    final_entities = set(entities)

    # Iterate over every combination of 2 entities
    for entity1, entity2 in combinations(entities, 2):
        # Create the merged entities
        merged_no_space = entity1 + entity2
        merged_with_space = entity1 + ' ' + entity2

        # Check if the merged entity is in the question
        if merged_no_space.lower() in question.lower() or merged_with_space.lower() in question.lower():
            # Add the merged entity to the final set
            final_entities.add(merged_no_space if merged_no_space in question else merged_with_space)

    # Convert the set back to a list
    final_entities = list(final_entities)

    # Return the final list of entities
    return final_entities


#Extracting the final entities after applying the preprocessing function
def extract_entities(entities, question):
    """
    Extract the main entities from the entities array that are present in the given question.
    Returns a list of extracted entities.
    """

    print("original: ", entities)
    extracted_concept = sorted(entities, key=lambda x: question.lower().find(x.lower()))
    extracted_concept = merge_entities(extracted_concept, question)
    # extracted_concept_cool = remove_covered_entities(extracted_concept)
    # extracted_concept = [x for x in extracted_concept if x in extracted_concept_cool]
    new_concept = [concept_one for concept_one in extracted_concept if concept_one not in common_english_words]
    if len(new_concept) != 0:
        extracted_concept = new_concept
    # Sort by their position in question
    extracted_concept = sorted(extracted_concept, key=lambda x: entity_rare(x), reverse=False)

    return extracted_concept

#Driver function to streamline all the processes in the pipeline
def calculate_fb_score(file_name):

  data = pd.read_csv(file_name, encoding='utf-8')

  # Iterate over each row in the DataFrame
  for index, row in data.iterrows():
      concept = row['Concept']
      extracted_concepts = eval(row['Entities'])
      questions = eval(row['Questions'])
      domain = row['Domain'].lower()

      beam_ouputs = []

      # Iterate over extracted concepts and questions
      for extracted_concept, question in zip(extracted_concepts, questions):

        final_extracted_entities = extract_entities(extracted_concept, question)

        backward_scores = []
        backward_texts = []

        for entity in final_extracted_entities:

          tokenized_entity = word_tokenize(entity)
          # remove the stop words in tokenized_entity
          tokenized_entity = [x for x in tokenized_entity if x.lower() not in stop_words] + [entity]
          print(entity, tokenized_entity)
          # getting the forward score and the forward text from the model
          output_forward = concept_guessing(entity, model, stopping_criteria, tokenizer)
          text_forward = tokenizer.decode(output_forward.sequences[0], skip_special_tokens=True)
          print(text_forward)
          backward_score, backward_text = concept_inference(entity, tokenized_entity, domain, output_forward)
          backward_scores.append(backward_score)
          backward_texts.append(backward_text)

        beam_ouputs.append([backward_scores, backward_texts])
        
calculate_fb_score(data_file)

#Source - https://github.com/soap117/Self-evaluation.git