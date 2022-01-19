from utility import *
from questions import *
from continuations import *


# load the language model
model = load_gptj()


def hasSVOExpand(sent):
  text = nlp(sent) 
  svos = findSVOs(text)
  if svos: 
    return True 
  else:
    return False


backwards = False

#check on random sample of stories
import random 
from tqdm import tqdm
r = list(range(len(stories)))
random.shuffle(r)

#at least one output  0.98
#total svoable percentages  0.8954248366013072

at_least_one_svoable_output = 0 
total_svoable_percents = 0 
total_sentences_checked = 0 
num_stories = 5


import json 
expansionfilenames = [
    "Archiveprompts/Prose_Enhancer_-_Past_Tense_1st_2021-07-13T19_20_37.875Z.story",
    "Archiveprompts/Prose_Enhancer_-_Past_Tense_2nd_2021-07-13T19_21_00.433Z.story",
    "Archiveprompts/Prose_Enhancer_-_Past_Tense_3rd_2021-07-13T19_21_12.748Z.story",
    "Archiveprompts/Prose_Enhancer_-_Past_Tense_3rd_2021-07-13T19_21_51.028Z.story",
        ]

expansion_prompt_jsons = []
for filename in expansionfilenames:
    with open(filename, 'r') as jsonfile:
        expansion_prompt_jsons.append(json.load(jsonfile))

expansion_prompts = [x['content']['story']['datablocks'][1]['dataFragment']['data'] for x in expansion_prompt_jsons]

def expand_text_simple(model, simple_sentence, expansion_prompt, verbose = False, width = 10):
  
  inp = expansion_prompt
  #inp = "Q. Why was 6 afraid of 7?"
  inp = inp.replace("[Insert Text Here]", simple_sentence)
  #inp = expansion_prompt_text.replace("[Insert Text Here]", "I shot the dog.")

  #print("input is ", inp, "\n\n")
  input_ids = tokenizer.encode(inp, return_tensors='pt').cuda()
  model_output = model.generate(
    input_ids,
    do_sample=True, 
    max_length=input_ids.shape[1] + 150, 
    top_k=50, 
    top_p=0.95, 
    early_stopping = True,
    num_return_sequences=1
  )

  #25 tokens for expansion
  text_output = tokenizer.decode(model_output[0], skip_special_tokens=True)
  #print("fulltext ", text_output)
  prose = text_output[len(inp):]
  #print(prose)
  return prose
    
def expand_story_simple(story, expansion_prompt, verbose = False):
    story_text = []
    for sentence in story:
        print("sentence is ", sentence)
        prose = expand_text_simple(sentence, expansion_prompt, verbose = verbose)
        prose = prose.split('\n')[0].strip()
        story_text.append(prose)
        print("prose is ", type(prose), prose)
    return story_text

#This stays on CPU
tokenizer_deberta = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge-mnli")
model_deberta = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xxlarge-mnli")

def not_contradict(sentA, sentB):
  if backwards:
    to_run = "[CLS]" + sentA  + "[SEP]" + sentB + "[SEP]"
  else:
    to_run = "[CLS]" + sentB  + "[SEP]" + sentA + "[SEP]"
  inputs = tokenizer_deberta(to_run, return_tensors="pt")
  with torch.no_grad():
    outputs = model_deberta(**inputs)

  outputs = softmax(np.array(outputs.logits.squeeze().cpu().tolist()))
  choice = np.argmax(outputs)
  if choice == 0:
    return -100
  print(outputs[2])
  return 100
  return outputs[2]
  
print(not_contradict("The sky is blue.", "The sky is not blue.")) #scores w/ BERT
print(not_contradict("The sky is blue.", "The sky is not red."))
print(not_contradict("The sky is blue.", "The ground is green."))

#Beams should be [[inp_story, [""]]] when we start
def beam_search(beams, width=20, diversity_width=2, graph=None, reverse_rank=True):
  print("beginning beam search fn")
  candidates = list()
  #Story is the story for this beam, q is questions already answered
  for story, q_prev, score in tqdm(beams):
    story_sents = sent_tokenize(story)
    #print(story)

    #Accumulate questions. q needs to start as [""]
    if backwards:
      questions = get_questions_past_tense(model, " ".join(story_sents[:min(len(story_sents),3)]), q_prev)
    else:
      questions = get_questions_future_tense(model, " ".join(story_sents[max(0, len(story_sents) - 3):]), q_prev)

    print("accumulated questions are ", questions)
    extensions = list()
    for q in questions:
      if backwards:
        continuation = continue_story_svo_customprompt_past_tense(model, story, q, width=-1)
      else:
        continuation = continue_story_svo_customprompt_future_tense(model, story, q, width=-1)
      print(continuation)
      if q_prev != ['']:
        q_cur = [[q] + q_prev]*len(continuation)
      else:
        q_cur = [[q]] * len(continuation)
      extensions += zip(continuation, q_cur)
    #Sort by most likely to imply and take the top k
    if backwards:
      implication_story = " ".join(story_sents[0:min(len(story_sents), 1)]) #The first k sentences sliding window
    else:
      #" ".join(story_sents[max(0, len(story_sents) - 3):])
      implication_story = " ".join(story_sents[-3:]) #The last k sentences sliding window
    extensions = list(filter(lambda x: hasSVOExpand(x[0]), extensions)) #Filter on if there is an SVO tuple
    extensions_ranks = list(map(lambda x: not_contradict(x[0], implication_story) + score, extensions)) #Rank. Include prior score via sum. This keeps track of beam scores over time.
    extensions_zip = list(filter(lambda x: x[1] > -100, zip(extensions, extensions_ranks))) #Zip
    extensions_zip = sorted(extensions_zip, key=lambda x: x[1], reverse=reverse_rank)
    extensions_zip = extensions_zip[:min(diversity_width, len(extensions_zip))] #Take top k
    if backwards:
      extensions = list(map(lambda x: (x[0][0]+" "+story, x[0][1], x[1]), extensions_zip)) #Sort
    else:
      extensions = list(map(lambda x: (story+" "+x[0][0], x[0][1], x[1]), extensions_zip)) #Sort
    #new_stories = list(map(lambda x: (x[0]+" "+story, x[1]), extensions)) #Concat
    #print(extensions)
    #Debug mode
    if graph is not None:
      for i in range(len(extensions)):
        graph.node(extensions[i][0])
        graph.edge(story_sents[0], extensions[i][0], label=extensions[i][1][0])

    #print("\n".join(new_stories))

    #Internally rank the new stories to preserve diversity
    candidates += extensions

  sorted_l = list(map(lambda x: (x[0], x[1], x[2]), sorted(candidates, key=lambda x: x[2], reverse=reverse_rank)))
  return sorted_l[:min(len(sorted_l), width)], graph

#Order will be given by beam rank, so this method eliminates the correct beam if duplicates are found
def remove_duplicates(beams):
  d = {}
  beams.reverse()
  for story, questions, scores in beams:
    d[story] = (questions, scores)
  to_ret = list()
  for k in d.keys():
    to_ret.append((k, d[k][0], d[k][1]))
  return to_ret

reverse = True
inp_story="The battle had been raging on for hours. John set his phasers to kill, he knew he had to make amends."
f=None

#beams, f = beam_search([(inp_story, [""], 0.0)], width=10, diversity_width=2, graph=f, reverse_rank=reverse)
#print("beams are ", beams)
#print("graph is ", f)

#for i in range(3):  
#  beams = remove_duplicates(beams)
#  print("beams are ", beams)
#  print("\nSTEP: " + str(i) + "\n\n")
#  print("\n".join(list(map(lambda x: x[0], beams))))
#  beams, f = beam_search(beams, width=5, diversity_width=10,graph=f, reverse_rank=reverse)


def gen_story_from_last_sentence(last_sentence, story_length=3):
  reverse = True
  inp_story=last_sentence
  f=None
  beams, f = beam_search([(inp_story, [""], 0.0)], width=10, diversity_width=2, graph=f, reverse_rank=reverse)
  print("beams are ", beams)
  print("graph is ", f)
  for i in range(story_length - 1):
    beams = remove_duplicates(beams)
    print("beams are ", beams)
    print("\nSTEP: " + str(i) + "\n\n")
    print("\n".join(list(map(lambda x: x[0], beams))))
    beams, f = beam_search(beams, width=5, diversity_width=10,graph=f, reverse_rank=reverse)
  
  return beams


story1 = "John's hand still trembles as he pushes open the twice-cooked door. \n The last time he saw the house he was glancing back over his shoulder as he and his sister fled into the trees. \n"
story2 = "Stella loves to eat peanuts! She goes to the market every day, and is best friends with the owner."
story3 = "There was a princess with long hair locked in a tower far away. \n A prince came to save her by climbing her hair and taking her out of the tower."
story4 = "I woke up realizing that I became a cat. \n I caught a mouse and at the next moment I realized that I'm a human again!"
story5 = "One night, I decided to go to McDonalds to get some ice cream. \n But when I got to the store, they said the machine was down; I was sad."
story6 = "He turned out the light and went into Jem's room. \n He would be there all night, and he would be there when Jem waked up in the morning. \n But little did Travis know, Jem knew about this all along."
story7 = "The hero charged at the dragon, both disappearing into the void. The world is saved and people will not forget the sacrifice of the nameless hero."

last_sentences = [story1, story2, story3, story4, story5, story6, story7]

import pickle 
for idx, sentence in enumerate(last_sentences):
    story_beams = gen_story_from_last_sentence(sentence, story_length = 3)
    
    with open('storypickle' + str(idx) + '.pickle', 'wb') as handle:
        pickle.dump(story_beams, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Dumping \n")
        print(story_beams)
    