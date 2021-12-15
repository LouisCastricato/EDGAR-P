#!/usr/bin/env python
# coding: utf-8

from subject_verb_object_extract import *
import csv

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

import spacy
nlp = spacy.load("en_core_web_sm")

from transformers import (
    GPTNeoModel, 
    GPTNeoForCausalLM,
    GPT2Tokenizer, 
    GPTNeoConfig,
    AutoConfig
)
from transformers import (
    BeamSearchScorer,    
    LogitsProcessorList,
    StoppingCriteriaList,
    MinLengthLogitsProcessor,
    MaxLengthCriteria,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
)
from transformers.generation_logits_process import (
    LogitsProcessor,
    NoBadWordsLogitsProcessor, 
    NoRepeatNGramLogitsProcessor, 
    RepetitionPenaltyLogitsProcessor,
)


import transformers
transformers.logging.set_verbosity(transformers.logging.CRITICAL)
try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

import torch
from torch.nn import CrossEntropyLoss

import numpy as np
from scipy.special import softmax

f = open('100KStories.csv') 
csv_reader = csv.reader(f)
stories = list(csv_reader)

def load_gptj():
    def no_init(loading_code):
        def dummy(self):
            return
        
        modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
        original = {}
        for mod in modules:
            original[mod] = mod.reset_parameters
            mod.reset_parameters = dummy
        
        result = loading_code()
        for mod in modules:
            mod.reset_parameters = original[mod]
        
        return result

    model = no_init(lambda: AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision='float16', low_cpu_mem_usage=True))
    model.config.pad_token_id = model.config.eos_token_id

    return model.half().cuda()

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

class HorizonRepetitionPenalty(LogitsProcessor):
  def __init__(self, penalty: float, horizon: torch.LongTensor, horizon_exclusive = False):
    if not isinstance(penalty, float) or not (penalty > 0):
      raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

    self.penalty = penalty
    self.horizon=horizon
    self.exclusive=horizon_exclusive
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    num_beams = input_ids.shape[0]
    horizon = torch.cat(num_beams*[self.horizon], dim=0)
    if not self.exclusive:
      input_ids = torch.cat((input_ids, horizon), dim=-1)
    else:
      input_ids = horizon
    for i in range(scores.shape[0]):
      for previous_token in set(input_ids[i].tolist()):
        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        if scores[i, previous_token] < 0:
          scores[i, previous_token] *= self.penalty
        else:
          scores[i, previous_token] /= self.penalty
    return scores

bad_words_list = ["Because", "Because,", "Because ", "because", " Because", " Because,", 
" because", "Yes", "Yes,", "Yes ", "yes", " Yes", " Yes,", " yes",
" No", " No,", " no",
"(", " (", ")", ") "]
bad_words_ids = list(map(lambda x: tokenizer(x)['input_ids'], bad_words_list))
expl = [[1427],[2602, 834],[29343],[37405],[35780],[2602]]
bad_words_ids += expl

def clean_story(story):
  #Need to remove colons and line breaks
  story = story.replace(":", "-")
  story = story.replace("\n\n", "")
  story = story.replace("\n", " ")
  story = story.replace("<|endoftext|>", "")
  return story

#Takes a model and computes the perplexity of the target sequence given the input sequence
def perplexity(encodings, stride=1, m=None):
  lls = []
  inp_ids = encodings['input_ids']
  start = encodings['start']
  max_length = len(encodings['input_ids'].squeeze())
  for i in range(start, inp_ids.size(1), stride):
      begin_loc = max(i + stride - max_length, 0)
      end_loc = min(i + stride, inp_ids.size(1))
      trg_len = end_loc - i    # may be different from stride on last loop
      input_ids = inp_ids[:,begin_loc:end_loc].to("cuda")
      target_ids = input_ids.clone()
      target_ids[:,:-trg_len] = -100

      with torch.no_grad():
          outputs = m(input_ids, labels=target_ids)
          log_likelihood = outputs[0] * trg_len
      lls.append(log_likelihood)

  return (torch.exp(torch.stack(lls).sum() / end_loc)).item()

#Constructs a sequence for determining the perplexity of a target given a prompt
def construct(prompt, target, force_start=None):
  prompt_tok = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
  target_tok = tokenizer(target, add_special_tokens=False, return_tensors="pt")
  if force_start is None:
    start = len(prompt_tok['input_ids'].squeeze())
  else:
    start = force_start
  #Start encodes where the prompt sequence ends and target begins
  return {
      'input_ids': torch.cat((prompt_tok['input_ids'],target_tok['input_ids']), dim=-1).cuda(),
      'attention_mask': torch.cat((prompt_tok['attention_mask'],target_tok['attention_mask']), dim=-1).cuda(),
      'start':start,
  }

def generate(model, ids, max_length=1024,
    horizon=None,
    horizon_penalty=None,
    beams=2,
    extra_bad_words = None,
    repetition_penalty=2.0,
    do_beams = False):

  bad_words_t = bad_words_ids
  if extra_bad_words is not None:
    bad_words_t += extra_bad_words
  model_out=None
  if horizon is None:
    print("generating with no horizon")
    model_out = model.generate(input_ids = ids['input_ids'],
    max_length=max_length,
    num_beams=beams,
    no_repeat_ngram_size=5,
    bad_words_ids=bad_words_t,
    repetition_penalty=repetition_penalty)[0]
  else:
    horizon_ids = tokenizer(horizon, return_tensors="pt")['input_ids'].cuda()
    input_ids = ids["input_ids"]
    model.config.max_length = max_length
    # instantiate logits processors
    logits_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(ids['input_ids'].shape[1], model.config.eos_token_id),
        NoRepeatNGramLogitsProcessor(5),
        NoBadWordsLogitsProcessor(bad_words_t, eos_token_id=model.config.eos_token_id),
        HorizonRepetitionPenalty(penalty=horizon_penalty, horizon=horizon_ids, horizon_exclusive=True),
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
    ])
    stopping_criteria = StoppingCriteriaList([
        MaxLengthCriteria(max_length=max_length),
    ])
    model_kwargs={
        "attention_mask":ids['attention_mask'],
        "use_cache":True,
    }
    with torch.no_grad():
      if do_beams:
          beam_scorer = BeamSearchScorer(
                batch_size=ids["input_ids"].shape[0],
                num_beams=beams,
                device=model.device,
                length_penalty=1.0,
                do_early_stopping=True,
                num_beam_hyps_to_keep=1,
            )        
     
          input_ids, model_kwargs = model._expand_inputs_for_generation(
              ids["input_ids"], expand_size=beams, is_encoder_decoder=model.config.is_encoder_decoder
          )
        
          model_out = model.beam_search(
              input_ids=input_ids, beam_scorer = beam_scorer, logits_processor=logits_processor,\
              stopping_criteria=stopping_criteria)[0]
      else:
          model_out = model.greedy_search(
              input_ids=input_ids, logits_processor=logits_processor,\
              stopping_criteria=stopping_criteria)[0]         
    
  return tokenizer.decode(model_out)


rep = transformers.RepetitionPenaltyLogitsProcessor(1.1)

# computes the loss of a single token
def compute_loss(logits, labels):
  # Shift so that tokens < n predict n
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous()
  # Flatten the tokens
  loss_fct = CrossEntropyLoss()
  return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))



#Computes perplexity using a rep penalty
def perplexity_w_rep(encodings, stride=1, m=None):
  lls = []
  inp_ids = encodings['input_ids']
  start = encodings['start']
  max_length = len(encodings['input_ids'].squeeze())
  for i in range(start, inp_ids.size(1), stride):
      begin_loc = max(i + stride - max_length, 0)
      end_loc = min(i + stride, inp_ids.size(1))
      trg_len = end_loc - i    # may be different from stride on last loop
      input_ids = inp_ids[:,begin_loc:end_loc].to("cuda")
      target_ids = input_ids.clone()
      target_ids[:,:-trg_len] = -100

      with torch.no_grad():
          outputs = m(input_ids, labels=target_ids)
          logits = outputs.logits.squeeze()
          for i in range(1, len(logits)):
            ids_t = input_ids[0, :i].unsqueeze(0)
            logits_t = logits[i].unsqueeze(0)
            # this can be generalized to arbirary logit processors. potentially useful for zhiyu
            logits[i] = rep(ids_t, logits_t).squeeze()
          loss = compute_loss(logits.unsqueeze(0), target_ids)
          log_likelihood = loss * trg_len
      lls.append(log_likelihood)

  return (torch.exp(torch.stack(lls).sum() / end_loc)).item()

def rank(model, string, force_start=2):
  #Pull out the last word to use the construct function
  t1 = string.split()
  t2 = " ".join(t1[-1:])
  t1 = " ".join(t1[:-1])

  #Filtering out the first few tokens helps significantly. so force_start = 3
  return perplexity_w_rep(construct(t1, t2, force_start = force_start), m=model) 




#Take zero means that the beam is carrying questions, so take the first element
def rank_sort(model, stories, take_zero=True):
  if not take_zero:
    ranks = list(map(lambda x: rank(x, force_start=1, model=model), stories))
  else:
    ranks = list(map(lambda x: rank(x[0], force_start=1, model=model), stories))
  ranked_stories = zip(stories, ranks)
  sorted_stories = sorted(ranked_stories, key=lambda x: x[1])
  return list(map(lambda x: x[0], sorted_stories))