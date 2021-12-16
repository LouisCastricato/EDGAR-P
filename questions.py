from utility import *

questions_bad_words = ["What", " What", "\nWhat", " what", "what"]
questions_bad_words_ids = list(map(lambda x: tokenizer(x)['input_ids'], questions_bad_words))

questions_bad_words_future_tense = ["Why", " Why", "\nWhy", " why", "why",\
  "Story", " Story", "\nStory", " story", "story", "story?", " story?",\
  "you", " you", "You", " You",\
  "I", " I",\
  "Think", " Think", "think", " think",
  "Erica", " Erica", "erica", " erica"]
questions_bad_words_ids_future_tense = list(map(lambda x: tokenizer(x)['input_ids'], questions_bad_words))



def get_questions_past_tense(model, story, bad_questions=None):
  instructions = "This is a rubric for grading a student's detective exam. If they give a wrong answer or an answer similar to a wrong answer, subtract one mark. If they give a right answer, add one mark. An answer cannot be both right and wrong. \n"
  story1 = "Question 1) John went for a swim.\n"
  good1 = "Acceptable Answers: How did John get to the swimming pool? What happened before John went swimming? Why did John go swimming?\n\n"
  bad1 = "Wrong Answers: Who is John? What did the pool water taste like? What happened after John went swimming? What does John do now?\n"


  story2 = "Question 2) The walk to school that day was long but, Tom was motivated to give Jim back his book. Tom gave the book to Jim.\n"
  good2 = "Acceptable Answers: Why was Tom motivated? How did Tom get the book? Why did Tom give Jim the book? Why did Jim want the book?\n\n"
  bad2 = "Wrong Answers: Who were they? Did Jim want the book? How are they similar? What is Tom wearing? When did Tom give Jim the book? Where is the book? What happens next? What is Jim going to do once he gets the book?\n"

  story3 = "Question 3) Erica was so happy to have finally crossed the street.\n"
  good3 = "Acceptable Answers: Why did Erica cross the street? Why was Erica unhappy? Why was Erica running from someone?\n\n"
  bad3 = "Wrong Answers:  What does Erica do after crossing the street? What does Erica do now that she is happy? What happens next?\n"

  if bad_questions == ['']:
    bad_questions=None
  story4 = "Question 4) " + story + "\n"
  if not bad_questions is None:
    bad4 = " " + " ".join(bad_questions) 
  else:
    bad4 = ""
  inp = instructions + story1 + bad1 +  good1 + story2 + bad2 + good2 + story3 + bad3 + good3 + story4


  out = generate(model, construct(inp, "\nAcceptable Answers: Why"), beams = 5, max_length=512, repetition_penalty=2.8, extra_bad_words=questions_bad_words_ids)
  #print(out)
  #Get questions out
  out = out.split("Question 4)")[1]
  try:
    out = out.split("Question")[0]
  except:
    pass
  out = out.split("Acceptable Answers:")[1]
  #print(out)
  out = sent_tokenize(out)
  good_questions = list()
  #As soon as we find a bad question, break
  for string in out:
    if not ("\n" in string):
      good_questions.append(string)
    else:
      break
  #Remove the space from the first question
  if good_questions[0][0] == ' ':
    good_questions[0] = good_questions[0][1:]
    
  good_questions = [x for x in good_questions if x != '<|endoftext|>']  
  return good_questions

def get_questions_future_tense(model, story, bad_questions=None):
  instructions = "This is a rubric for grading a student's detective exam. If they give a wrong answer or an answer similar to a wrong answer, subtract one mark. If they give a right answer, add one mark. An answer cannot be both right and wrong. \n"
  story1 = "Question 1) John went for a swim.\n"
  good1 = "Acceptable Answers: What happened after John went swimming? What does John want to do now?\n\n"
  bad1 = "Wrong Answers: Who is John? What did the pool water taste like? How did John get to the swimming pool? What happened before John went swimming? Why did John go swimming?\n"


  story2 = "Question 2) The walk to school that day was long but, Tom was motivated to give Jim back his book. Tom gave the book to Jim.\n"
  good2 = "Acceptable Answers: What is Jim going to do once he gets the book? How does Jim react to receiving the book? What happens next?\n\n"
  bad2 = "Wrong Answers: Who were they? Did Jim want the book? How are they similar? What is Tom wearing? When did Tom give Jim the book? Where is the book?\n"

  story3 = "Question 3) Erica was so happy to have finally crossed the street.\n"
  good3 = "Acceptable Answers:  What does Erica do after crossing the street? What is the result of her being happy? What happens after she is on the other side of the street?\n\n"
  bad3 = "Wrong Answers: Why did Erica cross the street? Why was she unhappy? Why was she running from someone?\n"

  if bad_questions == ['']:
    bad_questions=None
  story4 = "Question 4) " + story + "\n"
  if not bad_questions is None:
    bad4 = " " + " ".join(bad_questions) 
  else:
    bad4 = ""
  inp = instructions + story1 + bad1 +  good1 + story2 + bad2 + good2 + story3 + bad3 + good3 + story4


  out = generate(model, construct(inp, "\nAcceptable Answers: What happens after"), beams = 5, 
  max_length=512, repetition_penalty=2.0, extra_bad_words=questions_bad_words_ids_future_tense)
  #print(out)
  #Get questions out
  out = out.split("Question 4)")[1]
  try:
    out = out.split("Question")[0]
  except:
    pass
  out = out.split("Acceptable Answers:")[1]
  #print(out)
  out = sent_tokenize(out)
  good_questions = list()
  #As soon as we find a bad question, break
  for string in out:
    if not ("\n" in string):
      good_questions.append(string)
    else:
      break
  #Remove the space from the first question
  if good_questions[0][0] == ' ':
    good_questions[0] = good_questions[0][1:]
    
  good_questions = [x for x in good_questions if x != '<|endoftext|>']  
  return good_questions

