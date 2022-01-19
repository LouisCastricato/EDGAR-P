from utility import *

questions_bad_words = ["What", " What", "\nWhat", " what", "what"]
questions_bad_words_ids = list(map(lambda x: tokenizer(x)['input_ids'], questions_bad_words))

questions_bad_words_future_tense = ["!?", "Why",  "Story", "You", "I", "Think", "Jane", "Car", "Driver", "Question?", "Question"]
questions_bad_words_future_tense = sum(list(map(permute_string, questions_bad_words_future_tense)), [])
questions_bad_words_ids_future_tense = list(map(lambda x: tokenizer(x)['input_ids'], questions_bad_words_future_tense))

def process_question_output(out):
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
  return good_questions[0]  

def get_questions_past_tense(model, story, bad_questions=None):
  instructions = "This is a rubric for grading a student's detective exam. If they give a wrong answer or an answer similar to a wrong answer, subtract one mark. If they give a right answer, add one mark. An answer cannot be both right and wrong. \n"
  story1 = "Question 1) John went for a swim.\n"
  good1 = "Acceptable Answers: How did John get to the swimming pool? What happened before John went swimming? Why did John go swimming?\n\n"
  bad1 = "Wrong Answers: Who is John? What did the pool water taste like? What happened after John went swimming? What does John do now?\n"


  story2 = "Question 2) The walk to school that day was long but, John was motivated to give Jane back her book. John gave the book to Jane.\n"
  good2 = "Acceptable Answers: Why was John motivated? How did John get the book? Why did John give Jane the book? Why did Jane want the book?\n\n"
  bad2 = "Wrong Answers: Who were they? Did Jane want the book? How are they similar? What is John wearing? When did John give Jane the book? Where is the book? What happens next? What is Jane going to do once she gets the book?\n"

  story3 = "Question 3) Jane was so happy to have finally crossed the street.\n"
  good3 = "Acceptable Answers: Why did Jane cross the street? Why was Jane unhappy? Why was Jane running from someone?\n\n"
  bad3 = "Wrong Answers:  What does Jane do after crossing the street? What does Jane do now that she is happy? What happens next?\n"

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
  instructions = "" #This is a rubric for grading a student's detective exam. If they give a wrong answer or an answer similar to a wrong answer, subtract one mark. If they give a right answer, add one mark. An answer cannot be both right and wrong. \n"
  story1 = "Question 1) John went for a swim.\n"
  good1 = "Acceptable Answers: What happened after John went swimming? What does John want to do now?\n\n"
  #bad1 = "Wrong Answers: Who is John? What did the pool water taste like? How did John get to the swimming pool? What happened before John went swimming? Why did John go swimming?\n"


  story2 = "Question 2) The walk to school that day was long but, John was motivated to give Jane back his book. John gave the book to Jane.\n"
  good2 = "Acceptable Answers: What is Jane going to do once she gets the book? How does Jane react to receiving the book? What happens next?\n\n"
  #bad2 = "Wrong Answers: Who were they? Did Jane want the book? How are they similar? What is John wearing? When did John give Jane the book? Where is the book?\n"

  story3 = "Question 3) Jane was so happy to have finally crossed the street. She had finally gotten away from it.\n"
  good3 = "Acceptable Answers:  What proceeds Jane crossing the street? What is the result of her being happy? What happens after she is on the other side of the street?\n\n"
  #bad3 = "Wrong Answers: Why did Jane cross the street? Why was she unhappy? Why was she running from someone?\n"
  beam_count = 2
  if bad_questions == ['']:
    bad_questions=None

  if bad_questions is not None:
    bad_questions = " ".join(bad_questions)
  story4 = "Question 4) " + story + "\n"
  inp = instructions + story1 +  good1 + story2 + good2 + story3 + good3 + story4

  prompts = ["What happens", "What happens after", "How does the protagonist react to"]
  out = list(map(lambda i: generate(model, construct(inp, "\nAcceptable Answers: " + prompts[i]),
  max_length=20, repetition_penalty=2.0,
  extra_bad_words=questions_bad_words_ids_future_tense,
  horizon_penalty=1.5,
  horizon=None,
  num_return_sequences=2, beams=beam_count, do_sample=True)[0], range(3)))

  return list(map(process_question_output, out))

#model = load_gptj()

#print(get_questions_future_tense(model, "William loved puppies, until that one fateful day."))