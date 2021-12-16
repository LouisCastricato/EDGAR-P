from utility import *

def continue_story_svo_customprompt_past_tense(model, story, question, verbose = False, width = 10):
  instructions = "The following are results from a children's reasoning test.  \nThe child will be read a short story and then answer a question about what happened before the story.  \nThe answer cannot contradict the story.  \nWe have collected the results below:\n"
  story1="Story 1: Jim sat by the swings as Tom slowly approached. Tom gave the book to Jim during recess.\n"
  question1="Question: Why did Jim receive the book?\n"
  answer1="Answers:\n1. Tom desired to return the book to his friend.\n2. Jim bought the book from Tom.\n3. Tom noticed Jim dropped his book.\n4. Jim needed a book to study for his exam.\n5. Tom wanted to give Jim his favorite book.\n\n"
  story2 = "Story 2: Mary was happy to finally cross the street.\n"
  question2 = "Question: Why was Mary running?\n"
  answer2="Answers:\n1. Mary was running from a monster.\n2. Mary was running a marathon.\n3. Mary wanted to get away from her parents.\n\n" 
  story3 = "Story 3: " + story + "\n"
  question3 = "Question: " + question +"\n"
  inp = instructions + story1 + question1 +  answer1 + story2 + question2 + answer2 + story3 + question3
  #inp = "Q. Why was 6 afraid of 7?"
  if verbose:
    print("constructed inp ", inp, "\n"*2)
  con = construct(inp, "Answers:\n1.")
  
  #con = construct(inp, " A.")
  #print("construct is ", con, "\n"*2)
  out = generate(model, con, max_length=850, horizon=story, horizon_penalty=1.8, beams=5, repetition_penalty=2.8, do_beams=True)
  
  #return out
  out = out.split("Story 3")[1]
  #Remove the next story
  out = out.split("Story 4")[0]
  #Filter to correct answers
  #print("prefiltered out ", out)
  out = out.split("Answers:")[1]
  out = out.split("Wrong")[0]
  #print("filtered out is ", out)
  #If the user does not specify a width
  if width == -1:
    #Capture all of them
    width = 100
  responses = list()
  #Reads through the outputted list and returns every item
  for i in range(1, width):
    try:
      start = "\n"+str(i)
      end = "\n"+str(i+1)
      responses.append(out.split(start)[1].split(end)[0])
    except:
      break
  #Take responses that are long enough
  responses = list(filter(lambda x: len(x) > 5, responses))
  #print("og responses are ", responses)
  #Remove first space
  for i in range(len(responses)):
    try:
      if responses[i][:2] == '. ':
        responses[i] = responses[i][2:] 
      elif responses[i][:2] == ') ':
        responses[i] = responses[i][2:]
      elif responses[i][0] == ' ':
        responses[i] = responses[i][1:]

      responses[i] = " ".join(responses[i].split())
      responses[i] = clean_story(responses[i])
    except:
      continue
  return responses
  
def continue_story_svo_customprompt_future_tense(model, story, question, verbose = False, width = 10):
  instructions = "The following are results from a children's reasoning test.  \nThe child will be read a short story and then answer a question about what happened before the story.  \nThe answer cannot contradict the story.  \nWe have collected the results below:\n"
  story1="Story 1: Jim sat by the swings as Tom slowly approached. Tom gave the book to Jim during recess.\n"
  question1="Question: What did Jim do after recieving the book?\n"
  answer1="Answers:\n1. Jim jumped for joy!\n2. Jim threw the book away.\n3. Tom noticed that Jim looked unhappy.\n\n"
  story2 = "Story 2: Mary was happy to finally cross the street.\n"
  question2 = "Question: What happened after Mary crossed the street?\n"
  answer2="Answers:\n1. She sighed in relief.\n2. Her smile quickly faded when she remembered she left her wallet at the restaurant.\n3. She looked over her shoulder to make sure she was not being followed.\n\n" 
  story3 = "Story 3: " + story + "\n"
  question3 = "Question: " + question +"\n"
  inp = instructions + story1 + question1 +  answer1 + story2 + question2 + answer2 + story3 + question3
  #inp = "Q. Why was 6 afraid of 7?"
  if verbose:
    print("constructed inp ", inp, "\n"*2)
  con = construct(inp, "Answers:\n1.")
  
  #con = construct(inp, " A.")
  #print("construct is ", con, "\n"*2)
  out = generate(model, con, max_length=850, horizon=story, horizon_penalty=1.8, beams=5, repetition_penalty=1.8, do_beams=True)
  
  #return out
  out = out.split("Story 3")[1]
  #Remove the next story
  out = out.split("Story 4")[0]
  #Filter to correct answers
  #print("prefiltered out ", out)
  out = out.split("Answers:")[1]
  out = out.split("Wrong")[0]
  #print("filtered out is ", out)
  #If the user does not specify a width
  if width == -1:
    #Capture all of them
    width = 100
  responses = list()
  #Reads through the outputted list and returns every item
  for i in range(1, width):
    try:
      start = "\n"+str(i)
      end = "\n"+str(i+1)
      responses.append(out.split(start)[1].split(end)[0])
    except:
      break
  #Take responses that are long enough
  responses = list(filter(lambda x: len(x) > 5, responses))
  #print("og responses are ", responses)
  #Remove first space
  for i in range(len(responses)):
    try:
      if responses[i][:2] == '. ':
        responses[i] = responses[i][2:] 
      elif responses[i][:2] == ') ':
        responses[i] = responses[i][2:]
      elif responses[i][0] == ' ':
        responses[i] = responses[i][1:]

      responses[i] = " ".join(responses[i].split())
      responses[i] = clean_story(responses[i])
    except:
      continue
  return responses
  