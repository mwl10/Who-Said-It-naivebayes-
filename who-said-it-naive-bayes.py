 
# YOUR NAME HERE: Matthew Lowery
# ASSN 02: "Who Said It?"


### Some STEPs are already COMPLETE.
### Commands you need to EDIT are marked as such.
###   <-- They are shown as empty lists/None object/0.0, etc.
###    <-- so that the script can be run without breaking.

import nltk, random

#------------------------------------------------ STEP 1 (COMPLETE)
print("1. Loading Austen and Melville sentences...")
a_sents_all = nltk.corpus.gutenberg.sents('austen-emma.txt')
m_sents_all = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')

#------------------------------------------------ STEP 2
print("2. Discarding short sentences and labeling...")
a_sents = [(s, 'austen') for s in a_sents_all if len(s)>2]
m_sents = [(s, 'melville') for s in m_sents_all if len(s)>2]    # EDIT

#------------------------------------------------ STEP 3
print("3. Joining the two author sentence lists...")
sents = a_sents + m_sents      # EDIT

#------------------------------------------------ STEP 4 (COMPLETE)
print("4. Sentence stats:")
print(" # of total sentences:", len(sents))
print(" # of Austen sentences:", len(a_sents))
print(" # of Melville sentences:", len(m_sents))

#------------------------------------------------ STEP 5
print("5. Shuffling...")
# EDIT -- shuffle sents here
random.Random(10).shuffle(sents)
#random.shuffle(sents)

#------------------------------------------------ STEP 6
print("6. Partitioning...")

test_sents = sents[:1000] # EDIT
devtest_sents = sents[1000:2000]  # EDIT
train_sents = sents[2000:]    # EDIT

print(" # of test sentences:", len(test_sents))
print(" # of devtest sentences:", len(devtest_sents))
print(" # of training sentences:", len(train_sents))

#------------------------------------------------ STEP 7 (COMPLETE)
print("7. Defining a feature-generator function...")
mainchars = {'Emma', 'Harriet', 'Ahab', 'Weston', 'Knightley', 'Elton',
             'Woodhouse', 'Jane', 'Stubb', 'Queequeg', 'Fairfax', 'Churchill',
             'Frank', 'Starbuck', 'Pequod', 'Hartfield', 'Bates', 'Highbury',
             'Perry', 'Bildad', 'Peleg', 'Pip', 'Cole', 'Goddard',
             'Campbell', 'Donwell', 'Dixon', 'Taylor', 'Tashtego'}

noCharNames = False    # For [PART B] Q3
if noCharNames :
    print('NOTE: Top 35 proper nouns have been neutralized.')

def gen_feats(sent):
    featdict = {}
    for w in sent:
        if noCharNames == True:
            if w in mainchars:
                w = 'MontyPython'
        featdict['contains-'+ w.lower()] = 1
    return featdict

#------------------------------------------------ STEP 8
print("8. Generating feature sets...")

test_feats = [(gen_feats(n), author) for (n, author) in test_sents]
devtest_feats = [(gen_feats(n), author) for (n, author) in devtest_sents]
train_feats = [(gen_feats(n), author) for (n, author) in train_sents]

#------------------------------------------------ STEP 9
print("9. Training...")

whosaid = nltk.NaiveBayesClassifier.train(train_feats)      # EDIT

#------------------------------------------------ STEP 10
print("10. Testing...")
accuracy = nltk.classify.accuracy(whosaid, test_feats)      # EDIT
print(" Accuracy score:", accuracy)

#------------------------------------------------ STEP 11
print("11. Sub-dividing development testing set...")
# aa: real author Austen, guessed Austen
# mm: real author Melville, guessed Melville
# am: real author Austen, guessed Melville
# ma: real author Melville, guessed Austen
aa, mm, am, ma = [], [], [], []
for (sent, auth) in devtest_sents:
    guess = whosaid.classify(gen_feats(sent))
    if auth == 'austen' and guess == 'austen':
        aa.append( (auth, guess, sent) )
    elif auth == 'melville' and guess == 'melville':
        mm.append( (auth, guess, sent) )
    elif auth == 'austen' and guess == 'melville':
        am.append( (auth, guess, sent) )
    elif auth == 'melville' and guess == 'austen':
        ma.append( (auth, guess, sent) )
    # EDIT below to populate mm, am, ma

#------------------------------------------------ STEP 12
print("12. Sample CORRECT and INCORRECT predictions from dev-test set:")
print("-------")
for x in (aa, mm, am, ma):  # EDIT change (aa) to (aa, mm, am, ma)
    auth, guess, sent = random.choice(x)
    print('REAL=%-8s GUESS=%-8s' % (auth, guess))  # string formatting
    print(' '.join(sent))
    print("-------")
print()

#------------------------------------------------ STEP 13
print("13. Looking up 40 most informative features...")
whosaid.show_most_informative_feats_all(40)

#### CODE FOR PART B INSTEAD OF SHELL SESSION

# Part B number 4: Trying out sentences
austen_sent = "Anne was to leave them on the morrow, an event which they all dreaded."
carroll_sent = "So Alice began telling them her adventures from the time when she first saw the White Rabbit."
madeup_3 = "He knows the truth"
madeup_4 = "She knows the truth"
madeup_5 = "blahblahblah blahblah"
# tokenize the sentences
austen_sent_token = nltk.word_tokenize(austen_sent)
carroll_sent_token = nltk.word_tokenize(carroll_sent)
madeup_3_t = nltk.word_tokenize(madeup_3)
madeup_4_t = nltk.word_tokenize(madeup_4)
madeup_5_t = nltk.word_tokenize(madeup_5)

print("-------------------------------------------")
# pass them through the classifier's classify() method and print the guesses
print("Part B Number 4: Trying out sentences:")
print()

print("Guess by classifier for sentence from Austen's Persuasion:")
print(whosaid.classify(gen_feats(austen_sent_token)))
print("------")
print("Guess by classifier for sentence from Carroll's Alice in Wonderland:")
print(whosaid.classify(gen_feats(carroll_sent_token)))
print()
print("----------------------------------------------")


print("Part B Number 5: Label Probabilities for a sentence:")
print()

print(whosaid.prob_classify(gen_feats(austen_sent_token)).prob('austen'), "P(A)")
print(whosaid.prob_classify(gen_feats(austen_sent_token)).prob('melville'), "P(M)")
print("-------")
print(whosaid.prob_classify(gen_feats(carroll_sent_token)).prob('austen'), "P(A)")
print(whosaid.prob_classify(gen_feats(carroll_sent_token)).prob('melville'), "P(M)")
print("--------------------------------------------")


print("Part B Number 6: Trying out made-up sentences")
print()

print("made up sentence 3")
print(whosaid.prob_classify(gen_feats(madeup_3_t)).prob('austen'), "P(A)")
print(whosaid.prob_classify(gen_feats(madeup_3_t)).prob('melville'), "P(M)")
print("------")
print("made up sentence 4")
print(whosaid.prob_classify(gen_feats(madeup_4_t)).prob('austen'), "P(A)")
print(whosaid.prob_classify(gen_feats(madeup_4_t)).prob('melville'),  "P(M)")
print("------")
print("made up sentence 5")
print(whosaid.prob_classify(gen_feats(madeup_5_t)).prob('austen'), "P(A)")
print(whosaid.prob_classify(gen_feats(madeup_5_t)).prob('melville'),  "P(M)")
print("----------------------------------------------------------")

print("Part B Number 7: Base Probabilities")
print()

def sent_counter(author, sents):
    counter = 0
    for sent in sents:
        if (sent[1] == author):
            counter += 1
    return counter

print("Number of Melville sentences in train_sents")
print(sent_counter('melville', train_sents))
print("Number of Austen sentences in train_sents")
print(sent_counter('austen', train_sents))
print("---------------------------------------------------------")

print("Part B Number 8: Calculating odds Ratio")
print()

def very_counter(author, sents):
    counter = 0
    for sent in sents:
        if ((sent[1] == author) and (("very" in sent[0]) or ("Very" in sent[0]))):
            counter += 1
    return counter


print("Number of Austen sentences w/ 'very' ")
print(very_counter('austen', train_sents))

print("Number of Melville sentences w/ 'very'")
print(very_counter("melville", train_sents))

print("----------------------------------------------")

print("Part B Number 9: Feature weights in model")

print(whosaid.feature_weights('contains-very', 1))

print("-----------------------------------------------")

print("Part B Number 10: Zero Counts and feature weights")
print("feature weight for 'whale")
print(whosaid.feature_weights('contains-whale', 1))
print("feature weight for 'ahab'")
print(whosaid.feature_weights('contains-ahab', 1))
print()
print("feature weight for 'marriage'")
print(whosaid.feature_weights('contains-marriage', 1))
print("feature weight for 'Emma'")
print(whosaid.feature_weights('contains-emma', 1))
print()
print("feature weight for 'woodhouse' (only in austen's text)")
print(whosaid.feature_weights('contains-woodhouse', 1))
print("feature weight for 'boat' (only in melville's text)")
print(whosaid.feature_weights('contains-boat', 1))
print()
print("feature weight for 'cautiously' (1 time in both texts)")
print(whosaid.feature_weights('contains-cautiously', 1))

print()
#print("feature weight for 'internet'")
#print(whosaid.feature_weights('contains-internet', 1))

print("-------------")
print("Part B Number 10.f. ")
sent6 = "She hates the internet"
sent7 = "She hates the"
sent6_t = nltk.word_tokenize(sent6)
sent7_t = nltk.word_tokenize(sent7)
print(whosaid.prob_classify(gen_feats(sent6_t)).prob('austen'), "P(A)")
print(whosaid.prob_classify(gen_feats(sent7_t)).prob('austen'), "P(A)")

print("-------------------------------------------------")
print("Part B Number 11")

print(whosaid.feature_weights('contains-he', 1))
print(whosaid.feature_weights('contains-knows', 1))
print(whosaid.feature_weights('contains-the', 1))
print(whosaid.feature_weights('contains-truth', 1))

print("---------------------------------------------------")
print("Part B Number 12")

print(len(am), 'am')
print(len(ma), 'ma')
print(len(mm), 'mm')
print(len(aa), 'aa')

print("----------------------------------------------------")
print("Part B Number 13: Error Analysis ")
print("-----")
print("mis-classified austen sentences in dev-test:")
print()
for x in am:
    print(' '.join(x[2]))
print("-------")
print("confidence of mis-labeled dev-test sentences that melville wrote them when it was actually austen:")
for x in am:
    print(whosaid.prob_classify(gen_feats(x[2])).prob('melville'), "P(M)")
print("--------")
print("Lowest followed by Highest Confidence Wrong Melville guess sents in dev-test:")
print(' '.join(am[1][2]))
print(' '.join(am[8][2]))

print()
print("End Part B Analysis")
print("-----------------------------------------------------")



