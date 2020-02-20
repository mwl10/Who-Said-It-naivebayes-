 
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
sents = [a_sents + m_sents]      # EDIT

#------------------------------------------------ STEP 4 (COMPLETE)
print("4. Sentence stats:")
print(" # of total sentences:", len(sents))
print(" # of Austen sentences:", len(a_sents))
print(" # of Melville sentences:", len(m_sents))

#------------------------------------------------ STEP 5
print("5. Shuffling...")
# EDIT -- shuffle sents here
random.shuffle(sents)

#------------------------------------------------ STEP 6
print("6. Partitioning...")

test_sents = [sents[:1000]]    # EDIT
devtest_sents = [sents[1000:2000]]  # EDIT
train_sents = [sents[2000:]]    # EDIT

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
            if w in mainchars: w = 'MontyPython'
        featdict['contains-'+w.lower()] = 1
    return featdict

#------------------------------------------------ STEP 8
print("8. Generating feature sets...")
test_feats = [gen_feats(test_sents)]     # EDIT
devtest_feats = [gen_feats(devtest_feats)]  # EDIT
train_feats = [gen_feats(train_feats)]    # EDIT

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
    # EDIT below to populate mm, am, ma

#------------------------------------------------ STEP 12
print("12. Sample CORRECT and INCORRECT predictions from dev-test set:")
print("-------")
for x in (aa):  # EDIT change (aa) to (aa, mm, am, ma)
    auth, guess, sent = random.choice(x)
    print('REAL=%-8s GUESS=%-8s' % (auth, guess))  # string formatting
    print(' '.join(sent))
    print("-------")
print()

#------------------------------------------------ STEP 13
print("13. Looking up 40 most informative features...")
# EDIT -- use .show_most_informative_feats_all()
