from __future__ import print_function
import pandas
import spacy
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
import numpy

encoder = spacy.load('en_core_web_sm')

'''Load the training dataset'''
train_stories = pandas.read_csv(r'C:\Users\admin\Documents\GitHub\keras-rnn-notebooks\language_modeling\dataset\example_train_stories.csv', encoding='utf-8')


'''Split texts into lists of words (tokens)'''
def text_to_tokens(text_seqs):
    token_seqs = [[word.lower_ for word in encoder(text_seq)] for text_seq in text_seqs]
    return token_seqs
train_stories['Tokenized_Story'] = text_to_tokens(train_stories['Story'])


'''Count tokens (words) in texts and add them to the lexicon'''
def make_lexicon(token_seqs, min_freq=1):
    token_counts = {}
    for seq in token_seqs:
        for token in seq:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

    lexicon = [token for token, count in token_counts.items() if count >= min_freq]
    lexicon = {token:idx + 2 for idx,token in enumerate(lexicon)}
    lexicon['<UNK>'] = 1
    lexicon_size = len(lexicon)

    #print("LEXICON SAMPLE ({} total items):".format(len(lexicon)))
    #print(dict(list(lexicon.items())[:20]))

    return lexicon

lexicon = make_lexicon(token_seqs=train_stories['Tokenized_Story'], min_freq=1)

with open(r'C:\Users\admin\Documents\GitHub\keras-rnn-notebooks\language_modeling\example_model\lexicon.pkl', 'wb') as f: 
    pickle.dump(lexicon, f)


'''Make a dictionary where the string representation of a lexicon item can be retrieved from its numerical index'''
def get_lexicon_lookup(lexicon):
    lexicon_lookup = {idx: lexicon_item for lexicon_item, idx in lexicon.items()}
    lexicon_lookup[0] = ""
    print("LEXICON LOOKUP SAMPLE:")
    print(dict(list(lexicon_lookup.items())[:20]))
    return lexicon_lookup

lexicon_lookup = get_lexicon_lookup(lexicon)


'''Convert each text from a list of tokens to a list of numbers (indices)'''
def tokens_to_idxs(token_seqs, lexicon):
    idx_seqs = [[lexicon[token] if token in lexicon else lexicon['<UNK>'] for token in token_seq] for token_seq in token_seqs]
    return idx_seqs

train_stories['Story_Idxs'] = tokens_to_idxs(token_seqs=train_stories['Tokenized_Story'], lexicon=lexicon)                      
#train_stories[['Tokenized_Story', 'Story_Idxs']][:10]


'''create a padded matrix of stories'''
def pad_idx_seqs(idx_seqs, max_seq_len): 
    padded_idxs = pad_sequences(sequences=idx_seqs, maxlen=max_seq_len)
    return padded_idxs

max_seq_len = max([len(idx_seq) for idx_seq in train_stories['Story_Idxs']])

train_padded_idxs = pad_sequences(train_stories['Story_Idxs'], maxlen=max_seq_len + 1)
print(train_padded_idxs)
print("SHAPE:", train_padded_idxs.shape)

pandas.DataFrame(list(zip(["-"] + train_stories['Tokenized_Story'].loc[0], train_stories['Tokenized_Story'].loc[0])), columns=['Input Word', 'Output Word'])

print(pandas.DataFrame(list(zip(train_padded_idxs[0,:-1], train_padded_idxs[0, 1:])), columns=['Input Words', 'Output Words']))


def create_model(seq_input_len, n_input_nodes, n_embedding_nodes, n_hidden_nodes, stateful=False, batch_size=None):
    
    # Layer 1
    input_layer = Input(batch_shape=(batch_size, seq_input_len), name='input_layer')

    # Layer 2
    embedding_layer = Embedding(input_dim=n_input_nodes, output_dim=n_embedding_nodes, mask_zero=True, name='embedding_layer')(input_layer) #mask_zero=True will ignore padding
    # Output shape = (batch_size, seq_input_len, n_embedding_nodes)

    #Layer 3
    gru_layer1 = GRU(n_hidden_nodes, return_sequences=True, stateful=stateful, name='hidden_layer1')(embedding_layer)
    # Output shape = (batch_size, seq_input_len, n_hidden_nodes)

    #Layer 4
    gru_layer2 = GRU(n_hidden_nodes, return_sequences=True, stateful=stateful, name='hidden_layer2')(gru_layer1)
    # Output shape = (batch_size, seq_input_len, n_hidden_nodes)

    #Layer 5
    output_layer = TimeDistributed(Dense(n_input_nodes, activation="softmax"), name='output_layer')(gru_layer2)
    # Output shape = (batch_size, seq_input_len, n_input_nodes)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    #Specify loss function and optimization algorithm, compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam')
    
    return model



model = create_model(seq_input_len=train_padded_idxs.shape[-1] - 1, n_input_nodes = len(lexicon) + 1, n_embedding_nodes = 300, n_hidden_nodes = 500)


'''Train the model'''
model.fit(x=train_padded_idxs[:,:-1], y=train_padded_idxs[:, 1:, None], epochs=5, batch_size=20)
model.save_weights(r'C:\Users\admin\Documents\GitHub\keras-rnn-notebooks\language_modeling\example_model\model_weights.h5')



'''Load test set and apply same processing used for training stories'''
test_stories = pandas.read_csv(r'C:\Users\admin\Documents\GitHub\keras-rnn-notebooks\language_modeling\dataset\example_test_stories.csv', encoding='utf-8')
test_stories['Tokenized_Story'] = text_to_tokens(test_stories['Story'])
lexicon2 = make_lexicon(token_seqs=test_stories['Tokenized_Story'], min_freq=1)
test_stories['Story_Idxs'] = tokens_to_idxs(token_seqs=test_stories['Tokenized_Story'], lexicon=lexicon2)
test_padded_idxs = pad_sequences(test_stories['Story_Idxs'], maxlen=max_seq_len + 1)
print(test_stories['Story_Idxs'])

perplexity = numpy.exp(model.evaluate(x=test_padded_idxs[:,:-1], y=test_padded_idxs[:, 1:, None]))
print("PERPLEXITY ON TEST SET: {:.3f}".format(perplexity))


'''Create a new test model, setting batch_size = 1, seq_input_len = 1, and stateful = True'''
with open(r'C:\Users\admin\Documents\GitHub\keras-rnn-notebooks\language_modeling\pretrained_model\lexicon.pkl', 'rb') as f:
    lexicon = pickle.load(f)
lexicon_lookup = get_lexicon_lookup(lexicon)

predictor_model = create_model(seq_input_len=1, n_input_nodes=len(lexicon) + 1, n_embedding_nodes = 300, n_hidden_nodes = 500, stateful=True, batch_size = 1)

predictor_model.load_weights(r'C:\Users\admin\Documents\GitHub\keras-rnn-notebooks\language_modeling\pretrained_model\model_weights.h5') #Load weights from saved model


'''Re-encode the test stories with lexicon we just loaded'''
test_stories['Story_Idxs'] = tokens_to_idxs(token_seqs=test_stories['Tokenized_Story'], lexicon=lexicon2)


'''Compute the probability of a stories according to the language model'''
def get_probability(idx_seq):
    idx_seq = [0] + idx_seq
    probs = []
    for word, next_word in zip(idx_seq[:-1], idx_seq[1:]):
        p_next_word = predictor_model.predict(numpy.array(word)[None,None])[0,0] 
        p_next_word = p_next_word[next_word]
        probs.append(p_next_word)
    predictor_model.reset_states()
    return numpy.mean(probs) 

for _, test_story in test_stories[:10].iterrows():
    len_initial_story = len([word for sent in list(encoder(test_story['Story']).sents)[:-1] for word in sent])
    token_initial_story = test_story['Tokenized_Story'][:len_initial_story]
    idx_initial_story = test_story['Story_Idxs'][:len_initial_story]
    token_ending = test_story['Tokenized_Story'][len_initial_story:]
    
    rand_story = test_stories.loc[numpy.random.choice(len(test_stories))]
    len_rand_ending = len(list(encoder(rand_story['Story']).sents)[-1])
    token_rand_ending = rand_story['Tokenized_Story'][-len_rand_ending:]
    idx_rand_ending = rand_story['Story_Idxs'][-len_rand_ending:]

    # print("INITIAL STORY:", " ".join(token_initial_story))
    # prob_given_ending = get_probability(test_story['Story_Idxs'])
    # print("GIVEN ENDING: {} (P = {:.3f})".format(" ".join(token_ending), prob_given_ending))

    # prob_rand_ending = get_probability(idx_initial_story + idx_rand_ending)
    # print("RANDOM ENDING: {} (P = {:.3f})".format(" ".join(token_rand_ending), prob_rand_ending), "\n")
    
    print("\nINITIAL STORY:", " ".join(token_initial_story))
    # prob_given_ending = get_probability(test_story['Story_Idxs'])
    # print("GIVEN ENDING: {} (P = {:.3f})".format(" ".join(token_ending), prob_given_ending))
    print("GIVEN ENDING: ", format(" ".join(token_ending)))

    # print("PROBABILITY:", get_probability(test_story['Story_Idxs']))
    # prob_rand_ending = get_probability(idx_initial_story + idx_rand_ending)
    # print("RANDOM ENDING: {} (P = {:.3f})".format(" ".join(token_rand_ending), prob_rand_ending), "\n")
    print("RANDOM ENDING: ", format(" ".join(token_rand_ending)))
