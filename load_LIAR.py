import pandas as pd
import numpy as np
import h5py

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import tensorflow.keras as keras


train_data = pd.read_table('train.tsv', names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
                                            "barely-true", "false", "half-true", "mostly-true", "pants-fire", "venue"])

val_data = pd.read_table('valid.tsv', names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
                                            "barely-true", "false", "half-true", "mostly-true", "pants-fire", "venue"])

test_data = pd.read_table('test.tsv', names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
                                            "barely-true", "false", "half-true", "mostly-true", "pants-fire", "venue"])

print(train_data.shape, val_data.shape, test_data.shape)
print(train_data.label.unique())
train_data.head()

#process label
y_label_dict = {"pants-fire" : 0, "false" : 1, "barely-true" : 2, "half-true" : 3, "mostly-true" : 4, "true" : 5}
print(y_label_dict)

train_data['output'] = train_data['label'].apply(lambda x: y_label_dict[x])
val_data['output'] = val_data['label'].apply(lambda x: y_label_dict[x])
test_data['output'] = test_data['label'].apply(lambda x: y_label_dict[x])

num_classes = 6

#process job
#frequent_jobs = train_data['job'].str.lower().value_counts()[:20].reset_index().to_dict()['index']
#frequent_jobs = dict((v,k) for k,v in frequent_jobs.items())
frequent_jobs = { 'senator' : 0, 'president' : 1, 'governor' : 2, 
                 'u.s. representative' : 3, 'attorney' : 4, 'congressman' : 5, 
                 'congresswoman' : 5, 'social media posting' : 6, 'lawyer' : 4, 
                 'businessman' : 6,  'radio host' : 8, 'host':8,
                  'mayor' : 7, 'assembly' : 9,'representative' : 3, 
                 'senate' : 10,'state representatives' : 10,'milwaukee county executive' : 11,
                 'u.s. house of representatives' : 3,'house representatives' : 3,
                 'house of representatives' : 3,'house member':3}

def get_job_id(job):
  if isinstance(job, str):
    matched = [jb for jb in frequent_jobs if jb in job.lower() ]
    if len(matched)>0:
      return frequent_jobs[matched[0]]
    else:
      return len(set(frequent_jobs.values()))
  else:
    return len(set(frequent_jobs.values()))
  

train_data['job_id'] = train_data['job'].apply(get_job_id)
val_data['job_id'] = val_data['job'].apply(get_job_id)
test_data['job_id'] = test_data['job'].apply(get_job_id)

train_data['job_id'].value_counts()

#process party
frequent_parties = train_data['party'].str.lower().value_counts()[:5].reset_index().to_dict()['index']
frequent_parties = dict((v,k) for k,v in frequent_parties.items())
#frequent_parties['columnist']=frequent_parties['journalist']
#frequent_parties['talk-show-host']=frequent_parties['journalist']
def get_party_id(party):
  if isinstance(party, str):
    matched = [pt for pt in frequent_parties if pt in party.lower() ]
    if len(matched)>0:
      return frequent_parties[matched[0]]
    else:
      return len(set(frequent_parties.values())) 
  else:
    return len(set(frequent_parties.values())) 
  

train_data['party_id'] = train_data['party'].apply(get_party_id)
val_data['party_id'] = val_data['party'].apply(get_party_id)
test_data['party_id'] = test_data['party'].apply(get_party_id)

train_data['party_id'].value_counts()

#process state
other_states = ['wyoming', 'colorado', 'hawaii', 'tennessee', 'nevada', 'maine',
                'north dakota', 'mississippi', 'south dakota', 'oklahoma', 
                'delaware', 'minnesota', 'north carolina', 'arkansas', 'indiana', 
                'maryland', 'louisiana', 'idaho', 'iowa', 'west virginia', 
                'michigan', 'kansas', 'utah', 'connecticut', 'montana', 'vermont', 
                'pennsylvania', 'alaska', 'kentucky', 'nebraska', 'new hampshire', 
                'missouri', 'south carolina', 'alabama', 'new mexico']


frequent_states = {'texas': 1, 'florida': 2, 'wisconsin': 3, 'new york': 4, 
                    'illinois': 5, 'ohio': 6, 'georgia': 7, 'virginia': 8, 
                   'rhode island': 9, 'oregon': 10, 'new jersey': 11, 
                   'massachusetts': 12, 'arizona': 13, 'california': 14, 
                   'washington': 15}
for state in other_states:
  frequent_states[state]=0

def get_state_id(state):
    if isinstance(state, str):
        if state.lower().rstrip() in frequent_states:
            return frequent_states[state.lower().rstrip()]
        else:
            if 'washington' in state.lower():
                return frequent_states['washington']
            else:
                return len(set(frequent_states.values()))
    else:
        return len(set(frequent_states.values()))


train_data['state_id'] = train_data['state'].apply(get_state_id)
val_data['state_id'] = val_data['state'].apply(get_state_id)
test_data['state_id'] = test_data['state'].apply(get_state_id)

train_data['state_id'].value_counts()

#embedding
def load_statement_vocab_dict(train_data):
    vocabulary_dict = {}
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['statement'])
    vocabulary_dict = tokenizer.word_index
    return vocabulary_dict

def preprocess_statement(statement):
  statement = [w for w in statement.split(' ') if w not in stopwords.words('english')]
  statement = ' '.join(statement)
  text = text_to_word_sequence(statement)
  val = [0] * 10
  val = [vocabulary_dict[t] for t in text if t in vocabulary_dict] 
  return val

vocabulary_dict = load_statement_vocab_dict(train_data)
train_data['word_id'] = train_data['statement'].apply(preprocess_statement)
val_data['word_id'] = val_data['statement'].apply(preprocess_statement)
test_data['word_id'] = test_data['statement'].apply(preprocess_statement)

embeddings = {}
with open("glove.6B.100d.txt", encoding="utf8") as file_object:
  for line in file_object:
    word_embed = line.split()
    word = word_embed[0]
    embed = np.array(word_embed[1:], dtype="float32")
    embeddings[word.lower()]= embed

EMBED_DIM = 100

num_words = len(vocabulary_dict) + 1
embedding_matrix = np.zeros((num_words, EMBED_DIM))
for word, i in vocabulary_dict.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embeddings_index = None

X_train = train_data['word_id']
X_val = val_data['word_id']
X_test = test_data['word_id']

Y_train = train_data['output']
Y_train = keras.utils.to_categorical(Y_train, num_classes=6)

Y_val = val_data['output']
Y_val = keras.utils.to_categorical(Y_val, num_classes=6)

num_steps = 15
num_party = len(train_data.party_id.unique())
num_state = len(train_data.state_id.unique())
num_job = len(train_data.job_id.unique())

X_train = sequence.pad_sequences(X_train, maxlen=num_steps, padding='post',truncating='post')
X_val = sequence.pad_sequences(X_val, maxlen=num_steps, padding='post',truncating='post')
X_test = sequence.pad_sequences(X_test, maxlen=num_steps, padding='post',truncating='post')

#Meta data preparation
party_train = keras.utils.to_categorical(train_data['party_id'], num_classes=num_party)
state_train = keras.utils.to_categorical(train_data['state_id'], num_classes=num_state)
job_train = keras.utils.to_categorical(train_data['job_id'], num_classes=num_job)

#X_train_meta = party_train
X_train_meta = np.hstack((party_train, state_train, job_train))

party_val = keras.utils.to_categorical(val_data['party_id'], num_classes=num_party)
state_val = keras.utils.to_categorical(val_data['state_id'], num_classes=num_state)
job_val = keras.utils.to_categorical(val_data['job_id'], num_classes=num_job)

#X_val_meta = party_val
X_val_meta = np.hstack((party_val, state_val, job_val))

party_test = keras.utils.to_categorical(test_data['party_id'], num_classes=num_party)
state_test = keras.utils.to_categorical(test_data['state_id'], num_classes=num_state)
job_test = keras.utils.to_categorical(test_data['job_id'], num_classes=num_job)

#X_test_meta = party_test
X_test_meta = np.hstack((party_test, state_test, job_test))

filename = 'LIAR.hdf5'
with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(embedding_matrix)
    f['train'] = X_train
    f['train_label'] = Y_train
    f['train_meta'] = X_train_meta
    f['test'] = X_test
    f['test_label'] = test_data['output']
    f['test_meta'] = X_test_meta
    f['val'] = X_val
    f['val_label'] = Y_val
    f['val_meta'] = X_val_meta
    

