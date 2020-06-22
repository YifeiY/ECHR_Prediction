import os
import pickle
import numpy as np
import json


class DocumentAdapter():
  def __init__(self,DATA_DIR = 'DS', FORMAT = '.pkl'):
    print("DocumentAdapter assumes all relevant files are under DS/, all filename specified will be joint to this dir ")
    self.DATA_DIR = DATA_DIR
    self.FORMAT = FORMAT
    self.shelves = dict()
    self.exceptions = ['body',"VIOLATED_ARTICLES","VIOLATED_PARAGRAPHS","VIOLATED_BULLETPOINTS",
                       "NON_VIOLATED_ARTICLES","NON_VIOLATED_PARAGRAPHS","NON_VIOLATED_BULLETPOINTS","CONCLUSION"]
    print('use .load_shelf(train_anon) to load anonymized training data. For other usages, see doc_adapter.py')
    
  
  def make_shelf(self,partition,document_dicts,body = 'body',joining_literal = ' ',documents_name = None,remake = False):
    '''document_dicts: documents stored as dictoinary elements
    joining_literal: the char literal used to join the headings,
    documents_name: the name of the document collection
    remake: whether to remake the formated document storage'''
    
    # check if document archive already exists
    
    if not documents_name: documents_name = partition
    file_name = self.DATA_DIR +'/' + documents_name + self.FORMAT
    if remake and documents_name in os.listdir(self.DATA_DIR): os.remove(file_name)
    if documents_name in os.listdir(self.DATA_DIR): shelf = np.load(file_name)
    else:
      assert type(document_dicts) == dict
      shelf = dict() # shelf is a collection of document archives
      for k,document_dict in document_dicts.items():
        shelf[k] = dict()
        if 'body' not in document_dict.keys(): print('document',k,'is missing its body, or its body is specified with a different key than \'body\'')
        temp = joining_literal.join([str(_v) for (_k,_v) in document_dict.items() if _k not in self.exceptions])
        shelf[k]['headings'] = temp
        shelf[k]['body'] = document_dict['body']
        for e in self.exceptions:
          shelf[k][e] = document_dict[e] # preserve the targets
    pickle.dump(shelf,open(self.DATA_DIR + '/' + documents_name + self.FORMAT,'wb'))
    if partition: self.shelves[partition] = shelf
    print('loaded',documents_name,'to shelf partition',partition,'\tCurrent partitions:',','.join(str(k) for k in self.shelves.keys()))
  
  
  def load_shelf(self,shelf,partition =None):
    partition = str(shelf) if not partition else None
    shelf = self.DATA_DIR +'/' + shelf + self.FORMAT
    self.shelves[partition] = pickle.load(open(shelf,'rb'))
    print(partition,'shelf loaded')
    
  
  def make_shelf_from_raw(self,ECHR_RAW_DIR = '/ECHR_Dataset'):
    '''the directory should contain i folders, for example, [train,test,dev],
    within each folder, there should be a set of JSON files.
    A copy of the data can be downloaded at https://archive.org/download/ECHR-ACL2019/ECHR_Dataset.zip'''

    
    ECHR_RAW_DIR = self.DATA_DIR + ECHR_RAW_DIR
    partitions = {'train':'EN_train','test':'EN_test','dev':'EN_dev','train_anon':'EN_train_Anon','test_anon':'EN_test_Anon','dev_anon':'EN_dev_Anon',}
    
    def load_files(directory):
        raw_shelf = dict()
        for file in [ECHR_RAW_DIR + '/' + directory + '/' + file_name for file_name in os.listdir(ECHR_RAW_DIR +'/' + directory)]:
          temp = json.load(open(file))
          temp['body'] = temp['TEXT']
          del temp['TEXT']
          raw_shelf[file.split('/')[-1]] = temp
        return raw_shelf
        
    for (partition,directory) in partitions.items():
      self.make_shelf(partition,load_files(directory))
          
        
    
  def make_corpus(self):
    corpus_filename = self.DATA_DIR + '/corpus.txt'
    if 'corpus.txt' in os.listdir(self.DATA_DIR): os.remove(corpus_filename)
    with open(corpus_filename,'w',encoding='utf8') as corpus_file:
      for partition,shelf in self.shelves.items():
        print('making corpus for',partition,'partition')
        for doc in shelf.values():
          corpus_file.write(doc['headings']  + '\n'+'\n'.join(doc['body']) + "\n\n")

        

  def remake(self):
    print('remaking datasets and corpus file from raw data')
    self.make_shelf_from_raw()
    self.make_corpus()