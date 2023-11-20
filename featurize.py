'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')


# --------- Add your own feature methods ----------
def freq_dot_feature(text, freq):
    return text.count('.')


def freq_uline_feature(text, freq):
    return text.count('_')
def freq_line_feature(text, freq):
    return text.count('-')
def freq_dline_feature(text, freq):
    return text.count('/')

def freq_equ_feature(text, freq):
    return text.count('=')

def example_feature(text, freq):
    return int('example' in text)

def freq_cash_feature(text, freq):
    return float(freq['cash'])

def freq_html_feature(text, freq):
    return float(freq['today'])

def freq_guarantee_feature(text, freq):
    return float(freq['guarantee'])

def freq_loans_feature(text, freq):
    return float(freq['loans'])

def freq_debt_feature(text, freq):
    return float(freq['debt'])

def freq_call_feature(text, freq):
    return float(freq['call']) 
    
def freq_now_feature(text, freq):
    return float(freq['now'])
  
def freq_winner_feature(text, freq):
    return float(freq['winner'])


def freq_www_feature(text, freq):
    return float(freq['www']) 

def freq_offer_feature(text, freq):
    return float(freq['precise'])

def freq_junk_feature(text, freq):
    return float(freq['consolidate'])
def freq_body_feature(text, freq):
    return float(freq['ad'])
def freq_wild_feature(text, freq):
    return float(freq['refund'])
def freq_cock_feature(text, freq):
    return float(freq['full'])
def freq_off_feature(text, freq):
    return float(freq['treat'])   
def freq_call_feature(text, freq):
    return float(freq['specialties']) 
def freq_http_feature(text, freq):
    return float(freq['�'])
def freq_best_feature(text, freq):
    return float(freq['best'])
def freq_thanks_feature(text, freq):
    return float(freq['blood'])
def freq_click_feature(text, freq):
    return float(freq['worry']) 
def freq_orgasmic_feature(text, freq):
    return float(freq['no'])  
def freq_shot_feature(text, freq):
    return float(freq['style']) 
def freq_star_feature(text, freq):
    return text.count('only') 
def freq_cum_feature(text, freq):
    return float(freq['balancing']) 
def freq_br_feature(text, freq):
    return float(freq['holding'])  
def freq_owners_feature(text, freq):
    return float(freq['nothing'])
def freq_time_feature(text, freq):
    return float(freq['file'])
def freq_estimated_feature(text, freq):
    return float(freq['info'])
def freq_today_feature(text, freq):
    return float(freq['today']) 
def freq_slot_feature(text, freq):
    return float(freq['slot']) 
def freq_doctor_feature(text, freq):
    return float(freq['doctor'])
def freq_fuck_feature(text, freq):
    return float(freq['fuck']) 
def freq_warning_feature(text, freq): #
    return float(freq['warning'])  
def freq_market_feature(text, freq):
    return float(freq['market']) 
def freq_ect_feature(text, freq):
    return float(freq['ect'])

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
   # feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    #feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
   # feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))  
    #feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    #feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))  
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))  
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))
    feature.append(freq_br_feature(text, freq))

    feature.append(freq_record_feature(text, freq))
    feature.append(freq_width_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    feature.append(freq_best_feature(text, freq))
    feature.append(freq_market_feature(text, freq))
    feature.append(freq_warning_feature(text, freq))
    feature.append(freq_fuck_feature(text, freq))
    feature.append(freq_doctor_feature(text, freq)) 
    feature.append(freq_slot_feature(text, freq))
    feature.append(freq_today_feature(text, freq)) 
    feature.append(freq_estimated_feature(text, freq)) 
    feature.append(freq_time_feature(text, freq)) 
    feature.append(freq_owners_feature(text, freq))
    feature.append(freq_cum_feature(text, freq))  
    feature.append(freq_star_feature(text, freq))
    feature.append(freq_orgasmic_feature(text, freq))
    feature.append(freq_click_feature(text, freq))
    feature.append(freq_thanks_feature(text, freq))
    feature.append(freq_http_feature(text, freq))
    feature.append(freq_call_feature(text, freq))
    feature.append(freq_shot_feature(text, freq)) 
    feature.append(freq_off_feature(text, freq))
    feature.append(freq_cock_feature(text, freq)) 
    feature.append(freq_wild_feature(text, freq)) 
    feature.append(freq_body_feature(text, freq))
    feature.append(freq_junk_feature(text, freq)) 


    #feature.append(freq_offer_feature(text, freq))
    #feature.append(freq_dot_feature(text, freq))
    #feature.append(freq_equ_feature(text, freq))
    #feature.append(freq_cash_feature(text, freq))  
    #feature.append(freq_debt_feature(text, freq))
    #feature.append(freq_now_feature(text, freq))
    #feature.append(freq_winner_feature(text, freq))    
   # feature.append(freq_www_feature(text, freq))
    #feature.append(freq_click_feature(text, freq)) 
   
    #feature.append(freq_huge_feature(text, freq)) 
    # #feature.append(freq_monster_feature(text, freq))  
    
    #feature.append(freq_ect_feature(text, freq))
  


    return feature


# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
           
            design_matrix.append(feature_vector)
    return design_matrix



# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data-bow.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
