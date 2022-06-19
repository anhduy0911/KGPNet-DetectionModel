import os
import json
import re
import jellyfish
from pip import main
import numpy as np
import config as CFG

BASE = CFG.prescription_folder

paths = os.listdir(BASE)

def get_info(filename):
    with open(os.path.join(BASE, filename), encoding= 'utf8') as fr:
        data = json.load(fr)

    result = {}
    result['id'] = filename
    result['pills'] = []
    diag_str = ''
    visited = [0] * len(data)
    for idx, item in enumerate(data):
        if jellyfish.jaro_distance(item['text'], 'Chẩn đoán khác') > 0.8:
            visited[idx] = 1
        if item['label'] == 'diagnose' and visited[idx] == 0:
            diag_str += ' ' + item['text']
        if item['label'] == 'other':
            continue
        if item['label'] == 'drugname' and visited[idx] == 0:
            check_end_pill= False
            pill = {}
            pill['usage'] = ''
            pill['name'] = re.sub(r"^[0-9]+['\- /]*[\)]\s*", '', item['text'])
            visited[idx] = 1
            pill['quantity'] = ''
            for i in range(len(data)):
                if data[i]['label'] == 'drugname' and visited[i] == 0 and i != idx:
                    break
                if not check_end_pill:
                    if data[i]['label'] == 'quantity' and visited[i] == 0:
                        pill['quantity'] = re.findall(r'\d+',data[i]['text'])[0]
                        visited[i] = 1
                if data[i]['label'] == 'usage' and visited[i] == 0:
                    pill['usage'] += data[i]['text'] + ' '
                    check_end_pill = True
                    visited[i] = 1
            # pill['usage'] = pill['usage'][:-1]
            result['pills'].append(pill)
    
    result['diagnose'] = []
    if jellyfish.jaro_distance(diag_str[:10], 'Chẩn đoán') < 0.8:
        diag_words = [st for st in diag_str.split(' ') if st.strip() != '']
        print(diag_words)
        scores = [jellyfish.jaro_distance(str_t, 'Chẩn đoán') for str_t in diag_words]
        print(scores)
        keyword = diag_words[np.argmax(scores)]
        
        indx = diag_str.find(keyword)
        diag_str = diag_str[indx:]
        print(diag_str)
    
    wrong_str = ['F00%','IIO','331','Ell', 'G46?', '110', '160', '170', '[25', '[10', '[20', '140', '1677','149','167t', '125', '150', '(((I10)', 'bàn I', '142', 'KS2', 'EI1', 'KO4', "167'", 'JII', 'j42', '120', 'El1', '142', '[J20', '[31', 'G46%', 'M1?', '(S33)Sai', 'E1i']
    true_str =  ['F00*','I10','J31','E11', 'G46*', 'I10', 'I60', 'I70', 'I25', 'I10', 'I20', 'J40','I67', 'I49', 'I67', 'I25', 'I50', '(I10)', 'bàn 1', 'J42', 'K52', 'E11', 'K04','I67','J11', 'J42', 'J20', 'E11', 'J42', 'J20', 'J31', 'G46*', 'M10', '(S33)', 'E11']
    for old, new in zip(wrong_str,true_str):
        diag_str = diag_str.replace(old, new)
    
    result['diagnose'] = [i.strip() for i in re.split(';|:' , diag_str) if jellyfish.jaro_distance(i.strip(), 'Chẩn đoán') < 0.8]
    print(result['diagnose'])
    result['diagnose'] = [i.split(' ')[0] for i in result['diagnose'] if re.match(r'.\w\d\d*', i)]
    print(result['diagnose'])
    
    # with open('data/prescription/'+ filename, 'w', encoding= 'utf-8') as f:
    #     json.dump(result, f, ensure_ascii= False, indent= 4)
    
    return result

if __name__ == '__main__':
    json_merge = []
    for path in paths:
        res = get_info(path)
        json_merge.append(res)
        
    with open('data/prescription/_merged_prescriptions.json', 'w', encoding= 'utf-8') as f:
        json.dump(json_merge, f, ensure_ascii= False, indent= 4)
    # res = get_info('20220104_233915445103.json')