
import json
import os
import re
import itertools
import functools
import urllib.request
import math

SAVE_DIR = 'anime_infos'
SAVE_FILE = 'eval_anime_save.json'
FEATURE_FILE = 'feature.tsv'
ANSWER_FILE = 'answer.tsv'
TARGET_FILE = 'target.tsv'

def load_anime_info(fname):
    with open(fname, 'r') as f:
        anime_info = json.load(f)
    return anime_info

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)

def extract_names(txt):
    lines = re.split(r'\r?\n', txt)
    return [ re.split(r'\:', line)[1:3] for line in lines if line ]

def retrieve_features(anime_info):
    comment = anime_info['Comment']
    splitter = re.compile(r'^\*(.*?)\r\n', re.MULTILINE | re.DOTALL)
    heads_and_contents = splitter.split(comment)
    heads_and_contents.pop(0)

    features = []
    for k, v in grouper(2, heads_and_contents):
        if k == 'スタッフ' or k == 'キャスト':
            names = extract_names(v)

            for name in names:
                if len(name) >= 2:
                    for n in re.split(r'、', name[1]):
                        n = re.sub(r'\(.*?\)', '', n)
                        features.append(n)
    return features

def main():

    with open(SAVE_FILE, 'r') as f:
        answers = json.load(f)

    target_tids = []
    url = 'http://saisoku.douzemille.net/a/2016/spring.json'
    with urllib.request.urlopen(url) as res:
        animes = json.loads(res.read().decode('utf-8'))['animes']
        for anime in animes:
            tid = anime['tid']
            target_tids.append(tid)

    anime_features = []
    all_features = {}
    for file in os.listdir(SAVE_DIR):
        if file == '.keep':
            continue
        anime_info = load_anime_info(SAVE_DIR + '/' + file)
        if not ( anime_info['TID'] in answers or anime_info['TID'] in target_tids ):
            continue

        features = retrieve_features(anime_info)

        for f in features:
            c = all_features.get(f, 0)
            all_features[f] = c + 1

        anime_features.append({
            'tid': anime_info['TID'],
            'features': features,
        })

    # 指定回数以上登場している特徴のみ採用する
    features_base = [ f for f in all_features if all_features[f] >= 5]
    features_base.sort()


    print(len(answers))
    print(len(anime_features))
    print(target_tids)

    pos_ans_n = len([ a for a in answers.values() if a == 1])
    neg_ans_n = len(answers) - pos_ans_n
    pos_weight = math.ceil(neg_ans_n / pos_ans_n)
    print(pos_weight)

    with open(FEATURE_FILE, 'w') as ff, open(ANSWER_FILE, 'w') as af, open(TARGET_FILE, 'w') as tf:
        for anime_feature in sorted(anime_features, key=lambda x: int(x['tid'])):

            has_feature = functools.reduce( lambda h, f: h.update({f: 1}) or h, anime_feature['features'], {})
            feature_vector = [has_feature.get(f, 0) for f in features_base]

            if anime_feature['tid'] in answers:
                a = answers[anime_feature['tid']]
                for i in range(pos_weight if a == 1 else 1):
                    print(str(a), file=af)
                    print("\t".join([str(x) for x in feature_vector]), file=ff)
            elif anime_feature['tid'] in target_tids:
                row = [anime_feature['tid']]
                row.extend(feature_vector)
                print("\t".join([str(x) for x in row]), file=tf)

            print(anime_feature['tid'], [ features_base[i] for i in range(len(features_base)) if has_feature.get(features_base[i], 0)])

    print(len(features_base))


main()
