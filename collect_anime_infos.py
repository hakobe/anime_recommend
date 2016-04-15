import urllib.request
import json
import os.path
import time

SAVE_DIR = 'anime_infos'
SAVE_FILE = 'eval_anime_save.json'

def path_of(tid):
    return SAVE_DIR + '/' + tid + '.json'

def get_anime_info(tid):
    url = "http://cal.syoboi.jp/json.php?Req=TitleFull&TID=%s" % tid
    req = urllib.request.Request(url, data=None, headers={'accept': '*/*'})
    with urllib.request.urlopen(req) as res:
        anime_info = json.loads(res.read().decode('utf-8'))['Titles'][tid]
    return anime_info

def save_anime_info(tid, anime_info):
    with open(path_of(tid), 'w') as f:
        json.dump(anime_info, f)

def main():
    with open(SAVE_FILE, 'r') as f:
        saved = json.load(f)

    for tid in sorted(saved.keys()):
        print(tid)
        if os.path.exists(path_of(tid)):
            print('skip')
            continue
        time.sleep(0.3)
        anime_info = get_anime_info(tid)
        save_anime_info(tid, anime_info)

    url = 'http://saisoku.douzemille.net/a/2016/spring.json'
    with urllib.request.urlopen(url) as res:
        animes = json.loads(res.read().decode('utf-8'))['animes']
        for anime in animes:
            tid = anime['tid']
            print(tid)
            if os.path.exists(path_of(tid)):
                print('skip %s' % tid)
                continue
            print(tid)
            anime_info = get_anime_info(tid)
            save_anime_info(tid, anime_info)

main()
