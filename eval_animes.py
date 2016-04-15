import urllib.request
import json
import signal, sys

SAVE_DIR = 'anime_infos'
SAVE_FILE = 'eval_anime_save.json'

saved = {}
try:
    with open(SAVE_FILE, 'r') as f:
        saved = json.load(f)
except FileNotFoundError:
    pass

def save():
    print(json.dumps(results))
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f)
    sys.exit(0)

signal.signal(signal.SIGINT, lambda s, f: save())

results = saved

for year in [2013, 2014, 2015]:
    for season in ['winter', 'spring', 'summer', 'fall']:
        url = 'http://saisoku.douzemille.net/a/%s/%s.json' % (year, season)
        print(url)
        with urllib.request.urlopen(url) as res:
            animes = json.loads(res.read().decode('utf-8'))['animes']
            for anime in animes:
                tid = anime['tid']
                title = anime['title']
                if tid in results:
                    continue
                print("%s\t%s [y/n]" % (tid, title))
                ans = input()
                if ans == 'y':
                    results[tid] = 1
                else:
                    results[tid] = 0

save()
