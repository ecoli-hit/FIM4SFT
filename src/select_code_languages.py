import json

if __name__=='__main__':
    language_lines = dict()
    with open("/data/mxy/Finstruction/data/code/commitpackft/line_count.txt",'r') as f :
        lines = f.readlines()
        for line in lines: 
            split = line.split(' ')
            language_lines[split[0]] = int(split[1])

    # print(sorted(language_lines.items(),key=lambda d:d[1]))

    pick_languages = dict()
    for k,v in language_lines.items():
        if v > 1000:
            pick_languages[k] = v
    dp = sorted(pick_languages.items(),key=lambda d:d[1],reverse=True)
    print(dp)

    json.dump(dp,open("/data/mxy/Finstruction/pick_code_languages.json",'w'))
            