import json
import jsonlines

if __name__=='__main__':
    # with open("/data/mxy/Finstruction/data/code/commitpackft/data/bro/data.jsonl",'r', encoding="utf-8") as f:
    #     for line in f:
    #         data = json.loads(line)
    #         print(data)

    with open("/data/mxy/Finstruction/data/code/commitpackft/data/bro/data.jsonl", "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # print(item['old_contents'] + '\n' +item['subject'])
            print(idx)