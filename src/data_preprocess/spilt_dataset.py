import jsonlines
import os 

if __name__ == '__main__':
    path = "/data/mxy/Finstruction/data/code/commitpackft/data"
    folder_list = []


    for i,j,k in os.walk(path):
        # print(k)
        if k == []:
            continue
        folder_list.append(i)

    # folder_list.append("/data/mxy/Finstruction/data/code/commitpackft/data/c")

    for f in folder_list:
        with open(os.path.join(f,"data.jsonl"), "r+", encoding="utf8") as fp:
            with jsonlines.open(os.path.join(f,"train.jsonl"),mode='w') as writer_t:
                with jsonlines.open(os.path.join(f,"valid.jsonl"),mode='w') as writer_v:
                    cnt = 0 
                    for item in jsonlines.Reader(fp):
                        if cnt%20 == 0 :
                            writer_v.write(item)
                        else:
                            writer_t.write(item)
                        cnt +=1 