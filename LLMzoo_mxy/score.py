# promote_0 = "We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following."
# promote_1 = "\n\n instruction:"
# promote_2 = " response: \n\n"
# promote_3 = "\nPlescp ase rate according to the accuracy of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the accuracy. Please output a line that represents the value of the score and does not require additional information to be output."
#
# promote_total_all = promote_0 + promote_1 + promote_2
# import json
#
# res = []
# f_w_0 = open('/data/llx/ParroT/test/test_score_05_new.txt', 'w')
# f_w_1 = open('/data/llx/ParroT/test/test_score_15_new.txt', 'w')
# f_w_2 = open('/data/llx/ParroT/test/test_score_25_new.txt', 'w')
# f_w_3 = open('/data/llx/ParroT/test/test_score_35_new.txt', 'w')
# #
# f = open('/data/llx/LLMzoo/apaca_gpt4/apaca_gpt4.json', 'r')
# f = json.load(f)
# for index, item in enumerate(f):
#     Instruction = item['conversations'][0]['value']
#     Response = item['conversations'][1]['value']
#     promote_total_all = promote_0 + promote_1 + Instruction + promote_2 + Response + promote_3
#     promote_total_all = promote_total_all.replace('\n', ' ')
#     promote_total_all = promote_total_all + '\n'
#     res.append(promote_total_all)
#     res.append(promote_total_all)
#     res.append(promote_total_all)
#     res.append(promote_total_all)
#     res.append(promote_total_all)
# import random
#
# random.shuffle(res)
#
# for index in range(len(res)):
#     if index % 4 == 0:
#         f_w_0.writelines(res[index])
#     if index % 4 == 1:
#         f_w_1.writelines(res[index])
#     if index % 4 == 2:
#         f_w_2.writelines(res[index])
#     if index % 4 == 3:
#         f_w_3.writelines(res[index])
#
#
# f_w_0.close()
# f_w_1.close()
# f_w_2.close()
# f_w_3.close()
# num = 0
# f = open('/data/llx/ParroT/test/test_score.general-task.txt','r')
# for item in f :
#     if item == '### Response:\n':
#         num += 1
# print(num)
# flag = False
#

#
import re

score_dict = {}
import json

f = open('/data/llx/LLMzoo/apaca_gpt4/apaca_gpt4.json', 'r')
f = json.load(f)
for item in f:
    title = item['conversations'][0]['value']
    title = title.replace('\n', ' ')
    score_dict[title] = []
print('okk')


def extract_floats(s):
    return re.findall(r'(\d+\.\d+|\d+)', s)


def check(list_a):
    if len(list_a) > 2 or len(list_a) == 0:
        return False
    return True


def acc_score(list_a):
    final_res = 0
    if len(list_a) == 1:
        final_res = float(list_a[0])
    if len(list_a) == 2:
        a = float(list_a[0])
        b = float(list_a[1])
        if a > b:
            final_res = b
        else:
            final_res = a
    if final_res > 5.0:
        return 0
    else:
        return final_res


def not_acc_score(list_a):
    if len(list_a) == 0:
        return 0
    last_1 = float(list_a[len(list_a) - 1])
    last_2 = float(list_a[len(list_a) - 2])
    if float(last_1) == 5.0:
        if float(last_2) == 5.0 or float(last_2) < 5.0:
            return last_2
        else:
            return last_1
    if float(last_1) < 5.0:
        return last_1
    return 0


problem_key = []
tmp_title = ''
flag = False
res = ''
# f = open('/data/llx/ParroT/test/llama2_13b_test_score_all_test.text', 'r')
# f = open('/data/llx/ParroT/test/again_llama2_13b_test_score_all_test.text', 'r')
f = open('/data/llx/ParroT/test/score_all_5times_new_test.text', 'r')
total_num = 0
useful_num = 0
for item in f:
    if item == '\n':
        continue
    if '### Response' in item:
        flag = True
    if 'titleasdfghjk' in item and flag:
        sentence_score = 0
        score = extract_floats(res)
        if 'I\'m sorry' not in res:
            use = check(score)
            if not use:
                sentence_score = not_acc_score(score)
                useful_num += 1
            else:
                sentence_score = acc_score(score)
                useful_num += 1
        else:
            # useful_num += 1
            import random

            sentence_score = random.choice([float(0.0), float(1.0), float(2.0), float(3.0), float(4.0), float(5.0)])
        # print(sentence_score)
        if tmp_title in score_dict.keys():
            score_dict[tmp_title].append(sentence_score)
        else:
            # print(tmp_title)
            problem_key.append(tmp_title)
        total_num = total_num + 1
        res = ''
        flag = False

    if flag:
        res = res + item
    if 'titleasdfghjk:' in item:
        # print(item[item.find('instruction:') + 13:item.find('\\s response:') - 1])
        # tmp_title = item[item.find('instruction:') + 12:item.find('response:') - 1]
        tmp_title = item.split('titleasdfghjk:')[1]
        total_tmp_title = item
print(total_num, useful_num, useful_num / total_num)

problem_key_2 = []
for item in score_dict:
    if len(score_dict[item]) != 5:
        # score_dict[item] = [0.0]
        problem_key_2.append(item)
        # score_dict[item].append(float(0.0))

# for item in score_dict:
#     if len(score_dict[item]) != 5:
#         # score_dict[item] = [0.0]
#         problem_key_2.append(item)

score_dict_sort = sorted(score_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

score_dict_avg = {}

score_dict_variance = {}

import numpy as np

for item in score_dict:
    score_dict_avg[item] = np.mean(score_dict[item])

for item in score_dict:
    score_dict_variance[item] = np.std(score_dict[item])

score_dict_avg_sort = sorted(score_dict_avg.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

score_dict_finall = {}

for item in score_dict_avg:
    title = item
    score = score_dict_avg[item]
    finall_score = score / (1 + 0.2 * score_dict_variance[title])
    score_dict_finall[title] = finall_score

a_00 = 0
a_4 = 0
a_3 = 0
a_2 = 0
a_1 = 0
a_0 = 0  ## 左开右闭
#
# import random
# score_dict_finall = {}
# for item in score_dict:
#     score_dict_finall[item] = random.choice(score_dict[item])


for item in score_dict_finall:
    # score_tmp = float(score_dict_avg[item]) / score_dict_variance[item]
    score_tmp = score_dict_finall[item]
    # print(score_tmxp)
    if score_tmp > float(4):
        a_4 += 1
    elif score_tmp > float(3):
        # title_set.add(item[0])
        a_3 += 1
    elif score_tmp > float(2):
        # title_set.add(item[0])
        a_2 += 1
    elif score_tmp > float(1):
        # title_set.add(item[0])
        a_1 += 1
    elif score_tmp > float(0):
        a_0 += 1
    else:
        a_00 += 1

score_dict_finall_sort = sorted(score_dict_finall.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

import random

#
# random.shuffle(score_dict_finall_sort)
# random.shuffle(score_dict_finall_sort)

title_set = set()
num = 1



# for item in score_dict_finall_sort:
#     title_set.add(item[0])
#     num += 1
#     if num > 10400:
#         break


for item in score_dict_finall_sort:
    title_set.add(item[0])
    num += 1
    if num > 41600:
        break

print('a_4:', a_4, a_4 / 52002)
print('a_3:', a_3, a_3 / 52002)
print('a_2:', a_2, a_2 / 52002)
print('a_1:', a_1, a_1 / 52002)
print('a_0:', a_0, a_0 / 52002)
print('a_00:', a_00, a_00 / 52002)

finall_dataset_9k_20k = []
import json

f = open('/data/llx/LLMzoo/apaca_gpt4/apaca_gpt4_20per_random.json', 'r')
f = json.load(f)
num = 0
for item in f:
    title = item['conversations'][0]['value']
    title = title.replace('\n', ' ')
    if title in title_set:
        num += 1
        finall_dataset_9k_20k.append(item)
# #
f_w = open('/data/llx/LLMzoo/apaca_gpt4/llama13b_apaca_gpt4_80per_5times_avg_std.json', 'w')
json.dump(finall_dataset_9k_20k, f_w)
f_w.close()
print('okk', num)

# f = open('/data/llx/ParroT/test/again_llama2_13b_test_score_all_test.text', 'w')
# f_0 = open('/data/llx/ParroT/test/again_llama2_13b_test_score_05_new_gen_test.txt', 'r')
# f_1 = open('/data/llx/ParroT/test/again_llama2_13b_test_score_15_new_gen_test.txt', 'r')
# f_2 = open('/data/llx/ParroT/test/again_llama2_13b_test_score_25_new_gen_test.txt', 'r')
# f_3 = open('/data/llx/ParroT/test/again_llama2_13b_test_score_35_new_gen_test.txt', 'r')
# #
# for item in f_0:
#     f.writelines(item)
# for item in f_1:
#     f.writelines(item)
# for item in f_2:
#     f.writelines(item)
# for item in f_3:
#     f.writelines(item)
# f.close()

import json

# promte_1 = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n'
# promte_2 = '\n\n### Response:'
#
# f = open('/data/llx/ParroT/data/data_alpaca_gpt4_hf_en.json', 'r')
#
# for item in f:
#     item = eval(item)
#     print('okk')

# tmp_list = []
# f_p = open('/data/llx/LLMzoo/dolly/databricks-dolly-15k.jsonl', 'r')
# for item in f_p:
#     item = eval(item)
#     tmp_list.append(item)
#
# final_res_list = []
#
# for item in tmp_list:
#     instruction = item['instruction']
#     response = item['response']
#     tmp_dict = {}
#     tmp_dict['instruction'] = instruction
#     tmp_dict['output'] = response
#     # tmp_dict['prefix'] = promte_1 + instruction + promte_2
#     final_res_list.append(tmp_dict)
# print('okk')
#
# fo = open('/data/llx/ParroT/data/train_alp_dolly-15k.json', 'w')
#
# json.dump(final_res_list, fo, ensure_ascii=False, indent=4)
#
# fo.close()
